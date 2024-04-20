from pathlib import Path
import io

from diffusers import AutoencoderKL
from PIL import Image
import onnxruntime.capi
import onnxruntime.capi._pybind_state
import torch
from torch.nn.utils.parametrize import type_before_parametrizations
import numpy as np
import onnxruntime

def to_tensor(img: Image.Image | np.ndarray) -> torch.Tensor:
    """resize [0, 255] -> [-1, 1] and permute (H, W, C) -> (N, C, H, W)"""
    img = np.asarray(img).astype(np.float32) / 255. * 2 - 1
    img = img.transpose(2, 0, 1)[None]
    img = torch.tensor(img)
    return img

def from_tensor(x: torch.Tensor) -> Image.Image:
    """inverse of pil2vae"""
    x = x.cpu().detach().numpy()
    x = x[0].transpose(1, 2, 0)
    x = np.clip((x + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(x)


class PseudoGroupNorm(torch.nn.GroupNorm):
    def forward(self, x):
        orig_shape = x.size()
        x = x.contiguous()
        x = x.view(orig_shape[0], self.num_groups, -1)
        x = (x - x.mean(2, keepdim=True)) / torch.sqrt(torch.var(x, 2, unbiased=False, keepdim=True) + self.eps)
        x = x.view(orig_shape[0], self.num_channels, -1)
        x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        x = x.view(orig_shape)
        return x

    @classmethod
    def from_orig(cls, mod: torch.nn.GroupNorm):
        newmod = cls(mod.num_groups, mod.num_channels, mod.eps, mod.affine)
        newmod.weight.data[:] = mod.weight.data
        newmod.bias.data[:] = mod.bias.data
        return newmod

def swap_modules_inplace(module: torch.nn.Module, mapping):
    reassign = {}
    for name, mod in module.named_children():
        print(name)
        if type_before_parametrizations(mod) not in mapping:
            swap_modules_inplace(mod, mapping)
        if type_before_parametrizations(mod) in mapping:
            reassign[name] = mapping[type_before_parametrizations(mod)].from_orig(mod)
            print("reassigning", name, ":", type_before_parametrizations(mod), "->", mapping[type_before_parametrizations(mod)])
    for key, value in reassign.items():
        module._modules[key] = value
    return module

class LossModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            use_safetensors=True,
            torch_dtype=torch.float32,
            subfolder="vae")
        # output only mean
        quant = torch.nn.Conv2d(2 * 4, 4, 1)
        quant.weight.data[:] = vae.quant_conv.weight.data[:4, :, :, :]
        quant.bias.data[:] = vae.quant_conv.bias.data[:4]
        self.net = torch.nn.Sequential(vae.encoder, quant)
        self.fn = torch.nn.MSELoss(reduction="sum")
    def forward(self, img, zy):
        zx = self.net(img)
        return self.fn(zx, zy)


if __name__ == "__main__":
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        use_safetensors=True,
        torch_dtype=torch.float32,
        subfolder="vae")
    
    # output only mean
    quant = torch.nn.Conv2d(2 * 4, 4, 1)
    quant.weight.data[:] = vae.quant_conv.weight.data[:4, :, :, :]
    quant.bias.data[:] = vae.quant_conv.bias.data[:4]
    net = torch.nn.Sequential(vae.encoder, quant)
    image_path = Path("test/sample.png")
    # target_size = (512, 512)  # it takes too much memory for backward
    target_size = (128, 128)
    img = Image.open(image_path).convert("RGB").resize(target_size)
    with torch.no_grad():
        x = to_tensor(img)
        print(x.shape)
        # equivalent to vae.encode(x).latent_dist.mode()
        zx = net(x)
        print(zx.shape)

    if True:
        print("===== ONNX Generation =====")

        torch.onnx.export(
            net,
            (x,),
            "vae_encoder_unopt.onnx",
            input_names=["image"],
            output_names=["latents"],
            dynamic_axes={
                "image": {0: "batch", 2: "height", 3: "width"},
                "latents": {0: "batch", 2: "height_stride8", 3: "width_stride8"},
            },
        )

    if True:
        print("===== ONNX Optimization =====")
        # optimize onnx
        opt = onnxruntime.SessionOptions()
        # NOTE: Greater than ORT_ENABLE_EXTENDED may contain hardware specific optimizations.
        opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opt.optimized_model_filepath = "vae_encoder.onnx"
        opt.log_severity_level = 1  # log INFO
        _sess = onnxruntime.InferenceSession("vae_encoder_unopt.onnx", opt, providers=["CPUExecutionProvider"])

    if True:
        # See https://github.com/microsoft/onnxruntime/blob/a457c1df80f391ea6de0ab1d454297716976dbf3/orttraining/orttraining/python/training/experimental/gradient_graph/_gradient_graph_tools.py#L13
        print("===== ONNX-Training Generation =====")
        from onnxruntime.capi._pybind_state import GradientGraphBuilder
        lossnet = LossModule()
        print(lossnet)
        swap_modules_inplace(lossnet, {torch.nn.GroupNorm: PseudoGroupNorm})
        print(lossnet)
        print(lossnet(x, zx))

        f = io.BytesIO()
        torch.onnx.export(
            lossnet,
            (x, zx),
            f,
            input_names=["image", "target_latents"],
            output_names=["loss"],
            dynamic_axes={
                "image": {0: "batch", 2: "height", 3: "width"},
                "target_latents": {0: "batch", 2: "height_stride8", 3: "width_stride8"},
            },
            export_params=True,
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.TRAINING,
        )

        new_model = f.getvalue()
        builder = GradientGraphBuilder(new_model, {"loss"}, {"image"}, "loss")
        builder.build()
        builder.save("vae_encoder_grad.onnx")
        
    if True:
        print("===== ONNX-Training Test =====")
        opt = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession("vae_encoder_grad.onnx", opt, providers=["CPUExecutionProvider"])
        print([n.name for n in sess.get_outputs()])
        loss, dloss = sess.run(["loss", "image_grad"], {"image": x.numpy(), "target_latents": zx.numpy()})
        print("error", loss)
        print("image.grad", dloss)