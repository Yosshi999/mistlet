from abc import abstractmethod, ABCMeta
from itertools import product
from pathlib import Path

from diffusers import AutoencoderKL
from PIL import Image
import numpy as np
import onnxruntime
import torch
import tyro
from tqdm import tqdm

from utils import Args

VAE_FACTOR = 8
target_image_path = "MIST.png"


class Boundary(metaclass=ABCMeta):
    def __init__(self, bound):
        self.bound = bound
    @abstractmethod
    def sample(self, size, rng) -> np.ndarray:
        pass
    @abstractmethod
    def project(self, x: np.ndarray) -> np.ndarray:
        pass

class LinfBoundary(Boundary):
    def sample(self, size, rng) -> np.ndarray:
        return rng.integers(-self.bound, self.bound, size=size, endpoint=True)
    def project(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, -self.bound, self.bound)

def to_tensor(img: Image.Image | np.ndarray) -> torch.Tensor:
    """resize [0, 255] -> [-1, 1] and permute (H, W, C) -> (N, C, H, W)"""
    img = np.asarray(img).astype(np.float32) / 255. * 2 - 1
    img = img.transpose(2, 0, 1)[None]
    img = np.ascontiguousarray(img)
    return img

def from_tensor(x: torch.Tensor) -> Image.Image:
    """inverse of pil2vae"""
    x = x[0].transpose(1, 2, 0)
    x = np.clip((x + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(x)

def reconstruction(enc, dec, args: Args, img: Image.Image) -> Image.Image:
    x = to_tensor(img)
    zx, = enc.run(["latents"], {"image": x})
    y, = dec.run(["image"], {"latents": zx})
    return from_tensor(y)

def infer(sess, sess_grad, args: Args, img: Image.Image, target_img: Image.Image, mask: Image.Image) -> np.ndarray:
    img = np.array(img)[:, :, :3]
    target_img = np.array(target_img)[:, :, :3]

    rng = np.random.default_rng(seed=args.random_seed)
    linf = LinfBoundary(args.epsilon)
    delta = linf.sample(img.shape, rng)
    delta = np.clip(img.astype(np.int64) + delta, 0, 255) - img.astype(np.int64)

    # image is normalized to [-1, 1]
    x = to_tensor(img)
    y = to_tensor(target_img)
    zy, = sess.run(["latents"], {"image": y})

    # mask is normalized to [0, 1]
    mask = np.asarray(mask).astype(np.float32)[:, :, :3] / 255.0
    mask = mask.transpose(2, 0, 1)[None]
    mask = np.ascontiguousarray(mask)

    # perturb stepsize
    alpha = 1
    for i in tqdm(range(args.steps), desc="perturb", leave=False):
        d = to_tensor(delta)
        _loss, dloss = sess_grad.run(["loss", "image_grad"], {"image": x + d * mask, "target_latents": zy})
        sign = np.sign(dloss)
        sign = sign[0].transpose(1,2,0)

        # update delta using PGD
        delta = delta + sign * alpha
        delta = linf.project(delta)
        delta = np.clip(img.astype(np.int64) + delta, 0, 255) - img.astype(np.int64)

    return np.clip(img.astype(np.int64) + delta, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    args = tyro.cli(Args)
    out_folder = Path(args.output_dir)
    out_folder.mkdir(exist_ok=True)
    recon_folder = Path(args.recon_dir)
    recon_folder.mkdir(exist_ok=True)
    
    assert not args.cuda, "cuda is not supported"

    opt = onnxruntime.SessionOptions()
    sess = onnxruntime.InferenceSession("app/vae_encoder.onnx", opt, provider_options=["CPUExecutionProvider"])
    sess_grad = onnxruntime.InferenceSession("app/vae_encoder_grad.onnx", opt, providers=["CPUExecutionProvider"])

    if args.mask:
        assert args.input_dir_path is None, "masking is not implemented for batch processing"

    if args.input_dir_path:
        image_paths = sorted(Path(args.input_dir_path).iterdir())
    else:
        image_paths = [Path(args.input_image_path)]

    for image_path in image_paths:
        if args.keep_ratio:
            img = Image.open(image_path)
            shortest = min(img.width, img.height)
            scale = shortest / args.input_size
            factor = args.block_num * VAE_FACTOR
            target_size = (
                int(img.size[0] / scale) // factor * factor,
                int(img.size[1] / scale) // factor * factor
            )
        else:
            target_size = (args.input_size, args.input_size)
        img = Image.open(image_path).resize(target_size)
        if args.mask:
            mask = Image.open(args.mask_path).resize(target_size)
        else:
            mask = Image.new("RGB", target_size, (255, 255, 255))
        target_img = Image.open(target_image_path).resize(target_size)
        block_w = target_size[0] // args.block_num
        block_h = target_size[1] // args.block_num
        output_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        for i, j in tqdm(product(range(args.block_num), repeat=2), total=args.block_num**2):
            xyxy = (block_w * i, block_h * j, block_w * (i+1), block_h * (j+1))
            img_block = img.crop(xyxy)
            target_block = target_img.crop(xyxy)
            mask_block = mask.crop(xyxy)
            output_image[block_h * j : block_h * (j+1), block_w * i : block_w * (i+1)] = infer(sess, sess_grad, args, img_block, target_block, mask_block)
        
        output = Image.fromarray(output_image)
        output.save(out_folder / ("onnx_" + image_path.name))

        if args.reconstruction:
            sess_dec = onnxruntime.InferenceSession("app/vae_decoder.onnx", opt, provider_options=["CPUExecutionProvider"])
            img = Image.open(out_folder / image_path.name).resize(target_size)
            recon = reconstruction(sess, sess_dec, args, img)
            recon.save(recon_folder / ("onnx_" + image_path.name))
