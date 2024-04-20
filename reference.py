from abc import abstractmethod, ABCMeta
from itertools import product
from pathlib import Path

from diffusers import AutoencoderKL
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

from utils import parse_args, Args

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
    img = torch.tensor(img)
    return img

def from_tensor(x: torch.Tensor) -> Image.Image:
    """inverse of pil2vae"""
    x = x.cpu().detach().numpy()
    x = x[0].transpose(1, 2, 0)
    x = np.clip((x + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(x)

def reconstruction(net, args: Args, img: Image.Image) -> Image.Image:
    device = "cuda" if args.cuda else "cpu"
    x = to_tensor(img).to(device)
    with torch.no_grad():
        zx = net.encode(x).latent_dist.mode()
        y = net.decode(zx).sample
        return from_tensor(y)

def infer(net, args: Args, img: Image.Image, target_img: Image.Image, mask: Image.Image) -> np.ndarray:
    device = "cuda" if args.cuda else "cpu"

    img = np.array(img)[:, :, :3]
    target_img = np.array(target_img)[:, :, :3]

    rng = np.random.default_rng(seed=args.random_seed)
    g_cpu = torch.Generator()
    g_cpu.manual_seed(args.random_seed)
    linf = LinfBoundary(args.epsilon)
    delta = linf.sample(img.shape, rng)
    delta = np.clip(img.astype(np.int64) + delta, 0, 255) - img.astype(np.int64)

    # image is normalized to [-1, 1]
    x = to_tensor(img).to(device)
    y = to_tensor(target_img).to(device)
    # mask is normalized to [0, 1]
    mask = np.asarray(mask).astype(np.float32)[:, :, :3] / 255.0
    mask = torch.tensor(mask.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # perturb stepsize
    alpha = 1
    fn = torch.nn.MSELoss(reduction="sum")
    for i in tqdm(range(args.steps), desc="perturb"):
        d = to_tensor(delta).to(device).requires_grad_()
        zx = net.encode(x + d * mask).latent_dist.sample(generator=g_cpu)
        zy = net.encode(y).latent_dist.sample(generator=g_cpu).detach()
        loss = fn(zx, zy)
        loss.backward()
        sign = d.grad.data.sign().cpu().numpy()
        sign = sign[0].transpose(1,2,0)

        # update delta using PGD
        delta = delta + sign * alpha
        delta = linf.project(delta)
        delta = np.clip(img.astype(np.int64) + delta, 0, 255) - img.astype(np.int64)

    return np.clip(img.astype(np.int64) + delta, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    args = parse_args()
    out_folder = Path(args.output_dir)
    out_folder.mkdir(exist_ok=True)
    recon_folder = Path(args.recon_dir)
    recon_folder.mkdir(exist_ok=True)
    
    device = "cuda" if args.cuda else "cpu"
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        use_safetensors=True,
        torch_dtype=torch.float32,
        subfolder="vae").to(device)

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
        for i, j in tqdm(product(range(args.block_num), repeat=2)):
            xyxy = (block_w * i, block_h * j, block_w * (i+1), block_h * (j+1))
            img_block = img.crop(xyxy)
            target_block = target_img.crop(xyxy)
            mask_block = mask.crop(xyxy)
            output_image[block_h * j : block_h * (j+1), block_w * i : block_w * (i+1)] = infer(vae, args, img_block, target_block, mask)
        
        output = Image.fromarray(output_image)
        output.save(out_folder / image_path.name)

        if args.reconstruction:
            img = Image.open(out_folder / image_path.name).resize(target_size)
            recon = reconstruction(vae, args, img)
            recon.save(recon_folder / image_path.name)
