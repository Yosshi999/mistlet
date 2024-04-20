from dataclasses import dataclass
from typing import Optional
import tyro


@dataclass
class Args:
    """Configs for Mist V1.2"""
    input_image_path: str = "test/sample.png"
    """path of input image"""
    input_dir_path: Optional[str] = None
    """Path of the dir of images to be processed"""
    output_dir: str = "test/misted"
    """path of output dir"""
    recon_dir: str = "test/recon"
    """path of reconstructed dir"""
    reconstruction: bool = True
    """Save reconstructed misted sample"""
    epsilon: int = 16
    """The strength of Mist"""
    steps: int = 100
    """The step of Mist"""
    input_size: int = 512
    """The shorter length of input for Mist"""
    block_num: int = 1
    """The number of partitioned blocks"""
    mask: bool = False
    """Whether to mask certain region of Mist or not. Work only when input_dir_path is None. """
    mask_path: str = "test/processed_mask.png"
    """Path of the mask"""
    keep_ratio: bool = False
    """Whether to keep the original ratio of the image or not"""
    cuda: bool = False
    """Use CUDA"""
    random_seed: int = 0
    """Random Seed"""


def parse_args() -> Args:
    return tyro.cli(Args)
