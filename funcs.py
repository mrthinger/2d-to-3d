import argparse
import time
from typing import Literal
import numpy as np
import os
import torch
import torch.nn.functional as F
import ffmpeg
from depth_anything.dpt import DepthAnything
from dataclasses import dataclass

model_configs = {
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
}

EncoderType = Literal["vits", "vitb", "vitl"]


def load_model(
    encoder: EncoderType, device: torch.device, dtype: torch.dtype
) -> DepthAnything:
    depth_anything = DepthAnything(model_configs[encoder])
    depth_anything.load_state_dict(
        torch.load(f"./checkpoints/depth_anything_{encoder}14.pth")
    )
    depth_anything = depth_anything.to(device, dtype=dtype).eval()

    total_params = sum(param.numel() for param in depth_anything.parameters())
    print("Total parameters: {:.2f}M".format(total_params / 1e6))

    return depth_anything


def load_input_stats(device: torch.device, dtype: torch.dtype):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=dtype, device=device).view(
        1, 3, 1, 1
    )
    std = torch.tensor([0.229, 0.224, 0.225], dtype=dtype, device=device).view(
        1, 3, 1, 1
    )
    return mean, std


def get_filepaths(video_path):
    if os.path.isfile(video_path):
        if video_path.endswith("txt"):
            with open(video_path, "r") as f:
                lines = f.read().splitlines()
        else:
            filepaths = [video_path]
    else:
        filepaths = os.listdir(video_path)
        filepaths = [
            os.path.join(video_path, filename)
            for filename in filepaths
            if not filename.startswith(".")
        ]
        filepaths.sort()

    return filepaths


def resize_and_pad(frames_batch: torch.Tensor):
    batch_size, channels, height, width = frames_batch.shape

    aspect_ratio = width / height

    # Scale height to 518 (518/14 = 37)
    # TODO: auto determine this or set to 77
    target_height = 20 * 14  # 37 -> 77 patches (increased res / less downscale)

    target_width = int(target_height * aspect_ratio)

    # Round the width to the closest multiple of 14
    target_width = round(target_width / 14) * 14

    resized_frames = F.interpolate(
        frames_batch,
        size=(target_height, target_width),
        mode="bilinear", # TODO: bicubic on cuda
        align_corners=False,
    )
    return resized_frames


def preprocess_batch(frames_batch: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    frames_batch = frames_batch.permute(0, 3, 1, 2) / 255.0
    frames_batch = resize_and_pad(frames_batch)
    frames_batch = (frames_batch - mean) / std

    return frames_batch


@dataclass
class VideoInfo:
    width: int
    height: int
    num_frames: int
    framerate: str


def get_video_info(filepath: str) -> VideoInfo:
    probe = ffmpeg.probe(filepath)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    return VideoInfo(
        width=int(video_info["width"]),
        height=int(video_info["height"]),
        num_frames=int(video_info["nb_frames"]),
        framerate=video_info["avg_frame_rate"],
    )
