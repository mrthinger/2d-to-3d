from dotenv import load_dotenv

load_dotenv()

import time
import numpy as np
import os
import torch
import torch.nn.functional as F
import ffmpeg
from dataclasses import dataclass
from fsspec.callbacks import TqdmCallback
from tqdm import tqdm
import s3fs
from funcs import (
    EncoderType,
    get_filepaths,
    get_video_info,
    load_input_stats,
    load_model,
    preprocess_batch,
)


@dataclass
class Args:
    video_path: str = "./test10.avi"
    encoder: EncoderType = "vits"  #  vits, vitb, vitl
    outdir: str = "./out"


BUCKET_NAME = os.environ["BUCKET_NAME"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
BUCKET_HOST = os.environ["BUCKET_HOST"]


if __name__ == "__main__":
    args = Args()
    s3 = s3fs.S3FileSystem(
        endpoint_url=BUCKET_HOST, key=AWS_ACCESS_KEY_ID, secret=AWS_SECRET_ACCESS_KEY
    )

    BATCH_SIZE = 2
    DEVICE = "mps"
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16
    print(DEVICE, DTYPE)

    depth_anything = load_model(args.encoder, DEVICE, DTYPE)
    mean, std = load_input_stats(DEVICE, DTYPE)
    filepaths = get_filepaths(args.video_path)

    os.makedirs(args.outdir, exist_ok=True)

    filepath = filepaths[0]
    vinfo = get_video_info(filepath)

    output_width = vinfo.width * 2  # side by side

    vbuffer = np.zeros(
        (vinfo.num_frames, vinfo.height, output_width, 3), dtype=np.uint8
    )

    print(
        "Memory size of numpy video_buffer in bytes:",
        vbuffer.size * vbuffer.itemsize,
    )

    process = (
        ffmpeg.input(
            filepath,
            threads=16,
            thread_queue_size=8192,
        )
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run_async(pipe_stdout=True)
    )

    for i in range(vinfo.num_frames):
        in_bytes = process.stdout.read(vinfo.width * vinfo.height * 3)
        if not in_bytes:
            break

        vbuffer[i, :, : vinfo.width, :] = np.frombuffer(in_bytes, np.uint8).reshape(
            [vinfo.height, vinfo.width, 3]
        )

    process.wait()

    for i in tqdm(range(0, vinfo.num_frames, BATCH_SIZE), unit=f"{BATCH_SIZE} frames"):
        batch_start = i
        batch_end = min(i + BATCH_SIZE, vinfo.num_frames)
        batch_size = batch_end - batch_start

        frames_batch = torch.from_numpy(
            vbuffer[batch_start:batch_end, :, : vinfo.width, :]
        ).to(DEVICE, dtype=DTYPE)

        frames_batch = preprocess_batch(frames_batch, mean, std)

        with torch.no_grad():
            depth_batch = depth_anything(frames_batch)

        depth_batch = F.interpolate(
            depth_batch[None],
            (vinfo.height, vinfo.width),
            mode="bilinear",
            align_corners=False,
        )[0]

        depth_batch = (
            (depth_batch - depth_batch.min())
            / (depth_batch.max() - depth_batch.min())
            * 255.0
        )
        depth_color_batch = torch.repeat_interleave(
            depth_batch.unsqueeze(-1), 3, dim=-1
        )
        depth_color_batch = depth_color_batch.to(dtype=torch.uint8).cpu().numpy()

        vbuffer[batch_start:batch_end, :, vinfo.width :, :] = depth_color_batch

    filename = os.path.basename(filepath)
    timestamp = time.strftime("%H%M%S")
    output_name = filename[: filename.rfind(".")] + f"_video_depth_{timestamp}.mp4"
    local_output_path = os.path.join(
        args.outdir,
        output_name,
    )
    bucket_output_path = f"{BUCKET_NAME}/{output_name}"

    in_modified = ffmpeg.input(
        "pipe:",
        format="rawvideo",
        pix_fmt="rgb24",
        s=f"{output_width}x{vinfo.height}",
        framerate=vinfo.framerate,
        thread_queue_size=8192,
    )
    in_original = ffmpeg.input(
        filepath,
        thread_queue_size=8192,
        vn=None,
    )
    output = ffmpeg.output(
        in_modified,
        in_original,
        local_output_path,
        acodec="copy",
        crf=11,
        preset="faster",
        threads=16,
        framerate=vinfo.framerate,
        s=f"{output_width}x{vinfo.height}",
    )

    process = output.overwrite_output().run_async(pipe_stdin=True)

    # Write the frames to the input stream
    for frame in vbuffer:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    process.wait()

    # s3.put(local_output_path, bucket_output_path, callback=TqdmCallback())
