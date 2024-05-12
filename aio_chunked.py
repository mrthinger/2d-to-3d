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
    video_path: str = "./sw_middle_10s.mkv"
    encoder: EncoderType = "vitl"  #  vits, vitb, vitl
    outdir: str = "./out"


# BUCKET_NAME = os.environ["BUCKET_NAME"]
# AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
# AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
# BUCKET_HOST = os.environ["BUCKET_HOST"]


if __name__ == "__main__":
    args = Args()
    # s3 = s3fs.S3FileSystem(
    #     endpoint_url=BUCKET_HOST, key=AWS_ACCESS_KEY_ID, secret=AWS_SECRET_ACCESS_KEY
    # )

    BATCH_SIZE = 2
    DEVICE = "cuda"
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

    CHUNK_SIZE = 2 * 1024 * 1024 * 1024  # GB
    CHUNK_FRAMES = CHUNK_SIZE // (vinfo.width * vinfo.height * 3)

    vbuffer = np.zeros((CHUNK_FRAMES, vinfo.height, output_width, 3), dtype=np.uint8)

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
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="quiet")
        .run_async(pipe_stdout=True)
    )

    filename = os.path.basename(filepath)
    timestamp = time.strftime("%H%M%S")
    output_name = filename[: filename.rfind(".")] + f"_video_depth_{timestamp}.mkv"
    local_output_path = os.path.join(
        args.outdir,
        output_name,
    )
    # bucket_output_path = f"{BUCKET_NAME}/{output_name}"

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
        sn=None,
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
        loglevel="quiet",
    )

    process_output = output.overwrite_output().run_async(pipe_stdin=True)

    progress_bar = tqdm(total=vinfo.num_frames, unit="frames")

    for chunk_start in range(0, vinfo.num_frames, CHUNK_FRAMES):
        chunk_end = min(chunk_start + CHUNK_FRAMES, vinfo.num_frames)
        chunk_frames = chunk_end - chunk_start

        progress_bar.set_description('reading')
        for i in range(chunk_frames):
            in_bytes = process.stdout.read(vinfo.width * vinfo.height * 3)
            if not in_bytes:
                break

            vbuffer[i, :, : vinfo.width, :] = np.frombuffer(in_bytes, np.uint8).reshape(
                [vinfo.height, vinfo.width, 3]
            )

        for i in range(0, chunk_frames, BATCH_SIZE):
            progress_bar.set_description('depthing')
            batch_start = i
            batch_end = min(i + BATCH_SIZE, chunk_frames)
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

            # Write the processed batch to the output stream
            progress_bar.set_description('writing')
            for frame in vbuffer[batch_start:batch_end]:
                process_output.stdin.write(frame.tobytes())
                progress_bar.update(1)

        # Write the processed batch to the output stream
        # for frame in vbuffer[:chunk_frames]:
        #     process_output.stdin.write(frame.tobytes())
        #     progress_bar.update(1)

    process.wait()
    process_output.stdin.close()
    process_output.wait()

    # s3.put(local_output_path, bucket_output_path, callback=TqdmCallback())
