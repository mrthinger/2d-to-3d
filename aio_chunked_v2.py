import time
import numpy as np
import os
import torch
import torch.nn.functional as F
import ffmpeg
from dataclasses import dataclass
from tqdm.auto import tqdm
from funcs import (
    EncoderType,
    get_filepaths,
    get_video_info,
    load_input_stats,
    load_model_v2,
    preprocess_batch,
)


@dataclass
class Args:
    video_path: str = "./build/src/blade_runner_first_10sec.mkv"
    encoder: EncoderType = "vitl"  #  vits, vitb, vitl
    outdir: str = "./build/depth"


if __name__ == "__main__":
    args = Args()

    BATCH_SIZE = 4
    DEVICE = "cuda:0"
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16
    print(DEVICE, DTYPE)

    depth_anything = load_model_v2(args.encoder, DEVICE, DTYPE)
    # depth_anything = torch.compile(depth_anything)
    mean, std = load_input_stats(DEVICE, DTYPE)
    filepaths = get_filepaths(args.video_path)

    os.makedirs(args.outdir, exist_ok=True)

    filepath = filepaths[0]
    vinfo = get_video_info(filepath)

    output_width = vinfo.width * 2  # side by side

    CHUNK_SIZE =int(4 * 1024 * 1024 * 1024)  # GB
    CHUNK_FRAMES = CHUNK_SIZE // (output_width * vinfo.height * 3)

    vbuffer = np.zeros((CHUNK_FRAMES, vinfo.height, output_width, 3), dtype=np.uint8)

    print(
        "Memory size of numpy video_buffer in bytes:",
        vbuffer.size * vbuffer.itemsize,
    )

    process_input = (
        ffmpeg.input(
            filepath, threads=0, thread_queue_size=8192
        )
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="quiet",)
        .global_args("-hwaccel", "cuda")
        # .global_args("-hwaccel_output_format", "cuda")
        .run_async(pipe_stdout=True)
    )

    filename = os.path.basename(filepath)
    timestamp = time.strftime("%H%M%S")
    output_name = filename[: filename.rfind(".")] + f"_video_depth_{timestamp}"
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
        local_output_path + ".mkv",
        acodec="copy",
        vcodec="av1_nvenc",
        preset="p1", 
        threads=0,
        framerate=vinfo.framerate,
        s=f"{output_width}x{vinfo.height}",
        pix_fmt="yuv420p",
        loglevel="quiet",
    )

    process_output = output.overwrite_output().run_async(pipe_stdin=True)

    total_progress = tqdm(total=vinfo.num_frames, unit="frames", desc="Total Progress")
    chunk_progress = tqdm(total=CHUNK_FRAMES, unit="frames", leave=False, position=1)

    for chunk_start in range(0, vinfo.num_frames, CHUNK_FRAMES):
        chunk_end = min(chunk_start + CHUNK_FRAMES, vinfo.num_frames)
        chunk_frames = chunk_end - chunk_start

        chunk_progress.reset(total=chunk_frames)
        chunk_progress.set_description("Read")

        # Read phase
        for i in range(chunk_frames):
            in_bytes = process_input.stdout.read(vinfo.width * vinfo.height * 3)
            if not in_bytes:
                break

            vbuffer[i, :, : vinfo.width, :] = np.frombuffer(in_bytes, np.uint8).reshape(
                [vinfo.height, vinfo.width, 3]
            )
            chunk_progress.update(1)

        # Depth phase
        chunk_progress.reset(total=chunk_frames)
        chunk_progress.set_description("Depth")
        for i in range(0, chunk_frames, BATCH_SIZE):
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
                mode="bicubic",
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

            chunk_progress.update(batch_size)

        # Write phase
        chunk_progress.reset(total=chunk_frames)
        chunk_progress.set_description("Write")
        for frame in vbuffer[:chunk_frames]:
            process_output.stdin.write(frame.tobytes())
            chunk_progress.update(1)
            total_progress.update(1)


    chunk_progress.close()
    total_progress.close()

    process_input.stdout.close()
    print("Input stream closed")
    process_input.wait()
    print("Input process completed")
    process_output.stdin.close()
    print("Output stream closed")
    process_output.wait()
    print("Output process completed")

