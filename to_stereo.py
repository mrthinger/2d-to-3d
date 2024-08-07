import os
import sys
import time
import numpy as np
import ffmpeg
from dataclasses import dataclass
from numpy.typing import NDArray
from tqdm import tqdm
from datetime import datetime

from funcs import VideoInfo, get_video_info


def create_side_by_side_video(video_buffer: NDArray, max_shift: int):
    num_frames, height, width, _ = video_buffer.shape
    single_video_width = width // 2

    depth_map = video_buffer[:, :, single_video_width:, 0].astype(np.float32)
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    depth_range = max_depth - min_depth

    # Precalculate the disparity map using the depth map buffer
    depth_map -= min_depth
    depth_map /= depth_range
    depth_map *= max_shift
    depth_map = depth_map.astype(np.int32)

    for i in range(num_frames):
        for y in range(height):
            shifts = depth_map[i, y, :]
            shifted_xs = np.clip(np.arange(single_video_width) + shifts, 0, single_video_width - 1)
            video_buffer[i, y, single_video_width:, :] = video_buffer[i, y, shifted_xs, :]

    return video_buffer


if __name__ == "__main__":
    os.makedirs("./build/sbs", exist_ok=True)

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "sw-depth-000.mp4"

    filepath = f"./build/split/{filename}"

    # filename = "sw-depth-000.mp4"
    # filepath = f"./build/split/{filename}"

    filename = "og_video_depth_021413.mp4"
    filepath = f"./build/trailer/{filename}"

    max_shift = 20

    output_path = f"./build/trailer/deadpool-sbs-{max_shift}-{filename}"
    video_info = get_video_info(filepath)
    output_width = video_info.width

    CHUNK_SIZE = 1 * 1024 * 1024 * 1024  # GB
    CHUNK_FRAMES = CHUNK_SIZE // (output_width * video_info.height * 3)

    vbuffer = np.zeros(
        (CHUNK_FRAMES, video_info.height, output_width, 3), dtype=np.uint8
    )

    process_input = (
        ffmpeg.input(
            filepath,
            threads=0,
            thread_queue_size=8192,
        )
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="quiet")
        .run_async(pipe_stdout=True)
    )

    in_modified = ffmpeg.input(
        "pipe:",
        format="rawvideo",
        pix_fmt="rgb24",
        s=f"{output_width}x{video_info.height}",
        framerate=video_info.framerate,
        thread_queue_size=8192,
        hwaccel="videotoolbox",
    )
    in_original = ffmpeg.input(
        filepath,
        thread_queue_size=8192,
        vn=None,
        sn=None,
        hwaccel="videotoolbox",
    )
    process_output = (
        ffmpeg.output(
            in_modified,
            in_original,
            output_path,
            acodec="copy",
            vcodec="hevc_videotoolbox",
            threads=0,
            framerate=video_info.framerate,
            s=f"{output_width}x{video_info.height}",
            loglevel="quiet",
            # preset="faster",
            # pix_fmt="yuv420p",
            # video_bitrate='8000k',
            # **{'allow_sw': '1'}
            # **{"q:v": "88"},
        )
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    progress_bar = tqdm(total=video_info.num_frames, unit="frames")

    for chunk_start in range(0, video_info.num_frames, CHUNK_FRAMES):
        chunk_end = min(chunk_start + CHUNK_FRAMES, video_info.num_frames)
        chunk_frames = chunk_end - chunk_start

        progress_bar.set_description("read")
        for i in range(chunk_frames):
            in_bytes = process_input.stdout.read(
                video_info.width * video_info.height * 3
            )
            if not in_bytes:
                break

            vbuffer[i, :, : video_info.width, :] = np.frombuffer(
                in_bytes, np.uint8
            ).reshape([video_info.height, video_info.width, 3])

        progress_bar.set_description("process")
        start = time.time()
        vbuffer[:chunk_frames] = create_side_by_side_video(
            vbuffer[:chunk_frames], max_shift
        )
        end = time.time()
        elapsed = end - start
        fps = chunk_frames / elapsed
        print(f"Kernel execution time: {elapsed:.4f} seconds")
        print(f"Frames per second: {fps:.2f}")

        progress_bar.set_description("write")
        for frame in vbuffer[:chunk_frames]:
            process_output.stdin.write(frame.tobytes())
            progress_bar.update(1)

    process_output.stdin.close()
    process_output.wait()
    progress_bar.close()

    print("debug: waiting input (safe to close)")
    process_input.stdin.close()
    process_input.wait()
