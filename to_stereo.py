import os
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
    side_by_side_buffer = np.zeros_like(video_buffer)
    original_video = video_buffer[:, :, :single_video_width, :]
    depth_map = video_buffer[:, :, single_video_width:, 0]
    
    depth_map = depth_map.astype(np.float32)

    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    depth_range = max_depth - min_depth
    
    # Precalculate the disparity map using the depth map buffer
    depth_map = (depth_map - min_depth) / depth_range
    depth_map = (depth_map) * max_shift
    depth_map = depth_map.astype(np.int32)
    
    left_eye_view = original_video
    right_eye_view = np.zeros((num_frames, height, single_video_width, 3), dtype=np.uint8)
    
    for i in tqdm(range(num_frames)):
        for y in range(height):
            shifts = depth_map[i, y, :]
            shifted_xs = np.clip(np.arange(single_video_width) + shifts, 0, single_video_width - 1)
            right_eye_view[i, y, :, :] = original_video[i, y, shifted_xs, :]
    
    side_by_side_buffer[:, :, :single_video_width, :] = left_eye_view
    side_by_side_buffer[:, :, single_video_width:, :] = right_eye_view
    
    return side_by_side_buffer


if __name__ == "__main__":
    os.makedirs('./build/sbs', exist_ok=True)

    filename = "sw_qt_video_depth_211953.mp4"
    filepath = f"./build/depth/{filename}"
    max_shift = 20

    output_path = f"./build/sbs/{max_shift}-{filename}"
    video_info = get_video_info(filepath)
    output_width = video_info.width

    CHUNK_SIZE = 32 * 1024 * 1024 * 1024  # GB
    CHUNK_FRAMES = CHUNK_SIZE // (video_info.width * video_info.height * 3)

    vbuffer = np.zeros((CHUNK_FRAMES, video_info.height, output_width, 3), dtype=np.uint8)

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
    )
    in_original = ffmpeg.input(
        filepath,
        thread_queue_size=8192,
        vn=None,
        sn=None,
    )
    process_output = ffmpeg.output(
        in_modified,
        in_original,
        output_path,
        acodec="copy",
        crf=11,
        vcodec="libx264",
        pix_fmt="yuv420p",
        preset="faster",
        threads=0,
        framerate=video_info.framerate,
        s=f"{output_width}x{video_info.height}",
        loglevel="quiet",
    ).overwrite_output().run_async(pipe_stdin=True)

    progress_bar = tqdm(total=video_info.num_frames, unit="frames")

    for chunk_start in range(0, video_info.num_frames, CHUNK_FRAMES):
        chunk_end = min(chunk_start + CHUNK_FRAMES, video_info.num_frames)
        chunk_frames = chunk_end - chunk_start

        progress_bar.set_description('read')
        for i in range(chunk_frames):
            in_bytes = process_input.stdout.read(video_info.width * video_info.height * 3)
            if not in_bytes:
                break

            vbuffer[i, :, :video_info.width, :] = np.frombuffer(in_bytes, np.uint8).reshape(
                [video_info.height, video_info.width, 3]
            )

        progress_bar.set_description('process')
        vbuffer[:chunk_frames] = create_side_by_side_video(vbuffer[:chunk_frames], max_shift)

        progress_bar.set_description('write')
        for frame in vbuffer[:chunk_frames]:
            process_output.stdin.write(frame.tobytes())
            progress_bar.update(1)

    process_input.wait()
    process_output.stdin.close()
    process_output.wait()
    progress_bar.close()