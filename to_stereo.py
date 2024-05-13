import numpy as np
import ffmpeg
from dataclasses import dataclass
from numpy.typing import NDArray
from tqdm import tqdm
from datetime import datetime

from funcs import VideoInfo, get_video_info

def load_video_buffer(filepath, video_info: VideoInfo):
    video_buffer = np.zeros(
        (video_info.num_frames, video_info.height, video_info.width, 3), dtype=np.uint8
    )
    process = (
        ffmpeg.input(filepath, threads=16, thread_queue_size=8192)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run_async(pipe_stdout=True)
    )
    for i in range(video_info.num_frames):
        in_bytes = process.stdout.read(video_info.width * video_info.height * 3)
        if not in_bytes:
            break
        video_buffer[i] = np.frombuffer(in_bytes, np.uint8).reshape(
            [video_info.height, video_info.width, 3]
        )
    process.wait()
    return video_buffer


def write_output_video(output_path, video_buffer: NDArray, video_info: VideoInfo, filepath):
    in_modified = ffmpeg.input(
        "pipe:",
        format="rawvideo",
        pix_fmt="rgb24",
        s=f"{video_info.width}x{video_info.height}",
        framerate=video_info.framerate,
        thread_queue_size=8192,
    )
    in_original = ffmpeg.input(
        filepath,
        vn=None,
        sn=None,
        thread_queue_size=8192,
    )
    output = ffmpeg.output(
        in_modified,
        in_original,
        output_path,
        acodec="copy",
        crf=11,
        preset="faster",
        threads=16,
        framerate=video_info.framerate,
        s=f"{video_info.width}x{video_info.height}",
        # strict="-2", truehd audio causes issues
    )
    process = output.overwrite_output().run_async(pipe_stdin=True)
    for frame in video_buffer:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    process.wait()

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
    filepath = "./in_rgbd/upsidedown_video_depth_125922.mp4"
    max_shift = 30

    video_info = get_video_info(filepath)
    video_buffer = load_video_buffer(filepath, video_info)
    side_by_side_buffer = create_side_by_side_video(video_buffer, max_shift)


    now = datetime.now()
    hourminsec = now.strftime("%H%M%S")
    output_path = f"./out/musicvid-sbs-{hourminsec}.mp4"
    write_output_video(output_path, side_by_side_buffer, video_info, filepath)
