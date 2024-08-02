import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
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
    world_size: int = 8  # Total number of GPUs
    io_gpu: int = 0  # GPU to use for I/O operations

BATCH_SIZE = 4
DTYPE = torch.float16

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_depth_estimation(rank, args):
    try:
        setup(rank, args.world_size)
        
        if rank == args.io_gpu:
            run_main_process(args)
        else:
            run_worker_process(rank, args)
    # except Exception as e:
    #     print(f"Error in process {rank}: {str(e)}")
    finally:
        cleanup()

def run_main_process(args):
    DEVICE = f"cuda:{args.io_gpu}"
    
    filepaths = get_filepaths(args.video_path)
    os.makedirs(args.outdir, exist_ok=True)
    
    filepath = filepaths[0]
    vinfo = get_video_info(filepath)
    
    print(f"[DEBUG] Video info: {vinfo}")  # Debug print
    
    output_width = vinfo.width * 2  # side by side
    
    # Set up input and output processes
    process_input = setup_input_process(filepath)
    process_output = setup_output_process(args, filepath, vinfo, output_width)
    
    total_batch_size = BATCH_SIZE * (args.world_size - 1)
    input_buffer = torch.empty((total_batch_size, vinfo.height, vinfo.width, 3), dtype=torch.uint8, device=DEVICE)
    output_buffer = torch.empty((total_batch_size, vinfo.height, vinfo.width, 3), dtype=torch.uint8, device=DEVICE)
    next_input_buffer = torch.empty((total_batch_size, vinfo.height, vinfo.width, 3), dtype=torch.uint8, device=DEVICE)
    
    print(f"[DEBUG] Total batch size: {total_batch_size}")  # Debug print
    
    total_progress = tqdm(total=vinfo.num_frames, unit="frames", desc="Total Progress")
    
    # Broadcast video info to all processes
    vinfo_tensor = torch.tensor([vinfo.height, vinfo.width], dtype=torch.float32, device=DEVICE)
    dist.broadcast(vinfo_tensor, src=args.io_gpu)
    
    total_frames_processed = 0
    
    # Read first batch
    for i in range(total_batch_size):
        in_bytes = process_input.stdout.read(vinfo.width * vinfo.height * 3)
        if not in_bytes:
            break
        frame = torch.frombuffer(in_bytes, dtype=torch.uint8).reshape(vinfo.height, vinfo.width, 3).to(DEVICE)
        input_buffer[i] = frame
    
    while total_frames_processed < vinfo.num_frames:
        frames_to_process = min(total_batch_size, vinfo.num_frames - total_frames_processed)
        
        # Pad the input_buffer if necessary
        if frames_to_process < total_batch_size:
            padding = total_batch_size - frames_to_process
            input_buffer[frames_to_process:] = input_buffer[frames_to_process-1].repeat(padding, 1, 1, 1)
        
        # Distribute frames to worker processes
        for i in range(args.world_size):
            if i != args.io_gpu:
                start = (i - 1) * BATCH_SIZE if i > args.io_gpu else i * BATCH_SIZE
                end = start + BATCH_SIZE
                signal = torch.tensor([1], device=DEVICE)
                dist.send(signal, dst=i)
                frames_to_send = input_buffer[start:end].contiguous()
                dist.send(frames_to_send, dst=i)
        
        # Read next batch while processing current batch
        next_frames_to_process = min(total_batch_size, vinfo.num_frames - (total_frames_processed + frames_to_process))
        for i in range(next_frames_to_process):
            in_bytes = process_input.stdout.read(vinfo.width * vinfo.height * 3)
            if not in_bytes:
                break
            frame = torch.frombuffer(in_bytes, dtype=torch.uint8).reshape(vinfo.height, vinfo.width, 3).to(DEVICE)
            next_input_buffer[i] = frame
        
        # Gather results from worker processes
        for i in range(args.world_size):
            if i != args.io_gpu:
                start = (i - 1) * BATCH_SIZE if i > args.io_gpu else i * BATCH_SIZE
                end = start + BATCH_SIZE
                dist.recv(output_buffer[start:end], src=i)
        
        # Concatenate input and output frames
        combined_frames = torch.cat([input_buffer[:frames_to_process], output_buffer[:frames_to_process]], dim=2)
        combined_frames = combined_frames.cpu().numpy()

        # Write to output
        for i in range(frames_to_process):
            process_output.stdin.write(combined_frames[i].tobytes())
            total_progress.update(1)
        
        # Swap buffers and update counters
        input_buffer, next_input_buffer = next_input_buffer, input_buffer
        total_frames_processed += frames_to_process
    
    # Send termination signal to worker processes
    for i in range(args.world_size):
        if i != args.io_gpu:
            dist.send(torch.tensor([-1], device=DEVICE), dst=i)
    
    # Cleanup
    total_progress.close()
    process_input.stdout.close()
    process_input.wait()
    process_output.stdin.close()
    process_output.wait()
    
    print(f"Main process processed {total_frames_processed} frames")

def run_worker_process(rank, args):
    DEVICE = f"cuda:{rank}"
    
    depth_anything = load_model_v2(args.encoder, DEVICE, DTYPE)
    mean, std = load_input_stats(DEVICE, DTYPE)
    
    # Receive video info from main process
    vinfo_tensor = torch.empty(2, dtype=torch.float32, device=DEVICE)
    dist.broadcast(vinfo_tensor, src=args.io_gpu)
    vinfo_height, vinfo_width = vinfo_tensor.tolist()
    vinfo_height, vinfo_width = int(vinfo_height), int(vinfo_width)
    print(vinfo_height, vinfo_width)
    
    termination_signal = torch.empty(1, dtype=torch.int32, device=DEVICE)
    
    total_frames_processed = 0
    
    while True:
        # First, receive a small tensor to check for termination or skip
        dist.recv(termination_signal, src=args.io_gpu)
        
        if termination_signal.item() == -1:
            break  # Termination signal received
        elif termination_signal.item() == 0:
            continue  # No frames to process this iteration
        
        frames = torch.empty((BATCH_SIZE, vinfo_height, vinfo_width, 3), dtype=torch.uint8, device=DEVICE)
        dist.recv(frames, src=args.io_gpu)
        
        # Process only the received frames
        actual_batch_size = frames.shape[0]
        frames = frames[:actual_batch_size]
        
        frames = frames.to(DTYPE).contiguous()  # Ensure contiguity after type conversion
        frames = preprocess_batch(frames, mean, std)
        
        with torch.no_grad():
            frames = depth_anything(frames)
        
        frames = F.interpolate(
            frames[None],
            (vinfo_height, vinfo_width),
            mode="bicubic",
            align_corners=False,
        )[0]
        
        frames = (
            (frames - frames.min())
            / (frames.max() - frames.min())
            * 255.0
        )
        frames = torch.repeat_interleave(frames.unsqueeze(-1), 3, dim=-1).to(dtype=torch.uint8)
        
        dist.send(frames.contiguous(), dst=args.io_gpu)
        
        total_frames_processed += actual_batch_size
    
    print(f"Worker {rank} processed {total_frames_processed} frames")

def setup_input_process(filepath):
    return (
        ffmpeg.input(filepath, threads=0, thread_queue_size=8192)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="quiet")
        .global_args("-hwaccel", "cuda")
        .run_async(pipe_stdout=True)
    )

def setup_output_process(args, filepath, vinfo, output_width):
    filename = os.path.basename(filepath)
    timestamp = time.strftime("%H%M%S")
    output_name = filename[: filename.rfind(".")] + f"_video_depth_{timestamp}"
    local_output_path = os.path.join(args.outdir, output_name)
    
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
        preset="p7",
        threads=0,
        framerate=vinfo.framerate,
        s=f"{output_width}x{vinfo.height}",
        pix_fmt="yuv420p",
        loglevel="quiet",
    )
    
    return output.overwrite_output().run_async(pipe_stdin=True)

if __name__ == "__main__":
    args = Args()
    torch.multiprocessing.spawn(run_depth_estimation, args=(args,), nprocs=args.world_size)