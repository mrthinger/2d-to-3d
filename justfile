VERSION := "v0.1.0"

run:
    uvicorn main:app --reload

push:
    docker build --platform linux/amd64 \
    -t harbor.kymyth.com/kymyth/3d:latest \
    -t harbor.kymyth.com/kymyth/3d:{{VERSION}} \
    .
    docker push harbor.kymyth.com/kymyth/3d:latest
    docker push harbor.kymyth.com/kymyth/3d:{{VERSION}}

split-video:
    #!/usr/bin/env bash
    set -euxo pipefail
    duration=`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 build/misc/sw-depth.mp4`
    seg_time=`echo "$duration / 15.9" | bc`
    ffmpeg -i build/misc/sw-depth.mp4 -c copy -map 0 \
        -segment_time $seg_time -f segment -reset_timestamps 1 \
        build/split/sw-depth-%03d.mp4
    
combine-video:
    ffmpeg -f concat -safe 0 -i <(for f in build/split/sw-depth-*.mp4; do echo "file '$f'"; done) -c copy build/combine/sw.mp4