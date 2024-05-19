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

segment:
    mediafilesegmenter -iso-fragmented -t 4 \
        -f build/playlist/sw \
        -z sw20_iframe.m3u8 \
        -b http://localhost:8443/build/playlist/sw \
        build/spatial/sw20.mp4

split-video:
    #!/usr/bin/env bash
    set -euxo pipefail
    duration=`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 build/misc/sw-depth.mp4`
    seg_time=`echo "$duration / 15.9" | bc`
    ffmpeg -i build/misc/sw-depth.mp4 -c copy -map 0 \
        -segment_time $seg_time -f segment -reset_timestamps 1 \
        build/split/sw-depth-%03d.mp4
    
combine-video:
    #!/usr/bin/env bash
    set -euxo pipefail
    tmpfile=$(mktemp)
    find ./build/split/sbs-hevc-88 -name "*.mp4" -print0 | sort -z -n -t - -k 3 | xargs -0 -I {} echo "file '$(pwd)/{}'" >> $tmpfile
    echo $tmpfile
    ffmpeg -f concat -safe 0 -i $tmpfile -c copy build/sbs/sw-hevc-sbs-88.mp4
    rm $tmpfile
