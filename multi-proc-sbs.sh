process_file() {
    filename="$1"
    python to_stereo.py "$filename"
}

for i in {0..15}; do
    filename="sw-depth-$(printf "%03d" $i).mp4"
    process_file "$filename" &
done

wait