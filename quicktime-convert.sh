ffmpeg -i sw.mkv -c:v libx264 -threads 0 -profile:v high -level 4.0 -pix_fmt yuv420p -c:a aac -b:a 256k -movflags +faststart sw_qt.mp4
