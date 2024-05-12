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

