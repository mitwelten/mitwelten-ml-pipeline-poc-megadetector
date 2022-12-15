# poc-megadetector-pipeline
This repo implements a data pipeline for the megadetector model

## Setup and Installation
Prerequisites:
- Nvidia-Docker (See [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

### App


Build Image
```bash
docker build . -f app.Dockerfile --tag megadetector-gpu-image
```
Run Container
```bash
docker run \
    -it \
    -v $PWD:/root/ \
    --name megadetector-gpu \
    -p 8888:8888 \
    --gpus all \
    megadetector-gpu-image
```
### Flow

Build Image
```bash
docker build . -f flow.Dockerfile --tag megadetector-gpu-image
```
Run Container
```bash
docker run \
    -it \
    -v $PWD:/root/ \
    --name megadetector-flow \
    -p 8888:8888 \
    --gpus all \
    megadetector-gpu-image
```