![MIT License](https://img.shields.io/badge/Organization-Mitwelten-green)

# poc-megadetector-pipeline
This repo implements a data pipeline for the Megadetector model. The Megadetector is a generic object detection model trained on a large amount of camera trap images. It can detect animals, humans and vehicules.

## Setup and Installation
Prerequisites:
- Nvidia-Docker (See [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

### App
This build is for the reproducement of results and local testing.

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
This build is for the implementation as prefect flow.

Build Image
```bash
docker build . -f flow.Dockerfile --tag megadetector-flow-image
```
Run Container
```bash
docker run \
    -it \
    --name megadetector-flow \
    --gpus all \
    megadetector-fow-image
```

## Model weights
If the weights were not downloaded properly during the Dockerfile build, then download it manually.
Download the model weights from the official megadetector repo. Call from root:
```bash
python src/scripts/download_weights.py
```
This script will download the model weights and saves it to the folder `/model_weights`



## Sources

https://github.com/microsoft/CameraTraps/blob/main/megadetector.md
