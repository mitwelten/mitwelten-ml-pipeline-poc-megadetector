FROM ultralytics/yolov5:latest

RUN apt-get update
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Zurich
RUN apt-get install -y wget tzdata

WORKDIR /root

# download model weights
RUN python src/scripts/download_weights.py

# copy content from local repo
COPY . .

# install relevant packages
RUN python -m pip install -r requirements.txt

COPY model_weights/ model_weights/

ENTRYPOINT [ "bash" ]