FROM ultralytics/yolov5:latest

RUN apt-get update
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Zurich
RUN apt-get install -y wget tzdata

WORKDIR /root

# download model weigths
RUN python src/scripts/download_weights.py

# copy content from local repo
COPY . .

# install relevant packages
RUN python -m pip install -r requirements.txt
RUN python -m pip install jupyterlab

# expose port for jupyter lab
EXPOSE 8888

ENTRYPOINT [ "bash" ]