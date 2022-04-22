FROM tensorflow/tensorflow:2.7.0-gpu

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    protobuf-compiler \
    python3-pip \
    python3-pil \
    python3-lxml \
    python3-opencv \
    libgl1

RUN apt-get install -y \
    git \    
    wget

RUN python -m pip install -U pip

RUN pip install jupyter matplotlib pyyaml==5.4.1 imgaug==0.4.0 pascal_voc_writer==0.1.4

RUN git clone https://github.com/tensorflow/models.git /tensorflow/models

WORKDIR /tensorflow/models/research

RUN protoc object_detection/protos/*.proto --python_out=.

RUN export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

RUN cp object_detection/packages/tf2/setup.py ./

RUN python -m pip install .

ENV TF_CPP_MIN_LOG_LEVEL 3
