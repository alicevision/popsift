ARG CUDA_TAG=9.2-devel
FROM nvidia/cuda:$CUDA_TAG
LABEL maintainer="AliceVision Team alicevision@googlegroups.com"

# use CUDA_TAG to select the image version to use
# see https://hub.docker.com/r/nvidia/cuda/
#
# For example, to create a ubuntu 16.04 with cuda 8.0 for development, use
# docker build --build-arg CUDA_TAG=8.0-devel --tag popsift .
#
# then execute with nvidia docker (https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))
# docker run -it --runtime=nvidia popsift


# OS/Version (FILE): cat /etc/issue.net
# Cuda version (ENV): $CUDA_VERSION

# System update
RUN apt-get clean && apt-get update && apt-get install -y --no-install-recommends\
				build-essential \
				cmake \
				git \
				wget \
				unzip \
				yasm \
				pkg-config \
				libtool \
				nasm \
				automake \
				libpng12-dev \
				libjpeg-turbo8-dev \
        libdevil-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libboost-program-options-dev \
        libboost-thread-dev \
 && rm -rf /var/lib/apt/lists/*

COPY . /opt/popsift
WORKDIR /opt/popsift
RUN mkdir build && cd build && cmake .. && make install -j
