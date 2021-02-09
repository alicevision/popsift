ARG CUDA_TAG=11.2.0
ARG OS_TAG=18.04
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu${OS_TAG}
LABEL maintainer="AliceVision Team alicevision@googlegroups.com"

# use CUDA_TAG to select the image version to use
# see https://hub.docker.com/r/nvidia/cuda/
#
# For example, to create a ubuntu 16.04 with cuda 8.0 for development, use
# docker build --build-arg CUDA_TAG=8.0 --tag alicevision/popsift-deps:cuda${CUDA_TAG}-ubuntu${OS_TAG} .
#
# then execute with nvidia docker (https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))
# docker run -it --runtime=nvidia popsift_deps


# OS/Version (FILE): cat /etc/issue.net
# Cuda version (ENV): $CUDA_VERSION

# System update
RUN apt-get clean && apt-get update && apt-get install -y --no-install-recommends\
        build-essential \
        wget \
        unzip \
        libtool \
        automake \
        libssl-dev \
        libjpeg-turbo8-dev \
        libdevil-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libboost-program-options-dev \
        libboost-thread-dev \
 && rm -rf /var/lib/apt/lists/*

 # Manually install cmake
WORKDIR /tmp/cmake
ENV CMAKE_VERSION=3.17
ENV CMAKE_VERSION_FULL=${CMAKE_VERSION}.2
RUN wget https://cmake.org/files/v3.17/cmake-${CMAKE_VERSION_FULL}.tar.gz && \
    tar zxvf cmake-${CMAKE_VERSION_FULL}.tar.gz && \
    cd cmake-${CMAKE_VERSION_FULL} && \
    ./bootstrap --prefix=/usr/local  -- -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_USE_OPENSSL:BOOL=ON && \
    make -j$(nproc) install && \
    cd /tmp && \
    rm -rf cmake

COPY . /opt/popsift
WORKDIR /opt/popsift/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make install -j $(nproc) && \
    cd /opt && \
    rm -rf popsift
