ARG CUDA_TAG=10.2
ARG OS_TAG=18.04
FROM alicevision/popsift-deps:cuda${CUDA_TAG}-ubuntu${OS_TAG}
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
COPY . /opt/popsift
WORKDIR /opt/popsift/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make install -j $(nproc) && \
    cd /opt && \
    rm -rf popsift
