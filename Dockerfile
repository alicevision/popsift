ARG CUDA_TAG=9.2-devel
FROM nvidia/cuda:$CUDA_TAG

ARG		PYTHON

# OS/Version (FILE): cat /etc/issue.net
# Cuda version (ENV): $CUDA_VERSION

# System update
RUN apt-get clean && apt-get update && apt-get install -y \
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
        libboost-thread-dev

ADD . /opt/popsift
RUN cd /opt/popsift && mkdir build && cd build && cmake .. && make install -j8
