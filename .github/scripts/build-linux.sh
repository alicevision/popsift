#!/bin/bash
# Linux build functions for PopSift
# Usage: source this file and call the individual functions

setup_directories() {
    echo "Setting up build directories..."
    mkdir -p ./build_${BUILD_TYPE,,}
    mkdir -p ./build_as_3rdparty_${BUILD_TYPE,,}
    mkdir -p ../popsift_install_${BUILD_TYPE,,}
}

configure_cmake() {
    local build_type="$1"
    local deps_dir="$2"
    local build_dir="build_${build_type,,}"
    local install_dir="../popsift_install_${build_type,,}"
    
    echo "Configuring CMake for $build_type..."
    cd "./$build_dir"
    cmake .. \
     -DCMAKE_BUILD_TYPE="$build_type" \
     -DBUILD_SHARED_LIBS:BOOL=ON \
     -DCMAKE_PREFIX_PATH="$deps_dir" \
     -DPopSift_BUILD_DOCS:BOOL=OFF \
     -DCMAKE_INSTALL_PREFIX:PATH="$PWD/$install_dir"
    cd ..
}

build_and_install() {
    local build_type="$1"
    local build_dir="build_${build_type,,}"
    
    echo "Building and installing $build_type..."
    cd "./$build_dir"
    make -j$(nproc) install
    cd ..
}

build_as_third_party() {
    local build_type="$1"
    local deps_dir="$2"
    local build_dir="build_as_3rdparty_${build_type,,}"
    local install_dir="../popsift_install_${build_type,,}"
    
    echo "Testing third-party build for $build_type..."
    cd "./$build_dir"
    cmake ../src/application \
     -DBUILD_SHARED_LIBS:BOOL=ON \
     -DCMAKE_BUILD_TYPE="$build_type" \
     -DCMAKE_PREFIX_PATH:PATH="$PWD/$install_dir;$deps_dir"
    make -j$(nproc)
    cd ..
}
