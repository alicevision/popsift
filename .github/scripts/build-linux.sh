#!/bin/bash
# Linux build functions for PopSift
# Usage: source this file and call the individual functions

# Creates the necessary build directories for different build types
# Uses BUILD_TYPE environment variable to create build and install directories
setup_directories() {
    echo "Setting up build directories..."
    mkdir -p ./build_"${BUILD_TYPE,,}"
    mkdir -p ./build_as_3rdparty_"${BUILD_TYPE,,}"
    mkdir -p ../popsift_install_"${BUILD_TYPE,,}"
}

# Configures CMake for PopSift build with specified options
# Parameters:
#   $1 - build_type: Release or Debug
#   $2 - deps_dir: Directory containing pre-installed dependencies (e.g., /opt/)
# Creates a build directory named <build_$build_type> and runs CMake configuration with PopSift-specific options
configure_cmake() {
    local build_type="$1"
    local deps_dir="$2"
    local build_dir="build_${build_type,,}"
    local install_dir="../popsift_install_${build_type,,}"
    
    echo "Configuring CMake for $build_type..."
    cd "./$build_dir" || exit
    cmake .. \
     -DCMAKE_BUILD_TYPE="$build_type" \
     -DBUILD_SHARED_LIBS:BOOL=ON \
     -DCMAKE_PREFIX_PATH="$deps_dir" \
     -DPopSift_BUILD_DOCS:BOOL=OFF \
     -DCMAKE_INSTALL_PREFIX:PATH="$PWD/$install_dir"
    cd ..
}

# Builds and installs PopSift for the specified build type
# Parameters:
#   $1 - build_type: Release or Debug
# Uses parallel build with all available CPU cores
build_and_install() {
    local build_type="$1"
    local build_dir="build_${build_type,,}"
    
    echo "Building and installing $build_type..."
    cd "./$build_dir" || exit
    cmake --build . --config "$build_type" --parallel
    cmake --install . --config "$build_type"
    cd ..
}

# Tests building PopSift applications as a third-party consumer
# This verifies that the installed PopSift can be found and used by external projects
# Parameters:
#   $1 - build_type: Release or Debug
#   $2 - deps_dir: Directory containing pre-installed dependencies
# Builds only the application from src/application using the installed PopSift library
build_as_third_party() {
    local build_type="$1"
    local deps_dir="$2"
    local build_dir="build_as_3rdparty_${build_type,,}"
    local install_dir="../popsift_install_${build_type,,}"
    
    echo "Testing third-party build for $build_type..."
    cd "./$build_dir" || exit
    cmake ../src/application \
     -DBUILD_SHARED_LIBS:BOOL=ON \
     -DCMAKE_BUILD_TYPE="$build_type" \
     -DCMAKE_PREFIX_PATH:PATH="$PWD/$install_dir;$deps_dir"
    cmake --build . --config "$build_type" --parallel
    cd ..
}
