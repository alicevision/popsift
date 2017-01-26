# CMake below 3.4 does not work with CUDA separable compilation at all
cmake_minimum_required(VERSION 3.4)

project(PopSift VERSION 1.0.0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "Build type not set, building in Release configuration")
else()
  message(STATUS "Building in ${CMAKE_BUILD_TYPE} configuration")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_DEBUG} -O3")
set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_DEBUG}   -O3")

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DGRIFF_DEBUG")
#set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG}   -DGRIFF_DEBUG")

find_package(Boost 1.53.0 REQUIRED COMPONENTS system filesystem)

find_package(CUDA 7.0)

if(NOT CUDA_FOUND)
  message(FATAL_ERROR "Could not find CUDA >= 7.0")
endif()

set(CUDA_SEPARABLE_COMPILATION ON)

# The following if should not be necessary, but apparently there is a bug in FindCUDA.cmake that
# generate an empty string in the nvcc command line causing the compilation to fail.
# see https://gitlab.kitware.com/cmake/cmake/issues/16411
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Building in debug mode")
  set(CUDA_NVCC_FLAGS_DEBUG   "${CUDA_NVCC_FLAGS_DEBUG};-G;-g")
else()
  message(STATUS "Building in release mode")
  set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-O3;-DNDEBUG")
  # set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-G;-g;-DNDEBUG")
endif()
set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode;arch=compute_30,code=sm_30")
set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35")
set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode;arch=compute_50,code=sm_50")
set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode;arch=compute_52,code=sm_52")
set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode;arch=compute_52,code=compute_52")
# default stream legacy implies that the 0 stream synchronizes all streams
set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};--default-stream;legacy")
# default stream per-thread implies that each host thread has one non-synchronizing 0-stream
# set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};--default-stream;per-thread")
# print local memory usage per kernel: -Xptxas;-v
# CUDA >= 7.5: -Xptxas;--warn-on-local-memory-usage;-Xptxas;--warn-on-spills
message(STATUS "CUDA Version is ${CUDA_VERSION}")
if(CUDA_VERSION>=7.5)
  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-Xptxas;--warn-on-local-memory-usage")
  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-Xptxas;--warn-on-spills")
else(CUDA_VERSION>=7.5)
endif(CUDA_VERSION>=7.5)

# library required for CUDA dynamic parallelism, forgotten by CMake 3.4
cuda_find_library_local_first(CUDA_CUDADEVRT_LIBRARY cudadevrt "\"cudadevrt\" library")

add_subdirectory(src)

########### Add uninstall target ###############
CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)
ADD_CUSTOM_TARGET(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake") 

