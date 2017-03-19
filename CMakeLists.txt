# CMake below 3.4 does not work with CUDA separable compilation at all
cmake_minimum_required(VERSION 3.4)

project(PopSift VERSION 1.0.0)

OPTION(PopSift_BUILD_EXAMPLES "Build PopSift applications."  ON)
OPTION(USE_NVTX_PROFILING     "Use CUDA NVTX for profiling." OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

# set(CMAKE_BUILD_TYPE Debug)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "Build type not set, building in Release configuration")
else()
  message(STATUS "Building in ${CMAKE_BUILD_TYPE} configuration")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -O3")

# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -G")
# set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG}   -G")

find_package(Boost 1.53.0 REQUIRED COMPONENTS system thread)
if(WIN32)
  add_definitions("-DBOOST_ALL_NO_LIB")
  link_directories(Boost_LIBRARRY_DIR_DEBUG)
  link_directories(Boost_LIBRARRY_DIR_RELEASE)
endif(WIN32)

find_package(CUDA 7.0)

if(NOT CUDA_FOUND)
  message(FATAL_ERROR "Could not find CUDA >= 7.0")
endif()

if(USE_NVTX_PROFILING)
  message(STATUS "PROFILING CPU CODE: NVTX is in use")
endif(USE_NVTX_PROFILING)

set(CUDA_SEPARABLE_COMPILATION ON)

if(UNIX AND NOT APPLE)
  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-Xcompiler;-rdynamic;-lineinfo")
  # set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-Xptxas;-v")
  # set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-Xptxas;-warn-double-usage")
  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};--keep")
  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};--source-in-ptx")
endif(UNIX AND NOT APPLE)

# The following if should not be necessary, but apparently there is a bug in FindCUDA.cmake that
# generate an empty string in the nvcc command line causing the compilation to fail.
# see https://gitlab.kitware.com/cmake/cmake/issues/16411
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Building in debug mode")
  set(CUDA_NVCC_FLAGS_DEBUG   "${CUDA_NVCC_FLAGS_DEBUG};-G")
endif()
set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-O3")
set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode;arch=compute_30,code=sm_30")
set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35")
set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode;arch=compute_50,code=sm_50")
set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode;arch=compute_52,code=sm_52")
set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode;arch=compute_52,code=compute_52")

# default stream legacy implies that the 0 stream synchronizes all streams
# default stream per-thread implies that each host thread has one non-synchronizing 0-stream
# currently, the code requires legacy mode
set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};--default-stream;legacy")
# set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};--default-stream;per-thread")

message(STATUS "CUDA Version is ${CUDA_VERSION}")
if(CUDA_VERSION>=7.5)
  set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-Xptxas;-warn-lmem-usage")
  set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-Xptxas;-warn-spills")
  set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-Xptxas;--warn-on-local-memory-usage")
  set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-Xptxas;--warn-on-spills")
else(CUDA_VERSION>=7.5)
endif(CUDA_VERSION>=7.5)

# library required for CUDA dynamic parallelism, forgotten by CMake 3.4
cuda_find_library_local_first(CUDA_CUDADEVRT_LIBRARY cudadevrt "\"cudadevrt\" library")

if(USE_NVTX_PROFILING)
  # library required for NVTX profiling of the CPU
  cuda_find_library_local_first(CUDA_NVTX_LIBRARY nvToolsExt "NVTX library")
  add_definitions(-DUSE_NVTX)
endif(USE_NVTX_PROFILING)

add_subdirectory(src)

########### Add uninstall target ###############
CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)
ADD_CUSTOM_TARGET(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake") 

