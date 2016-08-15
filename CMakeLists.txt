# CMake below 3.4 does not work with CUDA separable compilation at all
CMAKE_MINIMUM_REQUIRED(VERSION 3.4)

PROJECT( PopSift )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

set(CMAKE_BUILD_TYPE Release) # Debug, Release

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_DEBUG} -O3")
set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_DEBUG}   -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DGRIFF_DEBUG")
set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG}   -DGRIFF_DEBUG")

find_package(Boost 1.53.0 REQUIRED COMPONENTS system filesystem)

if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  # link_directories(${Boost_LIB_DIR})
  add_definitions(${Boost_DEFINITIONS})
  message( STATUS "Found BOOST: ${Boost_LIBRARIES}" )
else(Boost_FOUND)
  message( FATAL_ERROR "Boost not found" )
endif(Boost_FOUND)

find_package(CUDA 7.0)

IF(CUDA_FOUND)
  SET(CUDA_SEPARABLE_COMPILATION ON)
  include_directories(${CUDA_INCLUDE_DIRS})
  add_definitions("-DWITH_CUDA")
  set(CUDA_NVCC_FLAGS_DEBUG   "${CUDA_NVCC_FLAGS_DEBUG};-G;-g")
  set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-O3")
  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode;arch=compute_30,code=sm_30")
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

  include_directories("${CMAKE_SOURCE_DIR}/popsift/Cuda-7.0-cub")
ELSE(CUDA_FOUND)
  message( FATAL_ERROR "Could not find CUDA >= 7.0" )
ENDIF(CUDA_FOUND)

add_subdirectory(src)

