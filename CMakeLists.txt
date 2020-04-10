# CMake below 3.4 does not work with CUDA separable compilation at all
cmake_minimum_required(VERSION 3.4)

project(PopSift VERSION 1.0.0)

OPTION(PopSift_BUILD_EXAMPLES "Build PopSift applications."  ON)
OPTION(PopSift_USE_NVTX_PROFILING     "Use CUDA NVTX for profiling." OFF)
OPTION(PopSift_ERRCHK_AFTER_KERNEL     "Synchronize and check CUDA error after every kernel." OFF)
OPTION(PopSift_USE_POSITION_INDEPENDENT_CODE "Generate position independent code." ON)
OPTION(PopSift_USE_GRID_FILTER "Switch off grid filtering to massively reduce compile time while debugging other things." ON)
OPTION(PopSift_USE_NORMF "The __normf function computes Euclidian distance on large arrays. Fast but stability is uncertain." OFF)
OPTION(PopSift_USE_TEST_CMD "Add testing step for functional verification" OFF)
OPTION(PopSift_BOOST_USE_STATIC_LIBS "Link examples with static Boost libraries" OFF)
OPTION(PopSift_NVCC_WARNINGS "Switch on several additional warning for CUDA nvcc" OFF)

if(PopSift_USE_POSITION_INDEPENDENT_CODE)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

# set(CMAKE_BUILD_TYPE Debug)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "Build type not set, building in Release configuration")
else()
  message(STATUS "Building in ${CMAKE_BUILD_TYPE} configuration")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -G")
# set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG}   -G")

if(BUILD_SHARED_LIBS)
  message(STATUS "BUILD_SHARED_LIBS ON")
  # Need to declare CUDA_USE_STATIC_CUDA_RUNTIME as an option to ensure that it is not overwritten in FindCUDA.
  option(CUDA_USE_STATIC_CUDA_RUNTIME "Use the static version of the CUDA runtime library if available" OFF)
  set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
  # Workaround to force deactivation of cuda static runtime for cmake < 3.10
  set(CUDA_cudart_static_LIBRARY 0)
  # Auto-build dll exports on Windows
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
  message(STATUS "BUILD_SHARED_LIBS OFF")
  option(CUDA_USE_STATIC_CUDA_RUNTIME "Use the static version of the CUDA runtime library if available" ON)
  set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
endif()

find_package(CUDA 7.0 REQUIRED)

if(NOT CUDA_FOUND)
  message(FATAL_ERROR "Could not find CUDA >= 7.0")
endif()

#
# Default setting of the CUDA CC versions to compile.
# Shortening the lists saves a lot of compile time.
#
if(CUDA_VERSION_MAJOR GREATER 7)
  set(PopSift_CUDA_CC_LIST_BASIC 30 35 50 52 60 61 62)
else()
  set(PopSift_CUDA_CC_LIST_BASIC 30 35 50 52 )
endif()
set(PopSift_CUDA_CC_LIST ${PopSift_CUDA_CC_LIST_BASIC} CACHE STRING "CUDA CC versions to compile")

if(PopSift_USE_NVTX_PROFILING)
  message(STATUS "PROFILING CPU CODE: NVTX is in use")
endif()

if(PopSift_ERRCHK_AFTER_KERNEL)
  message(STATUS "Synchronizing and checking errors after every kernel call")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-DERRCHK_AFTER_KERNEL")
endif()

set(CUDA_SEPARABLE_COMPILATION ON)

if(UNIX AND NOT APPLE)
  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-Xcompiler;-rdynamic;-lineinfo")
  # set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-Xptxas;-v")
  # set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-Xptxas;-warn-double-usage")
  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};--keep")
  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};--source-in-ptx")
endif()

# The following if should not be necessary, but apparently there is a bug in FindCUDA.cmake that
# generate an empty string in the nvcc command line causing the compilation to fail.
# see https://gitlab.kitware.com/cmake/cmake/issues/16411
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Building in debug mode")
  set(CUDA_NVCC_FLAGS_DEBUG   "${CUDA_NVCC_FLAGS_DEBUG};-G")
endif()
set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-O3")

if(PopSift_USE_POSITION_INDEPENDENT_CODE)
  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-Xcompiler;-fPIC")
endif()

#
# Add all requested CUDA CCs to the command line for offline compilation
#
list(SORT PopSift_CUDA_CC_LIST)
foreach(PopSift_CC_VERSION ${PopSift_CUDA_CC_LIST})
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode;arch=compute_${PopSift_CC_VERSION},code=sm_${PopSift_CC_VERSION}")
endforeach()

#
# Use the highest request CUDA CC for CUDA JIT compilation
#
list(LENGTH PopSift_CUDA_CC_LIST PopSift_CC_LIST_LEN)
MATH(EXPR PopSift_CC_LIST_LEN "${PopSift_CC_LIST_LEN}-1")
list(GET PopSift_CUDA_CC_LIST ${PopSift_CC_LIST_LEN} PopSift_CUDA_CC_LIST_LAST)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode;arch=compute_${PopSift_CUDA_CC_LIST_LAST},code=compute_${PopSift_CUDA_CC_LIST_LAST}")

# default stream legacy implies that the 0 stream synchronizes all streams
# default stream per-thread implies that each host thread has one non-synchronizing 0-stream
# currently, the code requires legacy mode
set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};--default-stream;legacy")
# set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};--default-stream;per-thread")

message(STATUS "CUDA Version is ${CUDA_VERSION}")
message(STATUS "Compiling for CUDA CCs: ${PopSift_CUDA_CC_LIST}")
if( ( CUDA_VERSION VERSION_EQUAL "7.5" ) OR ( CUDA_VERSION VERSION_GREATER "7.5") )
  if(PopSift_NVCC_WARNINGS)
    set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-Xptxas;-warn-lmem-usage")
    set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-Xptxas;-warn-spills")
    set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-Xptxas;--warn-on-local-memory-usage")
    set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-Xptxas;--warn-on-spills")
  endif()
endif()

if(PopSift_USE_NORMF AND CUDA_VERSION VERSION_GREATER "7.4")
  set(HAVE_NORMF   1)
else()
  set(HAVE_NORMF   0)
endif()

if( ( CUDA_VERSION VERSION_EQUAL "9.0" ) OR ( CUDA_VERSION VERSION_GREATER "9.0") )
  set(HAVE_SHFL_DOWN_SYNC   1)
else()
  set(HAVE_SHFL_DOWN_SYNC   0)
endif()

if(NOT PopSift_USE_GRID_FILTER)
  message(STATUS "Disabling grid filter compilation")
  set(DISABLE_GRID_FILTER   1)
else()
  set(DISABLE_GRID_FILTER   0)
endif()

# library required for CUDA dynamic parallelism, forgotten by CMake 3.4
cuda_find_library_local_first(CUDA_CUDADEVRT_LIBRARY cudadevrt "\"cudadevrt\" library")

if(PopSift_USE_NVTX_PROFILING)
  # library required for NVTX profiling of the CPU
  cuda_find_library_local_first(CUDA_NVTX_LIBRARY nvToolsExt "NVTX library")
  add_definitions(-DUSE_NVTX)
endif()

add_subdirectory(src)

if(PopSift_USE_TEST_CMD)
  add_subdirectory(testScripts)
endif()

########### Add uninstall target ###############
CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)
ADD_CUSTOM_TARGET(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake") 

