# CMake below 3.4 does not work with CUDA separable compilation at all
cmake_minimum_required(VERSION 3.12)

project(PopSift VERSION 1.0.0 LANGUAGES CXX)

# Set build path as a folder named as the platform (linux, windows, darwin...) plus the processor type
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

option(PopSift_BUILD_EXAMPLES "Build PopSift applications."  ON)
option(PopSift_BUILD_DOCS "Build PopSift documentation."  OFF)
option(PopSift_USE_NVTX_PROFILING     "Use CUDA NVTX for profiling." OFF)
option(PopSift_ERRCHK_AFTER_KERNEL     "Synchronize and check CUDA error after every kernel." OFF)
option(PopSift_USE_POSITION_INDEPENDENT_CODE "Generate position independent code." ON)
option(PopSift_USE_GRID_FILTER "Switch off grid filtering to massively reduce compile time while debugging other things." ON)
option(PopSift_USE_NORMF "The __normf function computes Euclidean distance on large arrays. Fast but stability is uncertain." OFF)
option(PopSift_NVCC_WARNINGS "Switch on several additional warning for CUDA nvcc" OFF)
option(PopSift_USE_TEST_CMD "Add testing step for functional verification" OFF)
option(PopSift_NO_DEPRECATED_CUDA_SM_WARNINGS "Suppress warnings about soon to be deprecated cuda SM" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

if(PopSift_USE_POSITION_INDEPENDENT_CODE AND NOT MSVC)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

# set(CMAKE_BUILD_TYPE Debug)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "Build type not set, building in Release configuration")
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
else()
  message(STATUS "Building in ${CMAKE_BUILD_TYPE} configuration")
endif()

# ensure the proper linker flags when building the static version on MSVC
if(MSVC AND NOT BUILD_SHARED_LIBS)
  foreach(config "DEBUG" "RELEASE" "MINSIZEREL" "RELWITHDEBINFO")
    string(REPLACE /MD /MT CMAKE_C_FLAGS_${config} "${CMAKE_C_FLAGS_${config}}")
    string(REPLACE /MD /MT CMAKE_CXX_FLAGS_${config} "${CMAKE_CXX_FLAGS_${config}}")
    message(STATUS "CMAKE_C_FLAGS_${config} ${CMAKE_C_FLAGS_${config}}")
    message(STATUS "CMAKE_CXX_FLAGS_${config} ${CMAKE_CXX_FLAGS_${config}}")
  endforeach()
endif()

# ==============================================================================
# GNUInstallDirs CMake module
# - Define GNU standard installation directories
# - Provides install directory variables as defined by the GNU Coding Standards.
# ==============================================================================
include(GNUInstallDirs)


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

# Require threads because of std::thread.
find_package(Threads REQUIRED)

###################
#  CUDA
###################
find_package(CUDA 7.0 REQUIRED)

if(NOT CUDA_FOUND)
  message(FATAL_ERROR "Could not find CUDA >= 7.0")
endif()

message(STATUS "CUDA Version is ${CUDA_VERSION}")

include(ChooseCudaCC)
if(NOT DEFINED PopSift_CUDA_CC_LIST)
  chooseCudaCC(PopSift_CUDA_CC_LIST_BASIC
               PopSift_CUDA_GENCODE_FLAGS
               MIN_CC 30
               MIN_CUDA_VERSION 7.0)
  set(PopSift_CUDA_CC_LIST ${PopSift_CUDA_CC_LIST_BASIC} CACHE STRING "CUDA CC versions to compile")
else()
  getFlagsForCudaCCList(PopSift_CUDA_CC_LIST
                        PopSift_CUDA_GENCODE_FLAGS)
endif()
list(APPEND CUDA_NVCC_FLAGS "${PopSift_CUDA_GENCODE_FLAGS}")

if(PopSift_USE_NVTX_PROFILING)
  message(STATUS "PROFILING CPU CODE: NVTX is in use")
endif()

if(PopSift_ERRCHK_AFTER_KERNEL)
  message(STATUS "Synchronizing and checking errors after every kernel call")
  list(APPEND CUDA_NVCC_FLAGS "-DERRCHK_AFTER_KERNEL")
endif()

set(CUDA_SEPARABLE_COMPILATION ON)

if(PopSift_NO_DEPRECATED_CUDA_SM_WARNINGS)
  list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")
endif()

if(UNIX AND NOT APPLE)
  list(APPEND CUDA_NVCC_FLAGS         "-Xcompiler;-rdynamic")
  # set(CUDA_NVCC_FLAGS       "${CUDA_NVCC_FLAGS};-Xptxas;-v")
  # set(CUDA_NVCC_FLAGS       "${CUDA_NVCC_FLAGS};-Xptxas;-warn-double-usage")
  list(APPEND CUDA_NVCC_FLAGS_DEBUG   "--keep")
  list(APPEND CUDA_NVCC_FLAGS_DEBUG   "--source-in-ptx")
endif()

# The following if should not be necessary, but apparently there is a bug in FindCUDA.cmake that
# generate an empty string in the nvcc command line causing the compilation to fail.
# see https://gitlab.kitware.com/cmake/cmake/issues/16411
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Building in debug mode")
  list(APPEND CUDA_NVCC_FLAGS_DEBUG   "-G")
endif()
list(APPEND CUDA_NVCC_FLAGS_RELEASE "-O3")

if(PopSift_USE_POSITION_INDEPENDENT_CODE AND NOT MSVC)
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler;-fPIC")
endif()

# this is to ensure that on MSVC the flags for the linker are properly propagate even to the intermediate
# linking step. This seems not the case e.g. on vcpkg using ninja build.
if(MSVC)
  if(BUILD_SHARED_LIBS)
    set(PopSift_MVSC_LINKER "/MD")
  else()
    set(PopSift_MVSC_LINKER "/MT")
  endif()
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(PopSift_MVSC_LINKER "${PopSift_MVSC_LINKER}d")
  endif()
  list(APPEND CUDA_NVCC_FLAGS   -Xcompiler ${PopSift_MVSC_LINKER})
endif()

list(APPEND CUDA_NVCC_FLAGS "-lineinfo")

# default stream per-thread implies that each host thread has one non-synchronizing 0-stream
# currently, the code requires legacy mode
list(APPEND CUDA_NVCC_FLAGS "--default-stream;legacy")
# set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};--default-stream;per-thread")

if(CUDA_VERSION VERSION_GREATER_EQUAL "7.5")
  if(PopSift_NVCC_WARNINGS)
    list(APPEND CUDA_NVCC_FLAGS_RELEASE "-Xptxas;-warn-lmem-usage")
    list(APPEND CUDA_NVCC_FLAGS_RELEASE "-Xptxas;-warn-spills")
    list(APPEND CUDA_NVCC_FLAGS_RELEASE "-Xptxas;--warn-on-local-memory-usage")
    list(APPEND CUDA_NVCC_FLAGS_RELEASE "-Xptxas;--warn-on-spills")
  endif()
endif()

set(PopSift_CXX_STANDARD 14) # Thrust/CUB requires C++14 starting with CUDA SDK 11
if(CUDA_VERSION_MAJOR LESS_EQUAL 8)
  set(PopSift_CXX_STANDARD 11)
endif()

if(NOT MSVC)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++${PopSift_CXX_STANDARD}")
      list(APPEND CUDA_NVCC_FLAGS "-std=c++${PopSift_CXX_STANDARD}")
endif()
set(CMAKE_CXX_STANDARD ${PopSift_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD ${PopSift_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


if(PopSift_USE_NORMF AND CUDA_VERSION VERSION_GREATER_EQUAL "7.5")
  set(PopSift_HAVE_NORMF   1)
else()
  set(PopSift_HAVE_NORMF   0)
endif()

if(CUDA_VERSION VERSION_GREATER_EQUAL "9.0")
  set(PopSift_HAVE_SHFL_DOWN_SYNC   1)
  set(PopSift_HAVE_COOPERATIVE_GROUPS   1)
else()
  set(PopSift_HAVE_SHFL_DOWN_SYNC   0)
  set(PopSift_HAVE_COOPERATIVE_GROUPS   0)
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
  set(PopSift_USE_NVTX 1)
else()
  set(PopSift_USE_NVTX 0)
endif()

add_subdirectory(src)

if(PopSift_BUILD_DOCS)
  add_subdirectory(doc)
endif()

set(PopSift_TESTFILE_PATH "popsift-samples/datasets/sample/big_set/" CACHE STRING "Base directory where your test files are stored")
if(PopSift_USE_TEST_CMD)
  if(NOT IS_ABSOLUTE("${PopSift_TESTFILE_PATH}"))
    get_filename_component(PopSift_TESTFILES "${PopSift_TESTFILE_PATH}" ABSOLUTE)
    set(PopSift_TESTFILE_PATH "${PopSift_TESTFILES}")
  endif()

  add_subdirectory(testScripts)
endif()

########### Add uninstall target ###############
CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)
ADD_CUSTOM_TARGET(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake")


######################################
# SUMMARY
######################################
message("\n")
message("******************************************")
message("Building configuration:\n")
message(STATUS "PopSift version: " ${PROJECT_VERSION})
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
message(STATUS "Build Shared libs: " ${BUILD_SHARED_LIBS})
message(STATUS "Build examples: " ${PopSift_BUILD_EXAMPLES})
message(STATUS "Build documentation: " ${PopSift_BUILD_DOCS})
message(STATUS "Generate position independent code: " ${CMAKE_POSITION_INDEPENDENT_CODE})
message(STATUS "Use CUDA NVTX for profiling: " ${PopSift_USE_NVTX_PROFILING})
message(STATUS "Synchronize and check CUDA error after every kernel: " ${PopSift_ERRCHK_AFTER_KERNEL})
message(STATUS "Grid filtering: " ${PopSift_USE_GRID_FILTER})
message(STATUS "Additional warning for CUDA nvcc: " ${PopSift_NVCC_WARNINGS})
message(STATUS "Compiling for CUDA CCs: ${PopSift_CUDA_CC_LIST}")
message(STATUS "Install path: " ${CMAKE_INSTALL_PREFIX})
message(STATUS "Testing step: " ${PopSift_USE_TEST_CMD})
if(PopSift_USE_TEST_CMD)
  message(STATUS "Path for test input: " ${PopSift_TESTFILE_PATH})
endif()
message("\n******************************************")
message("\n")
