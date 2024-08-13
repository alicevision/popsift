# First-order language CUDA requires at least CMake 3.18
cmake_minimum_required(VERSION 3.24)

# Default seletion of CUDA Compute Capabilities.
# This must be called before project() or cmake sets it to the oldest non-deprecated CC
# "all" and "all-major" work for Intel and perhaps for ARM with discrete GPUs, but not Tegra and Jetson.
if(EXISTS "/etc/nv_tegra_release")
  # The CC list for Tegras and Jetson will require manual updates
  set(CMAKE_CUDA_ARCHITECTURES "53;62;72;87"
      CACHE
      STRING "Which CUDA CCs to support: native, all, all-major or an explicit list delimited by semicolons")
else()
  # The CC list for discrete GPUs will require CMake updates
  set(CMAKE_CUDA_ARCHITECTURES "all-major"
      CACHE
      STRING "Which CUDA CCs to support: native, all, all-major or an explicit list delimited by semicolons")
endif()

project(PopSift VERSION 1.0.0 LANGUAGES CXX CUDA)

# Policy to support CUDA as a first-order language for CMake.
# Since CMake 3.18. See https://cmake.org/cmake/help/latest/policy/CMP0104.html
cmake_policy(SET CMP0104 NEW)

# Set build path as a folder named as the platform (linux, windows, darwin...) plus the processor type
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

option(PopSift_BUILD_EXAMPLES "Build PopSift applications."  ON)
option(PopSift_BUILD_DOCS "Build PopSift documentation."  OFF)
option(PopSift_USE_NVTX_PROFILING     "Use CUDA NVTX for profiling." OFF)
option(PopSift_ERRCHK_AFTER_KERNEL     "Synchronize and check CUDA error after every kernel." OFF)
option(PopSift_USE_POSITION_INDEPENDENT_CODE "Generate position independent code." ON)
option(PopSift_USE_GRID_FILTER "Switch off grid filtering to massively reduce compile time while debugging other things." ON)
option(PopSift_NVCC_WARNINGS "Switch on several additional warning for CUDA nvcc" OFF)
option(PopSift_USE_TEST_CMD "Add testing step for functional verification" OFF)
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

  # Auto-build dll exports on Windows
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
else()
  message(STATUS "BUILD_SHARED_LIBS OFF")

  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
endif()

# Require threads because of std::thread.
find_package(Threads REQUIRED)

###################
#  CUDA
###################
include(CheckLanguage)
check_language(CUDA)

# Use this if necessary: "cmake -DCUDAToolkit_ROOT=/some/path"
# target_link_libraries(binary_linking_to_cudart PRIVATE CUDA::cudart)
find_package(CUDAToolkit)

message(STATUS "CUDA Version is ${CUDAToolkit_VERSION}")
set(CUDA_VERSION ${CUDAToolkit_VERSION})

if(PopSift_USE_NVTX_PROFILING)
  message(STATUS "PROFILING CPU CODE: NVTX is in use")
endif()

if(PopSift_ERRCHK_AFTER_KERNEL)
  message(STATUS "Synchronizing and checking errors after every kernel call")
  list(APPEND CUDA_NVCC_FLAGS "-DERRCHK_AFTER_KERNEL")
endif()

# This may not be required any more.
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

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

set(PopSift_CXX_STANDARD 17) # Thrust/CUB requires C++14 starting with CUDA SDK 11
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


if(CUDA_VERSION VERSION_GREATER_EQUAL "9.0")
  set(PopSift_HAVE_SHFL_DOWN_SYNC   1)
else()
  set(PopSift_HAVE_SHFL_DOWN_SYNC   0)
endif()

if(NOT PopSift_USE_GRID_FILTER)
  message(STATUS "Disabling grid filter compilation")
  set(DISABLE_GRID_FILTER   1)
else()
  set(DISABLE_GRID_FILTER   0)
endif()

if(PopSift_USE_NVTX_PROFILING)
  # library required for NVTX profiling of the CPU
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
message(STATUS "Install path: " ${CMAKE_INSTALL_PREFIX})
message(STATUS "Testing step: " ${PopSift_USE_TEST_CMD})

message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_ID = ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

if(PopSift_USE_TEST_CMD)
  message(STATUS "Path for test input: " ${PopSift_TESTFILE_PATH})
endif()
message("\n******************************************")
message("\n")
