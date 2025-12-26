cmake_minimum_required(VERSION 3.22)

project(PopSift VERSION 1.0.0 LANGUAGES CXX)

# ============================================================================
# SYCL Compiler Detection
# ============================================================================
set(SYCL_IMPL "AUTO" CACHE STRING "SYCL implementation: AUTO, INTEL, LLVM, or ACPP")

if(SYCL_IMPL STREQUAL "AUTO")
    if(CMAKE_CXX_COMPILER MATCHES "acpp")
        set(SYCL_IMPL "ACPP")
        message(STATUS "✓ Auto-detected: AdaptiveCpp")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        set(SYCL_IMPL "INTEL")
        message(STATUS "✓ Auto-detected: Intel DPC++")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(SYCL_IMPL "LLVM")
        message(STATUS "✓ Auto-detected: Custom LLVM SYCL")
    else()
        message(FATAL_ERROR "Could not detect SYCL compiler!")
    endif()
endif()

message(STATUS "SYCL Implementation: ${SYCL_IMPL}")
message(STATUS "Using compiler: ${CMAKE_CXX_COMPILER}")

# ============================================================================
# Auto-detect GPU Architecture
# ============================================================================
if(NOT DEFINED GPU_ARCH)
    # First, try to detect NVIDIA GPU
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE GPU_COMPUTE_CAP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE NVIDIA_RESULT
    )
    
    if(NVIDIA_RESULT EQUAL 0 AND GPU_COMPUTE_CAP)
        string(REPLACE "." "" GPU_ARCH_NUM ${GPU_COMPUTE_CAP})
        set(GPU_ARCH "sm_${GPU_ARCH_NUM}")
        set(GPU_VENDOR "NVIDIA")
        set(GPU_BACKEND "cuda")
        message(STATUS "✓ Auto-detected NVIDIA GPU: ${GPU_ARCH} (compute capability ${GPU_COMPUTE_CAP})")
    else()
        # Try to detect AMD GPU using rocminfo
        execute_process(
            COMMAND bash -c "rocminfo | grep 'Name:.*gfx' | head -n 1 | sed 's/.*Name: *//'"
            OUTPUT_VARIABLE ROCM_ARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE ROCM_RESULT
        )
        
        if(ROCM_RESULT EQUAL 0 AND ROCM_ARCH)
            set(GPU_ARCH "${ROCM_ARCH}")
            set(GPU_VENDOR "AMD")
            set(GPU_BACKEND "hip")
            message(STATUS "✓ Auto-detected AMD GPU: ${GPU_ARCH}")
        else()
            # Check if using Intel DPC++ - don't need arch specification
            if(SYCL_IMPL STREQUAL "INTEL")
                set(GPU_ARCH "generic")
                set(GPU_VENDOR "INTEL")
                set(GPU_BACKEND "level_zero")
                message(STATUS "✓ Intel DPC++ detected - using runtime device selection")
            else()
                # Fallback to safe default for NVIDIA
                set(GPU_ARCH "sm_70")
                set(GPU_VENDOR "NVIDIA")
                set(GPU_BACKEND "cuda")
                message(WARNING "Could not detect GPU, using default: ${GPU_ARCH}")
            endif()
        endif()
    endif()
else()
    # User specified GPU_ARCH, detect vendor from format
    if(GPU_ARCH MATCHES "^sm_")
        set(GPU_VENDOR "NVIDIA")
        set(GPU_BACKEND "cuda")
    elseif(GPU_ARCH MATCHES "^gfx")
        set(GPU_VENDOR "AMD")
        set(GPU_BACKEND "hip")
    elseif(GPU_ARCH STREQUAL "generic" OR GPU_ARCH STREQUAL "pvc" OR GPU_ARCH STREQUAL "intel")
        set(GPU_VENDOR "INTEL")
        set(GPU_BACKEND "level_zero")
        set(GPU_ARCH "generic")
        message(STATUS "✓ Intel GPU specified - using runtime device selection")
    else()
        message(FATAL_ERROR "Unknown GPU architecture format: ${GPU_ARCH}")
    endif()
endif()

# ============================================================================
# C++ Standard (SET BEFORE COMPILER FLAGS!)
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Compiler-Specific SYCL Flags
# ============================================================================
if(SYCL_IMPL STREQUAL "ACPP")
    if(GPU_VENDOR STREQUAL "AMD")
        message(STATUS "Configuring for AdaptiveCpp ROCm backend (${GPU_ARCH})")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --acpp-targets=hip:${GPU_ARCH} -pthread")
        # Prevent HIP from defining vector types (SYCL already has them)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_NO_VECTOR_TYPES__")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --acpp-targets=hip:${GPU_ARCH} -pthread")
    else()
        message(STATUS "Configuring for AdaptiveCpp CUDA backend (${GPU_ARCH})")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --acpp-targets=cuda:${GPU_ARCH} -pthread")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --acpp-targets=cuda:${GPU_ARCH} -pthread")
    endif()
    
    # Force pthread detection
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_PTHREADS_INIT 1)
    
elseif(SYCL_IMPL STREQUAL "INTEL")
    message(STATUS "Configuring for Intel DPC++ CPU/OpenCL backend")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsycl")
    
elseif(SYCL_IMPL STREQUAL "LLVM")
    if(GPU_VENDOR STREQUAL "AMD")
        message(STATUS "Configuring for Custom LLVM SYCL AMD backend (${GPU_ARCH})")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=amdgcn-amd-amdhsa")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${GPU_ARCH}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    else()
        message(STATUS "Configuring for Custom LLVM SYCL CUDA backend (${GPU_ARCH})")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=${GPU_ARCH}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    endif()
    
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsycl -pthread")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L$ENV{HOME}/llvm/build/lib")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,$ENV{HOME}/llvm/build/lib")
    
    # Force pthread
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_PTHREADS_INIT 1)
endif()

# Debugging symbols
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -gline-tables-only")

# Build paths
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

# ============================================================================
# Build Options
# ============================================================================
option(PopSift_BUILD_EXAMPLES "Build PopSift applications" ON)
option(PopSift_BUILD_DOCS "Build PopSift documentation" OFF)
option(PopSift_ERRCHK_AFTER_KERNEL "Synchronize after kernels" OFF)
option(PopSift_USE_POSITION_INDEPENDENT_CODE "PIC" ON)
option(PopSift_USE_GRID_FILTER "Grid filtering" ON)
option(PopSift_USE_TEST_CMD "Testing" OFF)
option(BUILD_SHARED_LIBS "Shared libraries" ON)

if(PopSift_USE_POSITION_INDEPENDENT_CODE AND NOT MSVC)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(GNUInstallDirs)

if(BUILD_SHARED_LIBS)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# ============================================================================
# Find Threads with fallback
# ============================================================================
find_package(Threads QUIET)

if(NOT Threads_FOUND)
    message(WARNING "Using manual pthread configuration")
    set(Threads_FOUND TRUE)
    add_library(Threads::Threads INTERFACE IMPORTED)
    set_property(TARGET Threads::Threads PROPERTY INTERFACE_LINK_LIBRARIES pthread)
endif()

if(NOT PopSift_USE_GRID_FILTER)
  set(DISABLE_GRID_FILTER 1)
else()
  set(DISABLE_GRID_FILTER 0)
endif()

add_subdirectory(src)

if(PopSift_BUILD_DOCS)
  add_subdirectory(doc)
endif()

if(PopSift_USE_TEST_CMD)
  add_subdirectory(testScripts)
endif()

CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)
ADD_CUSTOM_TARGET(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake")

######################################
# SUMMARY
######################################
message("\n******************************************")
message("Building configuration:")
message(STATUS "SYCL Implementation: ${SYCL_IMPL}")
message(STATUS "GPU Vendor: ${GPU_VENDOR}")
message(STATUS "GPU Architecture: ${GPU_ARCH}")
message(STATUS "GPU Backend: ${GPU_BACKEND}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Linker Flags: ${CMAKE_EXE_LINKER_FLAGS}")
message("******************************************\n")
