cmake_minimum_required(VERSION 3.24)

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
# C++ Standard (SET BEFORE COMPILER FLAGS!)
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Compiler-Specific SYCL Flags
# ============================================================================
if(SYCL_IMPL STREQUAL "ACPP")
    message(STATUS "Configuring for AdaptiveCpp CUDA backend (sm_89)")
    # ✅ Don't add -std=c++17 manually for AdaptiveCpp!
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --acpp-targets=cuda:sm_89 -pthread")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --acpp-targets=cuda:sm_89 -pthread")
    
    # Force pthread detection
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_PTHREADS_INIT 1)
    
elseif(SYCL_IMPL STREQUAL "INTEL")
    message(STATUS "Configuring for Intel DPC++ CUDA backend (sm_89)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_89")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsycl")
    
elseif(SYCL_IMPL STREQUAL "LLVM")
    message(STATUS "Configuring for Custom LLVM SYCL CUDA backend (sm_89)")
    
    # ✅ Add UR loader path for custom LLVM
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_89")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    
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
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Linker Flags: ${CMAKE_EXE_LINKER_FLAGS}")
message("******************************************\n")