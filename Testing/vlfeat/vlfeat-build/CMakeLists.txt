# Builds only VLFeat's 'sift' command line tool.
#
# This file is copied into the root of a VLFeat source tree by
# Testing/vlfeat/CMakeLists.txt. It replaces VLFeat's own Makefile, which
# needs GNU make and a per-platform ARCH table, and works with gcc, clang
# and MSVC alike.
#
# SSE2, AVX and OpenMP are disabled so that results do not depend on the
# CPU's instruction set.

cmake_minimum_required(VERSION 3.24)
project(vlfeat_sift_cli LANGUAGES C)

if(NOT VLFEAT_OUTPUT_DIR)
  set(VLFEAT_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin")
endif()

file(GLOB _vl_sources "${CMAKE_CURRENT_SOURCE_DIR}/vl/*.c")

# The library sources are compiled straight into the executable. VL_BUILD_DLL
# selects the export (not import) declarations in vl/host.h, which MSVC needs
# when the code is not linked from a separate DLL.
add_executable(sift src/sift.c ${_vl_sources})

target_include_directories(sift PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(sift PRIVATE
                           VL_BUILD_DLL
                           VL_DISABLE_SSE2
                           VL_DISABLE_AVX
                           VL_DISABLE_OPENMP)

if(UNIX AND NOT APPLE)
  target_compile_definitions(sift PRIVATE _GNU_SOURCE)
endif()
if(MSVC)
  target_compile_definitions(sift PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

find_package(Threads REQUIRED)
target_link_libraries(sift PRIVATE Threads::Threads)
if(NOT WIN32)
  target_link_libraries(sift PRIVATE m)
endif()

# $<1:...> keeps multi-config generators from adding a per-config subdirectory,
# so the executable is always at VLFEAT_OUTPUT_DIR/sift.
set_target_properties(sift PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY "$<1:${VLFEAT_OUTPUT_DIR}>")
