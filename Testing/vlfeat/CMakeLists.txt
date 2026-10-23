# ==============================================================================
# Local, corrected VLFeat 'sift' command line tool
# ==============================================================================
#
# Fetches VLFeat, applies the patches in patches/ with 'git apply' and builds
# only the 'sift' CLI with CMake (sources: vl/*.c and src/sift.c). The result
# is the reference implementation that PopSift is compared against.
#
# Defines:
#   vlfeat_sift              target that downloads, patches and builds VLFeat.
#                            EXCLUDE_FROM_ALL: it only runs when asked for,
#                            either directly or through the vlfeat_build test.
#   PopSift_VLFEAT_SIFT_BIN  path of the built 'sift' executable.
#   vlfeat_build             CTest fixture setup (fixture name VLFEAT).
#   vlfeat_sift_smoke        CTest that runs 'sift' on a hand-crafted image.
#
# Included by Testing/CMakeLists.txt when BUILD_TESTING and
# PopSift_TEST_WITH_VLFEAT are both ON. It can also be configured on its own,
# without CUDA or HIP:
#
#   cmake -S Testing/vlfeat -B build-vlfeat
#   cmake --build build-vlfeat --target vlfeat_sift
#   ctest --test-dir build-vlfeat --output-on-failure

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # Standalone configuration
  cmake_minimum_required(VERSION 3.24)
  project(PopSiftVLFeat LANGUAGES NONE)
  include(CTest)
endif()

include(ExternalProject)
find_package(Git REQUIRED)

set(PopSift_VLFEAT_GIT_REPOSITORY "https://github.com/vlfeat/vlfeat.git"
    CACHE STRING "Git URL to fetch VLFeat from")
set(PopSift_VLFEAT_GIT_TAG "1b9075fc42fe54b42f0e937f8b9a230d8e2c7701"
    CACHE STRING "VLFeat commit to build against. Verify this works before using newer vlfeat commits.")
set(PopSift_VLFEAT_SOURCE_OOT_DIR ""
    CACHE PATH "Provide an out-of-tree source dir. Avoids downloading. An existing VLFeat checkout, copied into the build tree and patched there; the original is not modified.")
mark_as_advanced(PopSift_VLFEAT_GIT_REPOSITORY PopSift_VLFEAT_GIT_TAG)

if(PopSift_VLFEAT_SOURCE_OOT_DIR)
  if(NOT IS_DIRECTORY "${PopSift_VLFEAT_SOURCE_OOT_DIR}/vl")
    message(FATAL_ERROR
      "PopSift_VLFEAT_SOURCE_OOT_DIR ('${PopSift_VLFEAT_SOURCE_OOT_DIR}') does not look "
      "like a VLFeat checkout: it has no vl/ subdirectory.")
  endif()
  # A local directory given as URL is copied into SOURCE_DIR.
  set(_vlfeat_download_args URL "${PopSift_VLFEAT_SOURCE_OOT_DIR}")
else()
  # GIT_SHALLOW works here because the pinned commit is the tip of VLFeat's
  # master branch. Drop it if PopSift_VLFEAT_GIT_TAG is set to an older commit.
  set(_vlfeat_download_args
      GIT_REPOSITORY "${PopSift_VLFEAT_GIT_REPOSITORY}"
      GIT_TAG        "${PopSift_VLFEAT_GIT_TAG}"
      GIT_SHALLOW    TRUE)
endif()

# Fix for a bad comparison in vl/pgm.c that makes the CLI reject every PGM file.
set(_vlfeat_patches "${CMAKE_CURRENT_SOURCE_DIR}/patches/vlfeat-pgm-fix.patch")
string(REPLACE ";" "|" _vlfeat_patches_arg "${_vlfeat_patches}")

set(_vlfeat_bin_dir "${CMAKE_CURRENT_BINARY_DIR}/vlfeat/bin")
if(CMAKE_HOST_WIN32)
  set(_vlfeat_exe_suffix ".exe")
else()
  set(_vlfeat_exe_suffix "")
endif()

# The CMakeLists.txt in vlfeat-build/ is copied into the VLFeat source tree,
# which always is a private copy (a fresh clone or a copy of
# PopSift_VLFEAT_SOURCE_OOT_DIR) under this build directory.
ExternalProject_Add(vlfeat_sift
  EXCLUDE_FROM_ALL     TRUE
  PREFIX               "${CMAKE_CURRENT_BINARY_DIR}/vlfeat"
  ${_vlfeat_download_args}
  UPDATE_DISCONNECTED  TRUE
  LIST_SEPARATOR       "|"
  PATCH_COMMAND        "${CMAKE_COMMAND}"
                       "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}"
                       "-DSOURCE_DIR=<SOURCE_DIR>"
                       "-DPATCHES=${_vlfeat_patches_arg}"
                       -P "${CMAKE_CURRENT_SOURCE_DIR}/apply_patches.cmake"
               COMMAND "${CMAKE_COMMAND}" -E copy
                       "${CMAKE_CURRENT_SOURCE_DIR}/vlfeat-build/CMakeLists.txt"
                       "<SOURCE_DIR>/CMakeLists.txt"
  CMAKE_ARGS           -DCMAKE_BUILD_TYPE=Release
                       "-DVLFEAT_OUTPUT_DIR=${_vlfeat_bin_dir}"
  BUILD_COMMAND        "${CMAKE_COMMAND}" --build <BINARY_DIR> --config Release
  INSTALL_COMMAND      ""
  LOG_PATCH            TRUE
  LOG_CONFIGURE        TRUE
  LOG_BUILD            TRUE
  LOG_OUTPUT_ON_FAILURE TRUE
)

set(PopSift_VLFEAT_SIFT_BIN "${_vlfeat_bin_dir}/sift${_vlfeat_exe_suffix}"
    CACHE INTERNAL "Path of the VLFeat sift CLI built by vlfeat_sift")

if(BUILD_TESTING)
  # Fixture: (re-)builds vlfeat_sift when a test that requires VLFEAT runs.
  # FIXTURES_SETUP makes CTest run this first, also under 'ctest -R <name>',
  # and reports dependent tests as not run if it fails.
  add_test(NAME vlfeat_build
           COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}"
                   --target vlfeat_sift --config "$<CONFIG>")
  set_tests_properties(vlfeat_build PROPERTIES
                       FIXTURES_SETUP VLFEAT
                       LABELS "cpu;vlfeat")

  # Without the pgm fix, 'sift' rejects this file, so this also checks
  # that the patch was applied.
  add_test(NAME vlfeat_sift_smoke
           COMMAND "${PopSift_VLFEAT_SIFT_BIN}" --verbose
                   "--output=${CMAKE_CURRENT_BINARY_DIR}/box-12x12.sift"
                   "${CMAKE_CURRENT_SOURCE_DIR}/../testImages/box-12x12.pgm")
  set_tests_properties(vlfeat_sift_smoke PROPERTIES
                       FIXTURES_REQUIRED VLFEAT
                       LABELS "cpu;vlfeat")
endif()
