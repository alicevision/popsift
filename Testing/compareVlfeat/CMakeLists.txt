# Prep for CTest comparison between PopSift and VLFeat
#
# Download and patch last 2018 version of vlfeat.

include(ExternalProject)

set(PopSift_VLFEAT_GIT_REPOSITORY "https://github.com/vlfeat/vlfeat.git"
    CACHE STRING "Git URL to fetch VLFeat from")
set(PopSift_VLFEAT_GIT_TAG "1b9075fc42fe54b42f0e937f8b9a230d8e2c7701"
    CACHE STRING "VLFeat commit to build against. Verify this works before using newer vlfeat commits.")
set(PopSift_VLFEAT_SOURCE_DIR ""
    CACHE PATH "Optional: point at an existing VLFeat checkout instead of cloning one.")

set(PopSift_VLFEAT_MAX_AVG_DESC_DIST "-1.0"
    CACHE STRING "ctest needs pass/fail indicators. Set a positive number of compare average descriptor distance. -1 disables and reports only.")

find_program(CONVERT convert)
if(CONVERT MATCHES ".+-NOTFOUND")
  message(WARNING "PopSift_COMPARE_VLFEAT: ImageMagick's 'convert' was not "
                   "found. The comparison test will fail unless "
                   "PopSift_VLFEAT_TEST_IMAGE already has a .pgm available.")
endif()

# Figure out the VLFeat ARCH name for this platform. Our patch has extended this list
# for Tegra and M3 Mac.
set(_vlfeat_arch "")
if(APPLE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(_vlfeat_arch "arm64")
  else()
    set(_vlfeat_arch "maci64")
  endif()
elseif(UNIX)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_vlfeat_arch "glnxarm64")
  else()
    set(_vlfeat_arch "glnxa64")
  endif()
endif()

if(_vlfeat_arch STREQUAL "")
  message(WARNING "PopSift_COMPARE_VLFEAT: no known VLFeat ARCH mapping for "
                   "this platform (only Linux x86_64 and macOS are handled). "
                   "The vlfeat_external build step will likely need manual "
                   "adjustment in compareVlfeat/CMakeLists.txt.")
endif()

# Patch VLFeat download to fix a bad comparison that breaks vlfeat CLI
# Patch Makefiles to compile CLI-only on ARM platforms (we don't need the rest for our tests)
set(_vlfeat_patch_cmd
    patch -p1 -N -i "${CMAKE_CURRENT_SOURCE_DIR}/patches/vlfeat-pgm-fix.patch")

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
  list(APPEND _vlfeat_patch_cmd
       COMMAND patch -p1 -N -i "${CMAKE_CURRENT_SOURCE_DIR}/patches/vlfeat-macos-arm64.patch")
elseif(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  list(APPEND _vlfeat_patch_cmd
       COMMAND patch -p1 -N -i "${CMAKE_CURRENT_SOURCE_DIR}/patches/vlfeat-linux-arm64.patch")
endif()

# Use local version of download from Github
set(_vlfeat_download_args)
if(PopSift_VLFEAT_SOURCE_DIR)
  list(APPEND _vlfeat_download_args
       SOURCE_DIR "${PopSift_VLFEAT_SOURCE_DIR}"
       DOWNLOAD_COMMAND "")
else()
  list(APPEND _vlfeat_download_args
       GIT_REPOSITORY "${PopSift_VLFEAT_GIT_REPOSITORY}"
       GIT_TAG        "${PopSift_VLFEAT_GIT_TAG}"
       GIT_SHALLOW    TRUE)
endif()

# Disable OpenMP on all platforms.
# Disable SSE2 and AVX on ARM platforms
# (TBD: disable SSE2 and AVX in the Makefile using the patches)
set(_vlfeat_make_args ARCH=${_vlfeat_arch} DISABLE_OPENMP=yes)
if(_vlfeat_arch STREQUAL "arm64" OR _vlfeat_arch STREQUAL "glnxarm64")
  list(APPEND _vlfeat_make_args DISABLE_SSE2=yes DISABLE_AVX=yes)
endif()

if(_vlfeat_arch STREQUAL "arm64")
  # VER is derived on Linux. Fails for some reason on M3 Mac.
  list(APPEND _vlfeat_make_args VER=0.9.21)
endif()

# Don't download at configure time. Download and make only when ctest needs it.
ExternalProject_Add(vlfeat_external
  EXCLUDE_FROM_ALL     TRUE
  ${_vlfeat_download_args}
  UPDATE_DISCONNECTED  TRUE
  PATCH_COMMAND        ${_vlfeat_patch_cmd}
  CONFIGURE_COMMAND    ""
  BUILD_IN_SOURCE      TRUE
  BUILD_COMMAND        make ${_vlfeat_make_args}
  INSTALL_COMMAND      ""
  LOG_PATCH            TRUE
  LOG_BUILD            TRUE
)

ExternalProject_Get_property(vlfeat_external SOURCE_DIR)
set(PopSift_VLFEAT_SIFT_BIN "${SOURCE_DIR}/bin/${_vlfeat_arch}/sift")

# --- The comparison test itself ------------------------------------------

configure_file(run_comparison.sh.in
               "${CMAKE_CURRENT_BINARY_DIR}/run_comparison.sh"
               @ONLY)

# Magic created by Claude. I don't understand what FIXTURES_SETUP does.
# Fixture: (re-)builds vlfeat_external on demand. EXCLUDE_FROM_ALL keeps this
# out of the normal build, so this is the only thing that triggers it.
add_test(NAME vlfeat_build
         COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target vlfeat_external)
set_tests_properties(vlfeat_build PROPERTIES FIXTURES_SETUP VLFEAT)

# Compile popsift if needed
add_test(NAME popsift_bins_build
         COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                 --target popsift-demo --target popsift-compareSiftFiles)
set_tests_properties(popsift_bins_build PROPERTIES FIXTURES_SETUP POPSIFT_BINS)

add_test(NAME vlfeat-comparison
         COMMAND /bin/bash "${CMAKE_CURRENT_BINARY_DIR}/run_comparison.sh")
set_tests_properties(vlfeat-comparison PROPERTIES FIXTURES_REQUIRED "VLFEAT;POPSIFT_BINS")

