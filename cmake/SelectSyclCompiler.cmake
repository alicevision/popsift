# ============================================================================
# SelectSyclCompiler.cmake
#
# Included from the top-level CMakeLists.txt BEFORE project(), because the
# C++ compiler must be known before languages are enabled.
#
# If the user did not choose a compiler (no -DCMAKE_CXX_COMPILER, no CXX in
# the environment), look for a SYCL compiler driver in the usual install
# locations and use it as CMAKE_CXX_COMPILER. Nothing here is REQUIRED: if no
# driver is found, CMake picks its default compiler and the SYCL_IMPL
# detection after project() decides whether that is usable.
#
#   SYCL_IMPL=ACPP   look for AdaptiveCpp's acpp only
#   SYCL_IMPL=INTEL  look for Intel oneAPI DPC++ (icpx, or icx on Windows) only
#   SYCL_IMPL=AUTO   look for acpp first, then the oneAPI compiler
#   SYCL_IMPL=LLVM   no search, give -DCMAKE_CXX_COMPILER yourself
#
# Extra install prefixes can be given with the CMake variables or environment
# variables AdaptiveCpp_ROOT / ACPP_ROOT and ONEAPI_ROOT / CMPLR_ROOT.
# AdaptiveCpp_DIR (<prefix>/lib/cmake/AdaptiveCpp) is also understood.
# ============================================================================

set(SYCL_IMPL "AUTO" CACHE STRING "SYCL implementation: AUTO, INTEL, LLVM, or ACPP")
set_property(CACHE SYCL_IMPL PROPERTY STRINGS AUTO ACPP INTEL LLVM)

if(DEFINED CMAKE_CXX_COMPILER OR DEFINED ENV{CXX})
  # The user decided; respect it.
  return()
endif()

# ----------------------------------------------------------------------------
# AdaptiveCpp
# ----------------------------------------------------------------------------
if(SYCL_IMPL STREQUAL "AUTO" OR SYCL_IMPL STREQUAL "ACPP")
  set(_acpp_hints)
  foreach(_root ${AdaptiveCpp_ROOT} $ENV{AdaptiveCpp_ROOT} ${ACPP_ROOT} $ENV{ACPP_ROOT})
    list(APPEND _acpp_hints "${_root}/bin")
  endforeach()
  if(AdaptiveCpp_DIR)
    # <prefix>/lib/cmake/AdaptiveCpp -> <prefix>/bin
    get_filename_component(_acpp_prefix "${AdaptiveCpp_DIR}/../../.." ABSOLUTE)
    list(APPEND _acpp_hints "${_acpp_prefix}/bin")
  endif()

  # Fallback locations, searched after PATH:
  #   Homebrew on Apple silicon and on Intel macs / Linuxbrew,
  #   AdaptiveCpp's default CMAKE_INSTALL_PREFIX, common manual prefixes.
  set(_acpp_paths
      /opt/homebrew/bin
      /usr/local/bin
      /home/linuxbrew/.linuxbrew/bin
      /opt/AdaptiveCpp/bin
      /opt/adaptivecpp/bin
      /opt/acpp/bin
      "$ENV{HOME}/.local/bin")
  if(WIN32)
    list(APPEND _acpp_paths
         "$ENV{ProgramFiles}/AdaptiveCpp/bin"
         "$ENV{LOCALAPPDATA}/AdaptiveCpp/bin")
  endif()

  find_program(ACPP_EXECUTABLE
               NAMES acpp
               HINTS ${_acpp_hints}
               PATHS ${_acpp_paths}
               DOC "AdaptiveCpp compiler driver")

  if(ACPP_EXECUTABLE)
    message(STATUS "Found AdaptiveCpp: ${ACPP_EXECUTABLE}")
    set(CMAKE_CXX_COMPILER "${ACPP_EXECUTABLE}" CACHE FILEPATH "C++ compiler")
    return()
  elseif(SYCL_IMPL STREQUAL "ACPP")
    message(STATUS "AdaptiveCpp (acpp) not found; set AdaptiveCpp_ROOT or CMAKE_CXX_COMPILER")
  endif()
endif()

# ----------------------------------------------------------------------------
# Intel oneAPI DPC++
# ----------------------------------------------------------------------------
if(SYCL_IMPL STREQUAL "AUTO" OR SYCL_IMPL STREQUAL "INTEL")
  set(_icpx_hints)
  # CMPLR_ROOT is set by oneAPI's setvars script and points at compiler/<ver>
  foreach(_root ${CMPLR_ROOT} $ENV{CMPLR_ROOT})
    list(APPEND _icpx_hints "${_root}/bin")
  endforeach()
  foreach(_root ${ONEAPI_ROOT} $ENV{ONEAPI_ROOT})
    list(APPEND _icpx_hints "${_root}/compiler/latest/bin")
  endforeach()

  set(_icpx_paths
      /opt/intel/oneapi/compiler/latest/bin
      "$ENV{HOME}/intel/oneapi/compiler/latest/bin")
  if(WIN32)
    # icpx does not exist on Windows; icx is the DPC++ driver there.
    set(_icpx_names icx icpx)
    set(_pf86 "ProgramFiles(x86)")
    list(APPEND _icpx_paths
         "$ENV{${_pf86}}/Intel/oneAPI/compiler/latest/bin"
         "$ENV{ProgramFiles}/Intel/oneAPI/compiler/latest/bin")
  else()
    set(_icpx_names icpx)
  endif()

  find_program(ICPX_EXECUTABLE
               NAMES ${_icpx_names}
               HINTS ${_icpx_hints}
               PATHS ${_icpx_paths}
               DOC "Intel oneAPI DPC++ compiler driver")

  if(ICPX_EXECUTABLE)
    message(STATUS "Found Intel oneAPI DPC++: ${ICPX_EXECUTABLE}")
    set(CMAKE_CXX_COMPILER "${ICPX_EXECUTABLE}" CACHE FILEPATH "C++ compiler")
    return()
  elseif(SYCL_IMPL STREQUAL "INTEL")
    message(STATUS "Intel oneAPI compiler not found; source setvars or set ONEAPI_ROOT")
  endif()
endif()

if(NOT SYCL_IMPL STREQUAL "LLVM")
  message(STATUS "No SYCL compiler driver found, using CMake's default C++ compiler")
endif()
