#
# This file contains two functions:
# chooseCudaCC
# getFlagsForCudaCCList
#
# Motivation:
# CUDA hardware and SDKs are developing over time, different SDK support different
# hardware, and supported hardware differs depending on platform even for the same
# SDK version. This file attempts to provide a function that returns a valid selection
# of hardware for the current SDK and platform. It will require updates as CUDA develops,
# and it is currently not complete in terms of existing platforms that support CUDA.
#

#
# Return the minimal set of supported Cuda CC 
#
# Usage:
#   chooseCudaCC(SUPPORTED_CC SUPPORTED_GENCODE_FLAGS
#                [MIN_CUDA_VERSION X.Y]
#                [MIN_CC XX ])
#
# SUPPORTED_CC out variable. Stores the list of supported CC.
# SUPPORTED_GENCODE_FLAGS out variable. List of gencode flags to append to, e.g., CUDA_NVCC_FLAGS
# MIN_CUDA_VERSION the minimal supported version of cuda (e.g. 7.5, default 7.0).
# MIN_CC minimal supported Cuda CC by the project (e.g. 35, default 20)
#
# This function does not edit cache entries or variables in the parent scope
# except for the variables whose names are supplied for SUPPORTED_CC and
# SUPPORTED_GENCODE_FLAGS
#
# You may want to cache SUPPORTED_CC and append SUPPORTED_GENCODE_FLAGS to
# CUDA_NVCC_FLAGS.
# Like this:
#    set(MYCC ${MYCC} CACHE STRING "CUDA CC versions to compile")
# end
#    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${MY_GENCODE_FLAGS}")
#    
function(chooseCudaCC SUPPORTED_CC SUPPORTED_GENCODE_FLAGS)
  set(options "")
  set(oneValueArgs MIN_CUDA_VERSION MIN_CC)
  set(multipleValueArgs "")
  cmake_parse_arguments(CHOOSE_CUDA "${options}" "${oneValueArgs}" "${multipleValueArgs}" ${ARGN})

  if(NOT DEFINED CHOOSE_CUDA_MIN_CC)
    set(CHOOSE_CUDA_MIN_CC 20)
  endif()
  if(NOT DEFINED CHOOSE_CUDA_MIN_CUDA_VERSION)
    set(CHOOSE_CUDA_MIN_CUDA_VERSION 7.0)
  endif()

  find_package(CUDA ${CHOOSE_CUDA_MIN_CUDA_VERSION} REQUIRED)

  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "Could not find CUDA >= ${CHOOSE_CUDA_MIN_CUDA_VERSION}")
  endif()

  #
  # Create a list of possible CCs for each host processor.
  # This may require tuning: CUDA cards exist in AIX machines with POWER CPUs,
  # it is possible that non-Tegra ARM systems exist as well.
  # For now, this is my best guess.
  #
  set(TEGRA_SUPPORTED_PROCESSORS "armv71;arm;aarch64")
  set(OTHER_SUPPORTED_PROCESSORS "i686;x86_64;AMD64")

  set(CC_LIST_BY_SYSTEM_PROCESSOR "")
  if(CMAKE_SYSTEM_PROCESSOR IN_LIST OTHER_SUPPORTED_PROCESSORS)
    list(APPEND CC_LIST_BY_SYSTEM_PROCESSOR "20;21;30;35;50;52;60;61;70;75;80;86")
  endif()
  if(CMAKE_SYSTEM_PROCESSOR IN_LIST TEGRA_SUPPORTED_PROCESSORS)
    list(APPEND CC_LIST_BY_SYSTEM_PROCESSOR "32;53;62;72")
  endif()
  if(NOT CC_LIST_BY_SYSTEM_PROCESSOR)
    message(FATAL_ERROR "Unknown how to build for ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  #
  # Default setting of the CUDA CC versions to compile.
  # Shortening the lists saves a lot of compile time.
  #
  set(CUDA_MIN_CC 20)
  set(CUDA_MAX_CC 86)
  if(CUDA_VERSION VERSION_GREATER_EQUAL 11.1)
    set(CUDA_MIN_CC 35)
  elseif(CUDA_VERSION_MAJOR GREATER_EQUAL 11)
    set(CUDA_MIN_CC 35)
    set(CUDA_MAX_CC 80)
  elseif(CUDA_VERSION_MAJOR GREATER_EQUAL 10)
    set(CUDA_MIN_CC 30)
    set(CUDA_MAX_CC 75)
  elseif(CUDA_VERSION_MAJOR GREATER_EQUAL 9)
    set(CUDA_MIN_CC 30)
    set(CUDA_MAX_CC 72)
  elseif(CUDA_VERSION_MAJOR GREATER_EQUAL 8)
    set(CUDA_MAX_CC 62)
  elseif(CUDA_VERSION_MAJOR GREATER_EQUAL 7)
    set(CUDA_MAX_CC 53)
  else()
    message(FATAL_ERROR "We do not support a CUDA SDK below version 7.0")
  endif()
  if(${CHOOSE_CUDA_MIN_CC} GREATER ${CUDA_MIN_CC})
    set(CUDA_MIN_CC ${CHOOSE_CUDA_MIN_CC})
  endif()

  set(CC_LIST "")
  foreach(CC ${CC_LIST_BY_SYSTEM_PROCESSOR})
    if( (${CC} GREATER_EQUAL ${CUDA_MIN_CC}) AND
        (${CC} LESS_EQUAL ${CUDA_MAX_CC}) )
      list(APPEND CC_LIST ${CC})
    endif()
  endforeach()

  #
  # Add all requested CUDA CCs to the command line for offline compilation
  #
  set(GENCODE_FLAGS "")
  list(SORT CC_LIST)
  foreach(CC_VERSION ${CC_LIST})
    list(APPEND GENCODE_FLAGS "-gencode;arch=compute_${CC_VERSION},code=sm_${CC_VERSION}")
  endforeach()

  #
  # Use the highest request CUDA CC for CUDA JIT compilation
  #
  list(LENGTH CC_LIST CC_LIST_LEN)
  MATH(EXPR CC_LIST_LEN "${CC_LIST_LEN}-1")
  list(GET CC_LIST ${CC_LIST_LEN} CC_LIST_LAST)
  list(APPEND GENCODE_FLAGS "-gencode;arch=compute_${CC_LIST_LAST},code=compute_${CC_LIST_LAST}")

  #
  # Two variables are exported to the parent scope. One is passed through the
  # environment (CUDA_NVCC_FLAGS), the other is passed by name (SUPPORTED_CC)
  #
  set(${SUPPORTED_GENCODE_FLAGS} "${GENCODE_FLAGS}" PARENT_SCOPE)
  set(${SUPPORTED_CC} "${CC_LIST}" PARENT_SCOPE)
endfunction()

#
# Return the gencode parameters for a given list of CCs.
#
# Usage:
#   getFlagsForCudaCCList(INPUT_CC_LIST SUPPORTED_GENCODE_FLAGS)
#
# INPUT_CC_LIST in variable. Contains a list of supported CCs.
# SUPPORTED_GENCODE_FLAGS out variable. List of gencode flags to append to, e.g., CUDA_NVCC_FLAGS
#
function(getFlagsForCudaCCList INPUT_CC_LIST SUPPORTED_GENCODE_FLAGS)
  set(CC_LIST "${${INPUT_CC_LIST}}")

  #
  # Add all requested CUDA CCs to the command line for offline compilation
  #
  set(GENCODE_FLAGS "")
  list(SORT CC_LIST)
  foreach(CC_VERSION ${CC_LIST})
    list(APPEND GENCODE_FLAGS "-gencode;arch=compute_${CC_VERSION},code=sm_${CC_VERSION}")
  endforeach()

  #
  # Use the highest request CUDA CC for CUDA JIT compilation
  #
  list(LENGTH CC_LIST CC_LIST_LEN)
  MATH(EXPR CC_LIST_LEN "${CC_LIST_LEN}-1")
  list(GET CC_LIST ${CC_LIST_LEN} CC_LIST_LAST)
  list(APPEND GENCODE_FLAGS "-gencode;arch=compute_${CC_LIST_LAST},code=compute_${CC_LIST_LAST}")

  message(STATUS "Setting gencode flags: ${GENCODE_FLAGS}")

  #
  # Two variables are exported to the parent scope. One is passed through the
  # environment (CUDA_NVCC_FLAGS), the other is passed by name (SUPPORTED_CC)
  #
  set(${SUPPORTED_GENCODE_FLAGS} "${GENCODE_FLAGS}" PARENT_SCOPE)
endfunction()

