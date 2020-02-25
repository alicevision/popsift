#
# after returning from this function, do not forget to call the following:
#    set(RESULT_NAME ${RESULT_NAME} CACHE STRING "CUDA CC versions to compile")
# replacing your own variable for RESULT_NAME
#
# We assume that MINCUDAVERSION defaults to 7.0
#
function(ChooseCudaCC RESULT_NAME MINCC MINCUDAVERSION)
  if(NOT DEFINED ${MINCC})
    message(FATAL_ERROR "CMake function ChooseCudaCC must be called with a minimal CC")
  endif()
  if(NOT DEFINED ${MINCUDAVERSION})
    set(MINCUDAVERSION 70)
  endif()

  find_package(CUDA ${MINCUDAVERSION} REQUIRED)

  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "Could not find CUDA >= 7.0")
  endif()

  #
  # Create a list of possible CCs for each host processor.
  # This may require tuning: CUDA cards exist in AIX machines with POWER CPUs,
  # it is possible that non-Tegra ARM systems exist as well.
  # For now, this is my best guess.
  #
  if((CMAKE_SYSTEM_PROCESSOR STREQUAL "i686") OR (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"))
    set(CC_LIST_BY_SYSTEM_PROCESSOR 20 21 30 35 50 52 60 61 70 75)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
    set(CC_LIST_BY_SYSTEM_PROCESSOR 32 53 62 72)
  else()
    message(FATAL_ERROR "Unknown how to build for ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  #
  # Default setting of the CUDA CC versions to compile.
  # Shortening the lists saves a lot of compile time.
  #
  set(CUDA_MIN_CC 20)
  set(CUDA_MAX_CC 75)
  if(CUDA_VERSION_MAJOR GREATER_EQUAL 10)
    set(CUDA_MIN_CC 30)
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

  set(CC_LIST "")
  foreach(CC ${CC_LIST_BY_SYSTEM_PROCESSOR})
    if( (${CC} GREATER ${MINCC}) AND
        (${CC} GREATER_EQUAL ${CUDA_MIN_CC}) AND
	(${CC} LESS_EQUAL ${CUDA_MAX_CC}) )
      list(APPEND CC_LIST ${CC})
    endif()
  endforeach()

  #
  # Add all requested CUDA CCs to the command line for offline compilation
  #
  list(SORT CC_LIST)
  foreach(CC_VERSION ${CC_LIST})
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode;arch=compute_${CC_VERSION},code=sm_${CC_VERSION}")
  endforeach()

  #
  # Use the highest request CUDA CC for CUDA JIT compilation
  #
  list(LENGTH CC_LIST CC_LIST_LEN)
  MATH(EXPR CC_LIST_LEN "${CC_LIST_LEN}-1")
  list(GET CC_LIST ${CC_LIST_LEN} CC_LIST_LAST)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode;arch=compute_${CC_LIST_LAST},code=compute_${CC_LIST_LAST}")

  set(${RESULT_NAME} ${CC_LIST} PARENT_SCOPE)
endfunction()

