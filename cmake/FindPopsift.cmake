# Locate the Popsift libraries.
#
# Defines the following variables:
#
#   POPSIFT_FOUND        - TRUE if the popsift headers and libs are found
#   POPSIFT_INCLUDE_DIRS - The path to popsift headers
#
#   POPSIFT_LIBRARIES    - Libraries to link against to use popsift.
#   POPSIFT_LIBRARY_DIR  - The base directory to search for popsift.
#
# Accepts the following variables as input:
#
#   POPSIFT_DIR - (as a CMake or environment variable)
#                The root directory of the popsift install prefix

MESSAGE(STATUS "Looking for popsift.")

FIND_PATH(POPSIFT_INCLUDE_DIR popsift/popsift.h
  HINTS
    $ENV{POPSIFT_DIR}/include
    ${POPSIFT_DIR}/include
  PATH_SUFFIXES
    popsift
)

find_package(CUDA 7.0 REQUIRED)
find_package(Threads REQUIRED)

IF(POPSIFT_INCLUDE_DIR)
  MESSAGE(STATUS "popsift headers found in ${POPSIFT_INCLUDE_DIR}")
ELSE()
  MESSAGE(STATUS "POPSIFT_INCLUDE_DIR NOT FOUND")
ENDIF (POPSIFT_INCLUDE_DIR)

FIND_LIBRARY(POPSIFT_LIBRARY NAMES popsift
  HINTS
    $ENV{POPSIFT_DIR}
    ${POPSIFT_DIR}
  PATH_SUFFIXES
    lib
    lib/popsift
)
GET_FILENAME_COMPONENT(POPSIFT_LIBRARY_DIR "${POPSIFT_LIBRARY}" PATH)

SET(POPSIFT_LIBRARIES ${POPSIFT_LIBRARY})
SET(POPSIFT_INCLUDE_DIRS ${POPSIFT_INCLUDE_DIR})

IF(POPSIFT_LIBRARY)
  MESSAGE(STATUS "popsift libraries found: ${POPSIFT_LIBRARY}")
  MESSAGE(STATUS "popsift libraries directories: ${POPSIFT_LIBRARY_DIR}")
ENDIF (POPSIFT_LIBRARY)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set POPSIFT_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(popsift  DEFAULT_MSG
                                  POPSIFT_LIBRARY POPSIFT_INCLUDE_DIR)

MARK_AS_ADVANCED(POPSIFT_INCLUDE_DIR POPSIFT_LIBRARY)

