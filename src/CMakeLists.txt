FILE( GLOB_RECURSE
      src_files_cu
      ./popsift/*.cu)

FILE( GLOB_RECURSE
      src_files_cpp
      ./popsift/*.cpp)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR})

include_directories(${CMAKE_SOURCE_DIR}/src/common)

CUDA_ADD_LIBRARY( popsift STATIC
                  ${src_files_cu}
		  ${src_files_cpp} )

add_subdirectory(application)

install(TARGETS popsift DESTINATION lib EXPORT popsift-targets)

INSTALL(
  DIRECTORY popsift
  DESTINATION include
  COMPONENT headers
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)
