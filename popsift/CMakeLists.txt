FILE( GLOB_RECURSE
      src_files_cu
      ./src/*.cu
      ./common/*.cu )

FILE( GLOB_RECURSE
      src_files_cpp
      ./src/*.cpp
      ./common/*.cpp )

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR})

include_directories(${CMAKE_SOURCE_DIR}/popsift/common)

CUDA_ADD_LIBRARY( popsift STATIC
                  ${src_files_cu}
		  ${src_files_cpp} )

add_subdirectory(application)

