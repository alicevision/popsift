FILE( GLOB_RECURSE
      src_files_cu
      ./popsift/*.cu)

FILE( GLOB_RECURSE
      src_files_cpp
      ./popsift/*.cpp)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR})


CUDA_ADD_LIBRARY(popsift STATIC
                  ${src_files_cu}
		  ${src_files_cpp})

target_include_directories(popsift 
            PRIVATE ${Boost_INCLUDE_DIRS}
            PUBLIC ${CUDA_INCLUDE_DIRS})

target_compile_definitions(popsift PRIVATE ${Boost_DEFINITIONS})

set_property(TARGET popsift PROPERTY VERSION ${PROJECT_VERSION})

# cannot use PRIVATE here as there is a bug in FindCUDA 
# https://gitlab.kitware.com/cmake/cmake/issues/16097
target_link_libraries(popsift ${Boost_LIBRARIES} ${CUDA_CUDADEVRT_LIBRARY})
add_subdirectory(application)

install(TARGETS popsift DESTINATION lib EXPORT popsift-targets)

INSTALL(
  DIRECTORY popsift
  DESTINATION include
  COMPONENT headers
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)
