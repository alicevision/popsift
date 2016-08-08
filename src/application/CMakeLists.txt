include_directories(${CMAKE_SOURCE_DIR}/src/common)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})



add_executable( popsift-demo main.cpp
                        c_util_img.cpp c_util_img.h )

target_include_directories( popsift-demo PRIVATE
                                    ${CMAKE_SOURCE_DIR}/src/sift )

target_link_libraries( popsift-demo popsift
                               ${CUDA_LIBRARIES}
			       ${CUDA_CUDADEVRT_LIBRARY}
			       ${Boost_LIBRARIES} )

