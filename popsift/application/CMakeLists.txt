include_directories(${CMAKE_SOURCE_DIR}/popsift/common)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})



add_executable( sift_v4 main.cpp )
target_include_directories( sift_v4 PRIVATE
                                    ${CMAKE_SOURCE_DIR}/popsift/src )
target_link_libraries( sift_v4 popsift
                               ${CUDA_LIBRARIES}
			       ${CUDA_CUDADEVRT_LIBRARY}
			       -lboost_filesystem
			       -lboost_system )

