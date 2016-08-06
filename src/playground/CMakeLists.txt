set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

add_executable( compare-float-dump compare-float-dump.cc )

# cuda_add_executable( try-shfl try-shfl.cu )

# cuda_add_executable( try-prefix-sum try-prefix-sum.cu )

cuda_add_executable( try-sort try-sort.cu )

# cuda_add_executable( try-array try-array.cu )
# nvcc -g -G -o try-array try-array.cu

# cuda_add_executable( try-texture-compare try-texture-compare.cu )
# nvcc -o try-texture-compare try-texture-compare.cu

