all: try-array

try-array: try-array.cu
	nvcc -g -G -o try-array try-array.cu

try-texture-compare: try-texture-compare.cu
	nvcc -o try-texture-compare try-texture-compare.cu

clean:
	rm -f try-array
	rm -f try-texture-compare

