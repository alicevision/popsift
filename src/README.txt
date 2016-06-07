** SIFT, OpenCL Implementation **

** Read LICENSE.txt first
** Read the documentation in doc/
** Data (images) are provided in the common/ folder

** Prerequisites
CMake - Cross Platform Make
- http://www.cmake.org/

OpenCL SDK
 - Intel  : http://software.intel.com/en-us/articles/vcsource-tools-opencl-sdk/
 - AMD    : http://developer.amd.com/zones/openclzone/Pages/default.aspx
 - NVIDIA : http://developer.nvidia.com/opencl

** Compilation (all projects) in build/
$ cd /path/to/bemap
$ cd build/
$ cmake ../
$ make

** Compilation (only this project) in TestRun
$ cd /path/to/SIFT/cl/
$ cd TestRun
$ cmake ../
$ make

** Usage:
./SIFT_ocl -h
./SIFT_ocl [--verbose|-v] [--help|-h]
     [--octaves|-O INT] [--levels|-S INT] [--upsampling|-u INT]
     [--threshold|-t FLOAT] [--edge-threshold|-e FLOAT] [--floating-point]
     [--save-gauss] [--save-dog] [--save-mag] [--save-ori] [--fast-comp]
     [--use-gpu|-g] [--choose-dev|-d] [--choose-plat|-p DEV]
     [--dev-info] [--prep-time]
     FILENAME

* Options *
 --verbose                  Be verbose
 --help                     Print this message
 --octaves=INT              Number of octaves
 --levels=INT               Number of levels per octave
 --upsampling=INT           Number of upsamplings
 --threshold=FLOAT          Keypoint strength threhsold
 --edge-threshold=FLOAT     On-edge threshold
 --save-gauss               Save Gaussian Scale pyramid
 --save-dog                 Save Difference of Gaussian pyramid
 --save-mag                 Save Magnitudes pyramid
 --save-ori                 Save Orientations pyramid
 --floating-point           Save descriptor values in floating point
 --fast-comp                Faster descriptor computation (may decrease matching quality)
 --use-gpu                  Use GPU as the CL device
 --choose-dev               Choose which OpenCL device to use
 --choose-plat              Choose which OpenCL platform to use (0, 1, 2)
                                      [0] Advanced Micro Devices, Inc.
                                      [1] NVIDIA Corporation
                                      [2] Intel(R) Corporation
                                      default: Any CPU device
 --dev-info                 Show Device Info
 --prep-time                Show initialization, memory preparation, step execution, and copyback time

 The keypoints will be written to [filename].key

 * Examples *
./SIFT_ocl [OPTS...] -v -u 2 --save-gauss --use-gpu test_data.pgm
./SIFT_ocl [OPTS...] -O 5 -S 2 --outkey=test_output.key --dev-info test_data.ppm