CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT( SIFT )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

set(CMAKE_BUILD_TYPE Debug) # Debug, Release
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DGRIFF_DEBUG")
set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG}   -DGRIFF_DEBUG")
if(APPLE)
  set(CMAKE_CXX_FLAGS       "${CMAKE_CXX_FLAGS}       -Wno-c++11-extensions")
endif(APPLE)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FILENAME__='\"$(subst ${CMAKE_SOURCE_DIR}/,,$(abspath $<))\"'")


include_directories(${CMAKE_SOURCE_DIR}/common)

find_package(CUDA 6.5)
if(CUDA_FOUND)
  include_directories(${CUDA_INCLUDE_DIRS})
else(CUDA_FOUND)
  message( FATAL_ERROR "Failed to find CUDA" )
endif(CUDA_FOUND)


# set(CUDA_NVCC_FLAGS "-G -g -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52 -O3")
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52 -O3")

# set(CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52 -O3 -rdc=true --compiler-options -fPIC")
# set(CUDA_SEPARABLE_COMPILATION ON)
# set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
# set(BUILD_SHARED_LIBS ON)

CUDA_COMPILE( cuda_files_v3
	      src/SIFT.cu
	      src/s_image.cu
	      src/s_pyramid.cu
	      src/keep_time.cu
	      src/s_desc.v1.cu
	      src/s_ori.v1.cu
	      src/s_ori.v2.cu
	      src/s_extrema.v4.cu
	    )

ADD_EXECUTABLE( sift_v3
                src/main.cpp
	      	common/debug_macros.cpp
		common/device_prop.cpp
                common/c_util_img.cpp
                common/c_util_stopwatch.cpp
		${cuda_files_v3} )

TARGET_LINK_LIBRARIES ( sift_v3 ${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY})

