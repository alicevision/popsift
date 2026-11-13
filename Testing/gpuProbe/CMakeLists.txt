# popsift-gpu-probe: lists the GPUs and exits with 0 if one is usable, and
# with 77 if not. popsift_compare.py runs it before PopSift runs and skips
# them without a GPU. It only needs the runtime of the backend that PopSift
# is built with, CUDA or HIP, and also shows the effect of
# CUDA_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES.

add_executable(popsift-gpu-probe gpu_probe.cpp)
target_include_directories(popsift-gpu-probe PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
if(USE_HIP)
  # USE_HIP selects the HIP branch of cuda_to_hip.h.
  target_compile_definitions(popsift-gpu-probe PRIVATE USE_HIP)
  target_link_libraries(popsift-gpu-probe PRIVATE hip::host)
else()
  target_link_libraries(popsift-gpu-probe PRIVATE CUDA::cudart)
endif()
