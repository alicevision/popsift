@echo off
echo Downloading CUDA toolkit 9
appveyor DownloadFile  https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_windows -FileName cuda_9.1.85_windows.exe
dir
echo Installing CUDA toolkit 9
cuda_9.1.85_windows.exe -s nvcc_9.1 ^
                           cublas_9.1 ^
                           cublas_dev_9.1 ^
                           cudart_9.1 ^
                           curand_9.1 ^
                           curand_dev_9.1

echo CUDA toolkit 9 installed

dir "%ProgramFiles%"

set PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v9.1\bin;%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v9.1\libnvvp;%PATH%

nvcc -V