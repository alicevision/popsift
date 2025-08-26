@echo off
echo Downloading CUDA toolkit 12 for Windows 10
# appveyor DownloadFile https://developer.download.nvidia.com/compute/cuda/12.5.1/network_installers/cuda_12.5.1_windows_network.exe -Filename cuda_12.5.1_windows.exe

appveyor DownloadFile https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-12.5.82-archive.zip -Filename cuda_nvcc.zip
appveyor DownloadFile https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.5.82-archive.zip -Filename cuda_cudart.zip
appveyor DownloadFile https://developer.download.nvidia.com/compute/cuda/redist/visual_studio_integration/windows-x86_64/visual_studio_integration-windows-x86_64-12.5.82-archive.zip -Filename vs_integration.zip
dir

echo Unzipping CUDA toolkit 12
tar -xf cuda_nvcc.zip
tar -xf cuda_cudart.zip
tar -xf vs_integration.zip
dir

echo Making CUDA install dir(s)
mkdir "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
mkdir "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.5\extras"

echo Copying toolkit files to install dir(s)
xcopy cuda_cudart-windows-x86_64-12.5.82-archive "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.5" /s /e /i /y
xcopy cuda_nvcc-windows-x86_64-12.5.82-archive "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.5" /s /e /i /y
xcopy visual_studio_integration-windows-x86_64-12.5.82-archive "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.5\extras" /s /e /i /y


# echo Installing CUDA toolkit 12
# cuda_12.5.1_windows.exe
# cuda_9.1.85_windows.exe -s nvcc_12.5 cudart_12.5


echo CUDA toolkit 12 installed

dir "%ProgramFiles%"

set PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin;%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.5\libnvvp;%PATH%

dir "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA"
dir "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
dir "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"

nvcc -V
