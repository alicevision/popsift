# Windows build functions for PopSift
# Usage: source this file and call the individual functions

function Setup-Directories {
    param([string]$BuildType)
    
    Write-Host "Setting up build directories for $BuildType..."
    $buildDir = "build_$($BuildType.ToLower())"
    $thirdPartyDir = "build_as_3rdparty_$($BuildType.ToLower())"
    
    New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
    New-Item -ItemType Directory -Path $thirdPartyDir -Force | Out-Null
}

function Configure-CMake {
    param(
        [string]$BuildType,
        [string]$VcpkgRoot,
        [string]$WorkspaceDir
    )
    
    Write-Host "Configuring CMake for $BuildType..."
    $buildDir = "build_$($BuildType.ToLower())"
    $installDir = "$WorkspaceDir/install_$($BuildType.ToLower())"
    $vcpkgToolchain = "$VcpkgRoot/scripts/buildsystems/vcpkg.cmake"
    
    Set-Location $buildDir
    cmake .. -G "Visual Studio 17 2022" -A x64 `
      -DBUILD_SHARED_LIBS:BOOL=ON `
      -DPopSift_USE_NVTX_PROFILING:BOOL=OFF `
      -DPopSift_USE_GRID_FILTER:BOOL=OFF `
      -DPopSift_BUILD_DOCS:BOOL=OFF `
      -DPopSift_USE_POSITION_INDEPENDENT_CODE:BOOL=ON `
      -DPopSift_BUILD_EXAMPLES:BOOL=ON `
      -DCMAKE_BUILD_TYPE=$BuildType `
      -DCMAKE_INSTALL_PREFIX="$installDir" `
      -DCMAKE_TOOLCHAIN_FILE="$vcpkgToolchain"
    
    if ($LASTEXITCODE -ne 0) { 
        throw "CMake configuration failed for $BuildType"
    }
    Set-Location ..
}

function Build-AndInstall {
    param([string]$BuildType)
    
    Write-Host "Building and installing $BuildType..."
    $buildDir = "build_$($BuildType.ToLower())"
    
    Set-Location $buildDir
    cmake --build . --config $BuildType --parallel
    if ($LASTEXITCODE -ne 0) { 
        throw "Build failed for $BuildType"
    }
    
    cmake --build . --config $BuildType --target install
    if ($LASTEXITCODE -ne 0) { 
        throw "Install failed for $BuildType"
    }
    Set-Location ..
}

function Build-AsThirdParty {
    param(
        [string]$BuildType,
        [string]$VcpkgRoot,
        [string]$WorkspaceDir
    )
    
    Write-Host "Testing third-party build for $BuildType..."
    $thirdPartyDir = "build_as_3rdparty_$($BuildType.ToLower())"
    $installDir = "$WorkspaceDir/install_$($BuildType.ToLower())"
    $vcpkgToolchain = "$VcpkgRoot/scripts/buildsystems/vcpkg.cmake"
    
    Set-Location $thirdPartyDir
    cmake ../src/application -G "Visual Studio 17 2022" -A x64 `
      -DBUILD_SHARED_LIBS:BOOL=ON `
      -DCMAKE_BUILD_TYPE=$BuildType `
      -DCMAKE_PREFIX_PATH="$installDir" `
      -DCMAKE_TOOLCHAIN_FILE="$vcpkgToolchain"
    
    if ($LASTEXITCODE -ne 0) { 
        throw "Third-party CMake configuration failed for $BuildType"
    }
    
    cmake --build . --config $BuildType --parallel
    if ($LASTEXITCODE -ne 0) { 
        throw "Third-party build failed for $BuildType"
    }
    Set-Location ..
}
