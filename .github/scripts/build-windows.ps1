# Windows build functions for PopSift
# Usage: source this file and call the individual functions

<#
.SYNOPSIS
Creates the necessary build directories for different build types on Windows.

.DESCRIPTION
Sets up build directories for the main PopSift build and third-party build testing.
Creates directories with lowercase build type names (e.g. build_release) following Windows conventions.

.PARAMETER BuildType
The build configuration (Release or Debug)
#>
function Setup-Directories {
    param([string]$BuildType)
    
    Write-Host "Setting up build directories for $BuildType..."
    $buildDir = "build_$($BuildType.ToLower())"
    $thirdPartyDir = "build_as_3rdparty_$($BuildType.ToLower())"
    
    New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
    New-Item -ItemType Directory -Path $thirdPartyDir -Force | Out-Null
}

<#
.SYNOPSIS
Configures CMake for PopSift build on Windows using vcpkg for dependency management.

.DESCRIPTION
Sets up CMake configuration with Visual Studio 2022 generator, enables shared libraries,
and configures PopSift-specific options. Uses vcpkg manifest mode for dependency management.

.PARAMETER BuildType
The build configuration (Release or Debug)

.PARAMETER VcpkgRoot
Path to the vcpkg installation directory

.PARAMETER WorkspaceDir
Path to the workspace directory for install location
#>
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
      -DVCPKG_INSTALLED_DIR="$env:VCPKG_INSTALLED_DIR" `
      -DCMAKE_TOOLCHAIN_FILE="$vcpkgToolchain"
    
    if ($LASTEXITCODE -ne 0) { 
        throw "CMake configuration failed for $BuildType"
    }
    Set-Location ..
}

<#
.SYNOPSIS
Builds and installs PopSift for the specified build configuration.

.DESCRIPTION
Performs parallel build using all available CPU cores and installs the built
libraries and executables to the configured install directory.

.PARAMETER BuildType
The build configuration (Release or Debug)
#>
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

<#
.SYNOPSIS
Tests building PopSift applications as a third-party consumer on Windows.

.DESCRIPTION
Verifies that the installed PopSift can be found and used by external projects.
This is important for testing the installation and packaging. Uses vcpkg manifest
mode dependencies from the main project build since src/application doesn't have
its own vcpkg.json file.

.PARAMETER BuildType
The build configuration (Release or Debug)

.PARAMETER VcpkgRoot
Path to the vcpkg installation directory

.PARAMETER WorkspaceDir
Path to the workspace directory containing the main build and install
#>
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
    
    # In vcpkg manifest mode, dependencies are installed locally in vcpkg_installed/
    # Since src/application doesn't have vcpkg.json, we need to point to the main project's vcpkg_installed directory so the third-party build can find the dependencies
    $mainProjectVcpkgInstalled = $env:VCPKG_INSTALLED_DIR
    Write-Host "Dependencies installed in $mainProjectVcpkgInstalled..."
    # print first level content of the folder mainProjectVcpkgInstalled
    Get-ChildItem -Path $mainProjectVcpkgInstalled -Directory | ForEach-Object { Write-Host " - $($_.Name)" }

    Set-Location $thirdPartyDir
    cmake ../src/application -G "Visual Studio 17 2022" -A x64 `
      -DBUILD_SHARED_LIBS:BOOL=ON `
      -DCMAKE_BUILD_TYPE=$BuildType `
      -DCMAKE_PREFIX_PATH="$installDir;$mainProjectVcpkgInstalled/x64-windows" `
      -DCMAKE_TOOLCHAIN_FILE="$vcpkgToolchain" `
      -DVCPKG_INSTALLED_DIR="$mainProjectVcpkgInstalled" `
      -DVCPKG_TARGET_TRIPLET=x64-windows
    
    if ($LASTEXITCODE -ne 0) { 
        throw "Third-party CMake configuration failed for $BuildType"
    }
    
    cmake --build . --config $BuildType --parallel
    if ($LASTEXITCODE -ne 0) { 
        throw "Third-party build failed for $BuildType"
    }
    Set-Location ..
}
