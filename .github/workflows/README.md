# PopSift Continuous Integration Pipeline — Developer Documentation

## Overview

The PopSift CI pipeline is designed for cross-platform builds and tests on both Linux and Windows, leveraging GitHub Actions. It uses a modular approach with reusable scripts and a composite action to minimize code duplication, maximize maintainability, and provide clear step-by-step feedback.

---

## Pipeline Structure

Workflow File

- Location: [`continuous-integration.yml`](continuous-integration.yml)
- Triggers:
  - On push to `master` or `develop`
  - On pull requests (except for documentation-only changes)
  - Manual trigger via `workflow_dispatch`
- Jobs:
  - `build-linux`: Runs in Docker containers with pre-installed dependencies.
  - `build-windows`: Runs on Windows runners, installs CUDA and manages dependencies via vcpkg.

---

## Key Components

1. Composite Action: `build-config`
    - **Location**: [`action.yml`](../actions/build-config/action.yml)
    - **Purpose**:
      Encapsulates the build logic for both Linux and Windows, parameterized by the build type (`Release`/`Debug`) and platform.
    - **Inputs**:
        - `build-type`: `Release` or `Debug`
        - `platform`: `linux` or `windows`
        - `deps-install-dir`: (Linux) Path to dependencies (default: opt)
        - `vcpkg-root`: (Windows) Path to vcpkg installation
        - `workspace-dir`: (Windows) Path to workspace for the install location
    - **Steps**:
        - Calls platform-specific scripts for:
            - Directory setup
            - CMake configuration
            - Build and install
            - Third-party consumer build test

2. Linux Build Script
    - **Location**: build-linux.sh
    - **Functions**:
        - `setup_directories`: Prepares the build and install directories.
        - `configure_cmake`: Configures CMake with appropriate options and dependency paths.
        - `build_and_install`: Builds and installs PopSift using parallel jobs.
        - `build_as_third_party`: Tests PopSift as a third-party dependency by building the application against the installed library.
    - **Notes**:
        - Uses pre-built Docker images with all dependencies.
        - No vcpkg or dependency caching is needed.

3. Windows Build Script
    - **Location**: build-windows.ps1
    - **Functions**:
        - `Setup-Directories`: Prepares build and third-party directories.
        - `Configure-CMake`: Configures CMake with Visual Studio generator, CUDA toolset, and vcpkg toolchain.
        - `Build-AndInstall`: Builds and installs PopSift for the specified configuration.
        - `Build-AsThirdParty`: Tests PopSift as a third-party dependency, ensuring the install is usable by external projects.
    - **Notes**:
        - Handles CUDA installation and environment setup.
        - Uses vcpkg in manifest mode, with dependencies installed to a shared directory for both build types.

---

## Caching and Dependency Management

- **CUDA**:
  - Cached on Windows to speed up repeated runs.
  - If not cached, installed via the `Jimver/cuda-toolkit` action.
  - Environment variables (`CUDA_PATH`, `CUDA_BIN_PATH`, etc.) are set for all subsequent steps.

- **vcpkg**:
  - Cached on Windows for both the source and installed dependencies.
  - Uses manifest mode (vcpkg.json in project root).
  - VCPKG_INSTALLED_DIR is set to a shared location to avoid redundant installs for Release/Debug.

## Debugging and Diagnostics

- **Environment Variables**:
  - Steps print CUDA and vcpkg environment variables for transparency.
- **nvcc Check**:
  - Ensures `nvcc` is available in the `PATH` before CMake configuration.
- **vcpkg Installed Packages**:
  - Lists installed vcpkg packages for verification.

---

## Extending or Modifying the Pipeline

- To add a new build type:
  - Add another call to the composite action with the desired `build-type`.
- To add new dependencies:
  - Update [`vcpkg.json`](../../vcpkg.json) and the Docker images as needed.
- To change the build logic:
  - Edit the relevant script ([`build-linux.sh`](../scripts/build-linux.sh) or [`build-windows.ps1`](../scripts/build-windows.ps1)).
- To add more diagnostics:
  - Insert additional steps in the workflow or scripts as needed.
