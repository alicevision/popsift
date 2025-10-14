# PopSift Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

### Removed

## [0.10.0] - 2025-10-14

### Added

- Support for device selection and multiple GPUs [PR](https://github.com/alicevision/popsift/pull/121)
- Support for CUDA versions through 12.1 [PR](https://github.com/alicevision/popsift/pull/146)
- GitHub Actions for CI [PR](https://github.com/alicevision/popsift/pull/149)
- Missing thrust include [PR](https://github.com/alicevision/popsift/pull/135)
- Required thrust include for `s_filtergrid.cu` [PR](https://github.com/alicevision/popsift/pull/144)
- Added vlFeat's descriptor extraction method as an option [PR](https://github.com/alicevision/popsift/pull/167)
- Added CI for windows on Github Actions and refactoring [PR](https://github.com/alicevision/popsift/pull/172)

### Changed

- CMake: CUDA as first-order language, different CC selection [PR](https://github.com/alicevision/popsift/pull/156)
- CMake: SM 86 only for CUDA >= 11.1 [PR](https://github.com/alicevision/popsift/pull/116)
- CMake: Suppress deprecated CUDA SM warnings [PR](https://github.com/alicevision/popsift/pull/124)
- Replace exit() calls with thrown exceptions [PR](https://github.com/alicevision/popsift/pull/141)

### Fixed

- Errors with large images [PR](https://github.com/alicevision/popsift/pull/145)
- CMake: Explicit CC list for Jetson and Tegra platforms [PR](https://github.com/alicevision/popsift/pull/163)
- Remove broken and unused code path from L2 normalization [PR](https://github.com/alicevision/popsift/pull/166)
- Do not use NVTX [PR](https://github.com/alicevision/popsift/pull/162)
- Documentation: Biblio build [PR](https://github.com/alicevision/popsift/pull/132)
- Fixed include for CUDA 13 [PR](https://github.com/alicevision/popsift/pull/169)

### Documentation

- Update with vcpkg package [PR](https://github.com/alicevision/popsift/pull/129)

## [0.9] - 2021-02-13

### Added

- Support for CUDA 11 [PR](https://github.com/alicevision/popsift/pull/103)
- Improved checks for CUDA textures [PR](https://github.com/alicevision/popsift/pull/89)
- Documentation: ReadTheDocs integration [PR](https://github.com/alicevision/popsift/pull/100)
- CMake: Support for SM86 compute capability [PR](https://github.com/alicevision/popsift/pull/113)

### Changed

- Docker: Updated CMake and split Docker in 2 parts [PR](https://github.com/alicevision/popsift/pull/87)
- Code cleanup and improvements [PR](https://github.com/alicevision/popsift/pull/95)

### Fixed

- Race conditions identified by racecheck [PR](https://github.com/alicevision/popsift/pull/82)
- CMake: Windows build fixes [PR](https://github.com/alicevision/popsift/pull/92)
- Memory management: pair `malloc` with `free`, not `delete` [PR](https://github.com/alicevision/popsift/pull/108)
- CUDA: Pass correct shared memory size to orientation kernel [PR](https://github.com/alicevision/popsift/pull/109)
- Testing paths fixes [PR](https://github.com/alicevision/popsift/pull/104)
- Remove stale undefined function [PR](https://github.com/alicevision/popsift/pull/90)

### Removed

- Remove boost dependency from core popsift library [PR](https://github.com/alicevision/popsift/pull/81)

## 2019

- Bugfix: Support for images with different resolutions [PR](https://github.com/alicevision/popsift/pull/58)

## 2018

- CMake: Auto-build export symbols for shared libs on Windows [PR](https://github.com/alicevision/popsift/pull/54)
- Bugfix: freeing page-aligned memory on Win32 [PR](https://github.com/alicevision/popsift/pull/53)
- Paper published @MMSys18 (<https://dl.acm.org/doi/10.1145/3204949.3208136>)
- Docker support [PR](https://github.com/alicevision/popsift/pull/46)
- Appveyor CI windows [PR](https://github.com/alicevision/popsift/pull/41)
- Support for Cuda 9 [PR](https://github.com/alicevision/popsift/pull/38)
- Thrust with Cuda 7 [PR](https://github.com/alicevision/popsift/pull/35)

## 2017

- Grid filtering [PR](https://github.com/alicevision/popsift/pull/30)
- Improved Gauss filtering [PR](https://github.com/alicevision/popsift/pull/24)
- Support asynchronous SIFT extraction [PR](https://github.com/alicevision/popsift/pull/22)
- Windows port [PR](https://github.com/alicevision/popsift/pull/18)

## 2016

- Switch to modern CMake [PR](https://github.com/alicevision/popsift/pull/14)
- Travis CI Linux [PR](https://github.com/alicevision/popsift/pull/8)
- First open-source release
