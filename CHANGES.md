# PopSift Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

### Removed

## [1.0.0] - YYYY-MM-DD

### Added
- Improved checks for CUDA textures [PR](https://github.com/alicevision/popsift/pull/89)
- CMake: Improved support for all Cuda CC [PR](https://github.com/alicevision/popsift/pull/75)
- CMake: support for cuda 11 [PR](https://github.com/alicevision/popsift/pull/103)
- Support for Cuda CC 7 cards (RTX 2080) [PR](https://github.com/alicevision/popsift/pull/67)
- Support for Boost 1.70 [PR](https://github.com/alicevision/popsift/pull/65)
- Support for device selection and multiple GPUs [PR](https://github.com/alicevision/popsift/pull/121)

### Fixed
- CMake: fixes to allow building on Windows using vcpkg [PR](https://github.com/alicevision/popsift/pull/92)
- Fix race condition [PR](https://github.com/alicevision/popsift/pull/82)

### Changed
- Improved resource releasing [PR](https://github.com/alicevision/popsift/pull/71)

### Removed
- Remove boost dependency from the main library [PR](https://github.com/alicevision/popsift/pull/81)


## 2019

- Bugfix: Support for images with different resolutions [PR](https://github.com/alicevision/popsift/pull/58)


## 2018

- CMake: Auto-build export symbols for shared libs on Windows [PR](https://github.com/alicevision/popsift/pull/54)
- Bugfix: freeing page-aligned memory on Win32 [PR](https://github.com/alicevision/popsift/pull/53)
- Paper published @MMSys18 (https://dl.acm.org/doi/10.1145/3204949.3208136)
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
