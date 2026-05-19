# PopSift-SYCL

PopSift-SYCL is a cross-vendor, portable implementation of the SIFT algorithm using SYCL. It is a faithful port of the original PopSift CUDA implementation, designed to be available on a wider range of platforms and GPU vendors while maintaining high performance and correctness.

## Overview

The Scale-Invariant Feature Transform (SIFT) algorithm is one of the most widely used image-matching algorithms, consisting of keypoint detection and descriptor extraction stages. PopSift-SYCL implements SIFT for real-time feature extraction on consumer and professional GPUs, as well as CPUs, without compromising the details of the SIFT algorithm described in David Lowe's famous paper [1].

Unlike the original PopSift, which is vendor-locked to Nvidia GPUs via CUDA, PopSift-SYCL achieves cross-vendor compatibility by leveraging SYCL. This allows it to be compiled and executed on systems with AMD, Intel, or Nvidia GPUs, as well as on CPU-only systems, thereby making it accessible to a broader community of developers and researchers.

PopSift-SYCL can be compiled by both Intel's DPC++ compiler and the open-source AdaptiveCPP compiler, providing flexibility in toolchain selection and deployment options.

## HW Requirements

PopSift-SYCL is designed to work with a wide range of modern GPUs and CPUs:

- **Nvidia GPUs**: Compute capability >= 3.5 (tested on RTX 4050, V100, and other modern GPUs)
- **Intel GPUs**: Arc series and other Intel discrete GPUs supported by oneAPI
- **AMD GPUs**: Supported through AdaptiveCPP
- **CPU**: Can execute on multi-core CPUs with SYCL support

The code has been developed and tested with modern GPUs in mind, but portable SYCL implementations allow execution across various hardware configurations.

## Dependencies

PopSift-SYCL depends on:

* **Compiler**: One of the following:
  - Intel DPC++ compiler (part of Intel oneAPI toolkit)
  - AdaptiveCPP (open-source SYCL compiler)

* **SYCL Runtime**: Appropriate runtime for the target platform
  - Intel Level-Zero for Intel GPUs
  - CUDA backend for Nvidia GPUs
  - HIP backend for AMD GPUs

* **CMake** >= 3.18

Optionally, for building example applications:

* **Boost** >= 1.71 (required components: {atomic, chrono, date-time, system, thread}-dev)
* **DevIL** (libdevil-dev) for loading a broader range of image formats; otherwise only PGM is supported

## Build

To build PopSift-SYCL with DPC++:

```bash
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=dpcpp
make
make install
```

To build PopSift-SYCL with AdaptiveCPP:

```bash
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=acpp
make
make install
```

## Implementation Highlights

PopSift-SYCL maintains the core SIFT algorithm while adopting a synchronous processing model for research clarity. Key implementation features include:

- **Gaussian Pyramid Construction**: Separable Gaussian filtering with bilinear interpolation explicitly implemented in device code
- **Keypoint Detection**: One-work-item-per-pixel approach using bitmask operations for extrema detection
- **Dominant Orientation Computation**: One-work-item-per-extremum with 36-bin orientation histogram smoothing
- **Descriptor Extraction**: Loop-based approach computing 128-dimensional descriptors with trilinear interpolation
- **Cross-Vendor Compatibility**: Single Source Multiple Compilers (SSMC) design enabling compilation with multiple backends

## Usage

The main artifact created is `libpopsift.so`.
If enabled, the test application `popsift-demo` is created as well.
The only mandatory parameter is the input image, provided via the `-i` option, which takes the image path as its argument (e.g., `./popsift-demo -i <testImage>`).



## Performance

PopSift-SYCL achieves competitive performance across multiple platforms by leveraging SYCL's abstract execution model and backend-specific optimizations. The implementation has been tested on:

- **Nvidia GPUs**: RTX 4050, V100
- **Intel GPUs**: Arc series
- **CPU backends**: Multi-core CPU execution via OpenCL

Performance comparisons with the original CUDA PopSift are presented in the associated research paper [3], demonstrating the trade-offs between absolute performance and cross-vendor portability.

## Cite Us

If you use PopSift-SYCL for your publication, please cite us as:

```bibtex
@inproceedings{AlKhafaji2026PopSiftSYCL,
    author = {Al Khafaji, Mohammad Fadel and Griwodz, Carsten and Stensland, H{\aa}kon Kvale},
    title = {A cross-vendor implementation of PopSift using SYCL},
    booktitle = {Proceedings of the 14th International Workshop on OpenCL and SYCL},
    series = {IWOCL '26},
    year = {2026},
    location = {Heilbronn, Germany},
    month = {May}
}
```

And the original PopSift:

```bibtex
@inproceedings{Griwodz2018Popsift,
    author = {Griwodz, Carsten and Calvet, Lilian and Halvorsen, P{\aa}l},
    title = {Popsift: A Faithful SIFT Implementation for Real-time Applications},
    booktitle = {Proceedings of the 9th ACM Multimedia Systems Conference},
    series = {MMSys '18},
    year = {2018},
    isbn = {978-1-4503-5192-8},
    location = {Amsterdam, Netherlands},
    pages = {415--420},
    numpages = {6},
    doi = {10.1145/3204949.3208136},
    acmid = {3208136},
    publisher = {ACM},
    address = {New York, NY, USA},
}
```

## References

[1] Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision, 60(2), 91–110. doi:10.1023/B:VISI.0000029664.99615.94

[2] Griwodz, C., Calvet, L., & Halvorsen, P. (2018). Popsift: A Faithful SIFT Implementation for Real-time Applications. Proceedings of the 9th ACM Multimedia Systems Conference (pp. 415–420).

[3] M. F. A. Khafaji, C. Griwodz and H. K. Stensland. “A Cross-vendor Implementation of PopSift using SYCL”. In: Proceedings of the International Workshop on OpenCL and SYCL (IWOCL’26). ACM, 2026. isbn: 979-8-4007-2499-2. doi: 10.1145/3811257.3811261.
