# Prep-for-SYCL: PopSift C++ Refactoring

Prep-for-SYCL is an intermediate C++ refactoring of the PopSift CUDA implementation, designed as a stepping stone toward a cross-vendor SYCL port. It represents Phase 1 of a two-stage migration strategy, transforming vendor-locked CUDA code into vendor-neutral C++ while preserving algorithmic correctness.

## Overview

Prep-for-SYCL strips away CUDA-specific optimizations and constructs from PopSift, converting GPU device kernels into sequential CPU-based implementations while maintaining the core SIFT algorithm's integrity. This intermediate stage serves three critical purposes:

1. **Algorithm Validation**: Provides a reference implementation to verify algorithmic correctness against the original CUDA version
2. **Knowledge Transfer**: Deepens understanding of the SIFT algorithm and PopSift's implementation details by forcing sequential, CPU-based comprehension
3. **Bridge to SYCL**: Creates a vendor-neutral C++ baseline that simplifies the transition to SYCL's heterogeneous execution model

## Motivation: The Two-Stage Approach

Direct translation from CUDA to SYCL is conceptually complex because it requires simultaneously:
- Understanding SYCL's abstractions (buffers, work-items, work-groups)
- Eliminating CUDA-specific constructs
- Learning new parallel programming paradigms

The two-stage approach separates these concerns:

1. **Stage 1 (Prep-for-SYCL)**: Focus exclusively on removing CUDA specifics and creating a pure C++ baseline
2. **Stage 2 (PopSift-SYCL)**: Translate the vendor-neutral C++ implementation to SYCL with full knowledge of algorithmic requirements

This methodology offers several advantages:

- **Incremental Understanding**: Implementing SIFT sequentially in C++ builds deep algorithmic comprehension before introducing parallelization complexity
- **Simplified Refactoring**: SYCL translation becomes clearer when working from sequential C++ rather than parallel CUDA
- **Correctness Assurance**: The C++ version can be validated against CUDA, providing confidence that PopSift-SYCL performance differences stem from design choices rather than algorithmic errors
- **Educational Value**: The sequential implementation clearly exposes data dependencies, memory access patterns, and control flow that parallel GPU code obscures

## Key Refactoring Steps

### Decoupling Host and Device Code

The original PopSift tightly couples host and device execution contexts through CUDA-specific abstractions:
- Device memory structures and pitched allocations were converted to standard C++ containers with appropriate alignment
- CUDA texture memory, used for efficient hardware-accelerated bilinear interpolation, was replaced with explicit CPU interpolation functions
- Kernel launches with explicit grid and block configuration were replaced with nested loops, making computational ordering explicit

### Removing CUDA-Specific Constructs

All vendor-specific language features were systematically eliminated:

- **Execution Configuration**: Kernel launch syntax (`<<<gridDim, blockDim>>>`) replaced with function calls
- **Thread Synchronization**: `__syncthreads()` primitives removed or converted to sequential ordering guarantees
- **Shared Memory**: Device shared memory allocations converted to stack-allocated arrays or eliminated entirely
- **Device Function Qualifiers**: `__device__`, `__global__` removed; all functions become ordinary C++ functions
- **Memory Management**: CUDA runtime API (`cudaMalloc`, `cudaMemcpy`) replaced with standard C++ allocation (`new`, `std::vector`)
- **Intrinsics and Atomics**: CUDA-specific intrinsics (`__shfl_*`, `__popc`, `__ballot`) reimplemented using standard C++

### Algorithm Validation and Correctness Testing

The refactoring was performed incrementally, validating each SIFT pipeline stage:

- **Gaussian Pyramid**: Separable Gaussian filtering verified against CUDA outputs
- **Keypoint Detection**: Extrema detection and subpixel refinement validated
- **Dominant Orientation**: 36-bin histogram computation and peak detection verified
- **Descriptor Extraction**: 128-dimensional descriptor generation validated

Numerical differences between C++ and CUDA outputs were expected due to:
- Floating-point precision variations from different computation orders
- Compiler optimizations affecting intermediate results
- Differences between sequential and non-deterministic GPU execution

These discrepancies remain within acceptable tolerances for feature matching applications, confirming algorithmic equivalence.


## Dependencies

Prep-for-SYCL depends on:

* **C++ Compiler** supporting C++14 or later
* **CMake** >= 3.10

Optionally, for building example applications:

* **Boost** >= 1.71 (required components: {atomic, chrono, date-time, system, thread}-dev)
* **DevIL** (libdevil-dev) for loading a broader range of image formats; otherwise only PGM is supported

## Build

```bash
mkdir build && cd build
cmake ..
make
make install
```

## Performance Characteristics

As a sequential CPU implementation, Prep-for-SYCL is significantly slower than the original CUDA PopSift. Processing times depend on:

- Image resolution
- Number of detected keypoints
- Host CPU performance
- Compiler optimization level

Performance is not the primary goal; algorithmic correctness and clarity are paramount. Execution times serve as a baseline reference for comparing the final SYCL implementation's performance against pure CPU sequential execution.

## Lessons Learned

The sequential C++ implementation revealed several insights that informed the SYCL translation:

1. **GPU Memory Hierarchy Optimization**: PopSift heavily exploited shared memory and texture caches; algorithms optimized for these hierarchies required fundamental rethinking for CPU sequential execution

2. **Explicit Data Dependencies**: Sequential execution exposed subtle data dependencies and memory access patterns that parallel GPU execution obscures

3. **Interpolation Operations**: Manually implementing bilinear and trilinear interpolation (rather than relying on GPU texture hardware) demonstrated the operations that GPU hardware handles transparently. This experience directly guided the SYCL implementation, as AdaptiveCPP's lack of SYCL image support necessitated the same manual approach

4. **Parallelization Opportunities**: Explicit sequential execution clarified which portions of the pipeline could be parallelized independently versus those with tight dependencies

5. **Numerical Stability**: Understanding floating-point precision variations between different execution orders proved essential for validating correctness across different parallel implementations

## Relation to PopSift-SYCL

Prep-for-SYCL serves as the foundation for PopSift-SYCL (Phase 2). The C++ refactoring:

- **Provides Algorithmic Baseline**: All subsequent SYCL kernels are designed and validated against this C++ reference
- **Informs Kernel Design**: Understanding sequential control flow and data dependencies directly informs SYCL work-item and work-group design
- **Guides Optimization Strategy**: Insights from sequential bottlenecks inform which SYCL kernels warrant optimization effort

## Building and Running Examples

After building Prep-for-SYCL, example applications demonstrate the API:

```bash
# Run the demo application
./popsift-demo -i image.pgm 

# The application will:
# 1. Load the grayscale image
# 2. Extract SIFT features sequentially on CPU
# 3. Display timing information for each pipeline stage
# 4. Optionally output results to a file
```

## Repository

This intermediate implementation is maintained in the `wip/prep-for-sycl` branch of the PopSift repository:

```
https://github.com/alicevision/popsift/tree/wip/prep-for-sycl
```

## References

[1] Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision, 60(2), 91–110. doi:10.1023/B:VISI.0000029664.99615.94

[2] Griwodz, C., Calvet, L., & Halvorsen, P. (2018). Popsift: A Faithful SIFT Implementation for Real-time Applications. Proceedings of the 9th ACM Multimedia Systems Conference (pp. 415–420).
