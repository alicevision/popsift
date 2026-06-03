//********************************************************//
// CUDA-to-HIP compatibility shim for PopSift             //
//                                                        //
// Minimal-footprint port: every other                    //
// source file keeps its plain CUDA spelling. On AMD this  //
// header includes the HIP runtime and #defines the CUDA   //
// symbols the project uses to their HIP equivalents. On   //
// NVIDIA it is a no-op that pulls in <cuda_runtime.h>.    //
//                                                        //
// Symbol names follow PyTorch's authoritative hipify map: //
// torch/utils/hipify/cuda_to_hip_mappings.py             //
//********************************************************//

#ifndef POPSIFT_CUDA_TO_HIP_H
#define POPSIFT_CUDA_TO_HIP_H

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>
// Backward-compat note (ROCm 7.2.x -> newer ROCm): newer ROCm's
// <hip/hip_bf16.h> defines real __shfl_*_sync<...> functions whose names collide
// with the function-like __shfl_*_sync macros defined further below -- if a
// rocThrust/rocprim include pulls that header in AFTER the macros, the macros
// rewrite the header's own definitions and it fails to compile ("use of
// undeclared identifier 'mask'"). Pull the header in FIRST so the real functions
// are defined before the macros. On older ROCm (7.2.x) the header exists and has
// no such functions, so this is a harmless no-op there -- guarded with
// __has_include so it also tolerates any ROCm that ships without the header.
// Only needed where rocThrust/rocprim is pulled in (the device .cu TUs, compiled
// by HIP clang); gate on __clang__ so the plain-C++ host example consumers (which
// may be built by gcc and never include rocThrust) do not try to parse hip_bf16.h,
// whose vector intrinsics require clang builtins.
#if defined(__clang__) && defined(__has_include)
#  if __has_include(<hip/hip_bf16.h>)
#    include <hip/hip_bf16.h>
#  endif
#endif

// ---- Error handling ----
#define cudaError_t            hipError_t
#define cudaSuccess            hipSuccess
#define cudaGetErrorString     hipGetErrorString
#define cudaGetLastError       hipGetLastError
#define cudaPeekAtLastError    hipPeekAtLastError
#define cudaDeviceSynchronize  hipDeviceSynchronize
#define cudaDeviceReset        hipDeviceReset

// ---- Device management ----
#define cudaGetDeviceCount       hipGetDeviceCount
#define cudaGetDevice            hipGetDevice
#define cudaGetDeviceProperties  hipGetDeviceProperties
#define cudaSetDevice            hipSetDevice
#define cudaDeviceProp           hipDeviceProp_t
#define cudaDeviceSetLimit       hipDeviceSetLimit

// ---- Events / streams ----
#define cudaEvent_t            hipEvent_t
#define cudaEventCreate        hipEventCreate
#define cudaEventDestroy       hipEventDestroy
#define cudaEventRecord        hipEventRecord
#define cudaEventSynchronize   hipEventSynchronize
#define cudaEventElapsedTime   hipEventElapsedTime
#define cudaStream_t           hipStream_t
#define cudaStreamCreate       hipStreamCreate
#define cudaStreamDestroy      hipStreamDestroy
#define cudaStreamSynchronize  hipStreamSynchronize
#define cudaStreamWaitEvent    hipStreamWaitEvent

// ---- Linear / pitched / host memory ----
#define cudaMalloc         hipMalloc
#define cudaMallocManaged  hipMallocManaged
#define cudaMallocHost     hipHostMalloc
#define cudaFree           hipFree
#define cudaFreeHost       hipHostFree
#define cudaMallocPitch    hipMallocPitch
#define cudaHostRegister   hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostRegisterMapped   hipHostRegisterMapped

#define cudaMemcpy         hipMemcpy
#define cudaMemcpyAsync    hipMemcpyAsync
#define cudaMemcpy2D       hipMemcpy2D
#define cudaMemcpy2DAsync  hipMemcpy2DAsync
#define cudaMemset         hipMemset
#define cudaMemsetAsync    hipMemsetAsync

// ---- 3D / layered memory (Gaussian pyramid arrays) ----
#define cudaMemcpy3D           hipMemcpy3D
#define cudaMemcpy3DParms      hipMemcpy3DParms
#define cudaPitchedPtr         hipPitchedPtr
#define cudaExtent             hipExtent
#define cudaPos                hipPos
#define make_cudaPitchedPtr    make_hipPitchedPtr
#define make_cudaExtent        make_hipExtent
#define make_cudaPos           make_hipPos

// ---- Constant / symbol memory ----
#define cudaMemcpyToSymbol         hipMemcpyToSymbol
#define cudaMemcpyToSymbolAsync    hipMemcpyToSymbolAsync
#define cudaMemcpyFromSymbol       hipMemcpyFromSymbol
#define cudaMemcpyFromSymbolAsync  hipMemcpyFromSymbolAsync
#define cudaGetSymbolAddress       hipGetSymbolAddress

// ---- memcpy kinds ----
#define cudaMemcpyKind            hipMemcpyKind
#define cudaMemcpyHostToDevice    hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost    hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice  hipMemcpyDeviceToDevice
#define cudaMemcpyHostToHost      hipMemcpyHostToHost

// ---- CUDA arrays + channel format (texture/surface backing store) ----
#define cudaArray                    hipArray
#define cudaArray_t                  hipArray_t
#define cudaMallocArray              hipMallocArray
#define cudaMalloc3DArray            hipMalloc3DArray
#define cudaFreeArray                hipFreeArray
#define cudaArrayLayered             hipArrayLayered
#define cudaArraySurfaceLoadStore    hipArraySurfaceLoadStore
#define cudaArrayDefault             hipArrayDefault
#define cudaChannelFormatDesc        hipChannelFormatDesc
#define cudaCreateChannelDesc        hipCreateChannelDesc
#define cudaChannelFormatKindFloat    hipChannelFormatKindFloat
#define cudaChannelFormatKindUnsigned hipChannelFormatKindUnsigned
#define cudaChannelFormatKindSigned   hipChannelFormatKindSigned

// ---- Texture objects ----
#define cudaTextureObject_t       hipTextureObject_t
#define cudaCreateTextureObject   hipCreateTextureObject
#define cudaDestroyTextureObject  hipDestroyTextureObject
#define cudaResourceDesc          hipResourceDesc
#define cudaTextureDesc           hipTextureDesc
#define cudaResourceTypePitch2D   hipResourceTypePitch2D
#define cudaResourceTypeArray     hipResourceTypeArray
#define cudaResourceTypeLinear    hipResourceTypeLinear
#define cudaAddressModeClamp      hipAddressModeClamp
#define cudaAddressModeWrap       hipAddressModeWrap
#define cudaFilterModeLinear      hipFilterModeLinear
#define cudaFilterModePoint       hipFilterModePoint
#define cudaReadModeElementType        hipReadModeElementType
#define cudaReadModeNormalizedFloat    hipReadModeNormalizedFloat
#define cudaTextureType2D         hipTextureType2D
#define cudaTextureType2DLayered  hipTextureType2DLayered

// ---- Surface objects (pyramid write path) ----
#define cudaSurfaceObject_t       hipSurfaceObject_t
#define cudaCreateSurfaceObject   hipCreateSurfaceObject
#define cudaDestroySurfaceObject  hipDestroySurfaceObject
#define cudaBoundaryModeZero      hipBoundaryModeZero
#define cudaBoundaryModeClamp     hipBoundaryModeClamp
#define cudaBoundaryModeTrap      hipBoundaryModeTrap

// HIP layered-image coherency is broken on gfx90a/CDNA2 (observed ROCm 7.2.1),
// filed as ROCm/clr#275 (the partial fix ROCm/rocm-systems#6683 covers only
// surf2DLayered): after a layered array is written layer-by-layer via
// surf2DLayeredwrite, a read in a later kernel launch (tex2DLayered OR
// surf2DLayeredread, host hipMemcpy3D too) returns a single (last-written) layer's
// data for EVERY layer index -- the layer dimension is effectively collapsed.
// A standalone reproducer confirms it and also
// confirms a NON-layered 3D array (surf3Dwrite/surf3Dread/tex3D) is fully
// coherent across launches. So on HIP the pyramid arrays are allocated as
// non-layered 3D arrays (see sift_octave.cu) and the layered builtins map to the
// 3D ones with the layer index used as the z coordinate (a 1:1 mapping: the
// element write/read addresses are identical). CUDA keeps real layered arrays and
// the native builtins, byte-for-byte unchanged.
//
// HIP's surf3Dwrite has no boundary-mode parameter (CUDA's surf2DLayeredwrite has
// a trailing 6th arg); hipBoundaryModeZero matches the AMD image-store default
// (out-of-range writes dropped), so dropping the argument is faithful.
template <typename T>
__device__ __forceinline__ void popsift_surf2DLayeredwrite(
    T data, hipSurfaceObject_t surfObj, int x, int y, int layer,
    int /*boundaryMode*/)
{
    surf3Dwrite(data, surfObj, x, y, layer);
}
#define surf2DLayeredwrite(data, surf, x, y, layer, mode) \
    popsift_surf2DLayeredwrite((data), (surf), (x), (y), (layer), (mode))

// ---- Directed-rounding FP intrinsics ----
// HIP/AMD does not provide the round-toward-+infinity intrinsics. PopSift's only
// uses are descriptor weight accumulation where the operands are non-negative and
// the comment notes _ru is merely a fast round; the round-to-nearest form is the
// faithful HIP equivalent.
#define __fmaf_ru(a, b, c)  __fmaf_rn((a), (b), (c))
#define __fmul_ru(a, b)     __fmul_rn((a), (b))

// ---- Warp intrinsics ----
// PopSift (assist.h) passes the CUDA 32-bit full mask 0xffffffff to the *_sync
// builtins. On a 64-lane CDNA wavefront a uint32 mask is meaningless; HIP's
// mask-free builtins poll the whole active wavefront, which is the faithful
// equivalent of "operate over the (sub)warp" on the blocks PopSift launches.
// Map the _sync forms (both the 2-arg and the explicit-width 3/4-arg overloads
// resolve to these) to the mask-free HIP builtins.
#define __shfl_sync(mask, ...)        __shfl(__VA_ARGS__)
#define __shfl_up_sync(mask, ...)     __shfl_up(__VA_ARGS__)
#define __shfl_down_sync(mask, ...)   __shfl_down(__VA_ARGS__)
#define __shfl_xor_sync(mask, ...)    __shfl_xor(__VA_ARGS__)
#define __ballot_sync(mask, pred)     __ballot(pred)
#define __any_sync(mask, pred)        __any(pred)
#define __all_sync(mask, pred)        __all(pred)

#else // NVIDIA / CUDA

#include <cuda_runtime.h>

#endif

#endif // POPSIFT_CUDA_TO_HIP_H
