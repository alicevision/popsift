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

// HIP layered images are broken on gfx90a/CDNA2 and on gfx1100/RDNA3 (observed
// ROCm 7.2.1), filed as ROCm/clr#275: after a layered array is written
// layer-by-layer via surf2DLayeredwrite, a read in a later kernel launch
// (tex2DLayered OR surf2DLayeredread, host hipMemcpy3D too) returns a single
// (last-written) layer's data for EVERY layer index -- the layer dimension is
// effectively collapsed. The defect is in the write: surf2DLayeredwrite passed
// the layer index in the mipmap level slot, so every layer landed in the same
// slot. ROCm/rocm-systems#6683 corrects that, and with it every read path above
// returns correct per-layer data, but it is not in ROCm 7.2.x.
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
// Nothing to map: HIP provides the mask-free __shfl/__ballot/__any/__all
// builtins, so assist.h picks them through PopSift_HAVE_SHFL_DOWN_SYNC=0.

#else // NVIDIA / CUDA

#include <cuda_runtime.h>

#endif

#endif // POPSIFT_CUDA_TO_HIP_H
