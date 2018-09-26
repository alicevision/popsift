#ifndef MEMARR_H
#define MEMARR_H

#include <vector>
#include "helper.hpp"

template<typename T>
class arr {
 public:

  size_t len;
  T *host;
  T *device;

  arr(size_t l): len(l){}


  size_t size() const {
    return len * sizeof(T);
  }

  void mallocDevice(){
    SAFE_CUDA_CALL(cudaMalloc( (void**)&device, len * sizeof(T) ));
  }

  void mallocHost(){
    host = new T[len];
  }


  void toDevice() {
    SAFE_CUDA_CALL(cudaMemcpy( device, host, len * sizeof(T), cudaMemcpyHostToDevice ));
  }

  void toHost() {
    SAFE_CUDA_CALL(cudaMemcpy( host, device, len * sizeof(T), cudaMemcpyDeviceToHost ));
  }



};

#endif

