#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#include <stdio.h>
#include <cuda_runtime.h>

// Print file, line and message if a CUDA call fails
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
  printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); } while (0)

#endif
