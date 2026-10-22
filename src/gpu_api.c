#ifdef __USE_CUDA
#include <cuda_runtime.h>
#endif
#include <stdio.h>

void helloCuda(){
#ifdef __USE_CUDA
  int driverVersion = 0;
  cudaDriverGetVersion(&driverVersion);
  printf("CUDA driver: %d\n", driverVersion);

  int runtimeVersion = 0;
  cudaRuntimeGetVersion(&runtimeVersion);
  printf("CUDA runtime: %d\n", runtimeVersion);

  int numDevices;
  cudaError_t stat = cudaGetDeviceCount(&numDevices);

  for (int i = 0; i < numDevices; i++){
    struct cudaDeviceProp prop;
    stat = cudaGetDeviceProperties(&prop, i);

    printf("%d: %s, CC %d.%d, %d SMs, %luMB\n", i, prop.name,
            prop.major, prop.minor,
            prop.multiProcessorCount,
            prop.totalGlobalMem/1024/1024);
    }
#else
  printf("no-gpu\n");
#endif
}
