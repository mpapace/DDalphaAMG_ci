#include <stdio.h>
#include <stddef.h>
#ifdef __USE_CUDA
#include "cuda_schwarz_PRECISION.h"
#include "cuda_oddeven_PRECISION.h"
#endif
#include "gpu_api_PRECISION.h"

void gpu_red_black_schwarz_api_PRECISION(
    const int cycles,
    const int block_thread_start[8], const int block_thread_end[8],
    const int res, const int init_res, const int start, const int end,
    const int method, const int num_blocks, const int *block_list,
    const size_t elem_bytes,
    void *Dphi, void *eta, void *r, void *x, void *latest_iter)
{
#ifdef __USE_CUDA
  cuda_red_black_schwarz_PRECISION(cycles, block_thread_start, block_thread_end,
                                   res, init_res, start, end,
                                   method, num_blocks, block_list, elem_bytes,
                                   Dphi, eta, r, x, latest_iter);
#else
  printf("no-gpu\n");
#endif
}

void gpu_LU_perform_fwd_bwd_subs_PRECISION(const int n_sites, const size_t elem_bytes,
                                           void *leta, void *lphi, void *clover)
{
#ifdef __USE_CUDA
  cuda_LU_perform_fwd_bwd_subs_PRECISION(n_sites, elem_bytes, leta, lphi, clover);
#endif
}
