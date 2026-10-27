#ifndef GPU_API
#define GPU_API

#include <stddef.h>

void helloCuda();

void gpu_red_black_schwarz_api(
    const int cycles,
    const int block_thread_start[8], const int block_thread_end[8],
    const int res, const int init_res, const int start, const int end,
    const int method, const int num_blocks, const int *block_list,
    const size_t elem_bytes,
    void *Dphi, void *eta, void *r, void *x, void *latest_iter);

#endif
