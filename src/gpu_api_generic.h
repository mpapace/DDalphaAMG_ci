#ifndef GPU_API_PRECISION_HEADER
#define GPU_API_PRECISION_HEADER

void gpu_red_black_schwarz_api_PRECISION(
    const int cycles,
    const int block_thread_start[8], const int block_thread_end[8],
    const int res, const int init_res, const int start, const int end,
    const int method, const int num_blocks, const int *block_list,
    const size_t elem_bytes,
    void *Dphi, void *eta, void *r, void *x, void *latest_iter);

void gpu_LU_perform_fwd_bwd_subs_PRECISION(const int n_sites, const size_t elem_bytes,
                                           void *leta, void *lphi, void *clover);
#endif
