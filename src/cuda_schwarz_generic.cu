/*
 * Copyright (C) 2016, Matthias Rottmann, Artur Strebel, Simon Heybrock, Simone Bacchio, Bjoern Leder.
 *
 * This file is part of the DDalphaAMG solver library.
 *
 * The DDalphaAMG solver library is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The DDalphaAMG solver library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *
 * You should have received a copy of the GNU General Public License
 * along with the DDalphaAMG solver library. If not, see http://www.gnu.org/licenses/.
 *
 */

#include <stdio.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include "cuda_check.h"

extern "C"
void cuda_red_black_schwarz_PRECISION(
    const int cycles,
    const int block_thread_start[8], const int block_thread_end[8],
    const int res, const int init_res, const int start, const int end,
    const int method, const int num_blocks, const int *block_list,
    const size_t elem_bytes,
    void *Dphi, void *eta, void *r, void *x, void *latest_iter)
{
  const size_t vbytes = (size_t)end * elem_bytes;
  const int bl_n = (method == 2) ? 8 * num_blocks
                 : (method == 3) ? 16 * num_blocks : 0;

  int  *cuda_block_thread_start = NULL;
  int  *cuda_block_thread_end   = NULL;
  int  *cuda_block_list         = NULL;
  void *cuda_Dphi        = NULL;
  void *cuda_eta         = NULL;
  void *cuda_r           = NULL;
  void *cuda_x           = NULL;
  void *cuda_latest_iter = NULL;

  CK(cudaMalloc(&cuda_block_thread_start, 8 * sizeof(int)));
  CK(cudaMalloc(&cuda_block_thread_end,   8 * sizeof(int)));
  if (bl_n) CK(cudaMalloc(&cuda_block_list, bl_n * sizeof(int)));
  CK(cudaMalloc(&cuda_Dphi,        vbytes));
  CK(cudaMalloc(&cuda_eta,         vbytes));
  CK(cudaMalloc(&cuda_r,           vbytes));
  CK(cudaMalloc(&cuda_x,           vbytes));
  CK(cudaMalloc(&cuda_latest_iter, vbytes));

  CK(cudaMemcpy(cuda_block_thread_start, block_thread_start, 8 * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(cuda_block_thread_end,   block_thread_end,   8 * sizeof(int), cudaMemcpyHostToDevice));
  if (bl_n) CK(cudaMemcpy(cuda_block_list, block_list, bl_n * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(cuda_Dphi,        Dphi,        vbytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(cuda_eta,         eta,         vbytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(cuda_r,           r,           vbytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(cuda_x,           x,           vbytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(cuda_latest_iter, latest_iter, vbytes, cudaMemcpyHostToDevice));

  printf("\nWe are on GPU\n");

  CK(cudaFree(cuda_block_thread_start));
  CK(cudaFree(cuda_block_thread_end));
  CK(cudaFree(cuda_block_list));      
  CK(cudaFree(cuda_Dphi));
  CK(cudaFree(cuda_eta));
  CK(cudaFree(cuda_r));
  CK(cudaFree(cuda_x));
  CK(cudaFree(cuda_latest_iter));
}
