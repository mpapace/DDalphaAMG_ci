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
#include <thrust/complex.h>
#include "cuda_check.h"

typedef thrust::complex<PRECISION> cplx_PRECISION;

// Solves L*U*x = b for one lattice site: two BS x BS blocks
// (BS = 6: single flavour, BS = 12: doublet)
// U is given by u_ii = 1, u_ij = l_ji* / l_ii (modified LU, stored in LU)
template <int BS>
__device__ static void LU_fwd_bwd_subs_site_PRECISION(
    cplx_PRECISION *x, const cplx_PRECISION *b, const cplx_PRECISION *LU)
{
  for (int n = 0; n < 2; n++) {
    // forward substitution with L
    for (int i = 0; i < BS; i++) {
      cplx_PRECISION xi = b[i];
      for (int j = 0; j < i; j++)
        xi -= LU[i*BS+j] * x[j];
      x[i] = xi;
    }
    // backward substitution with U
    for (int i = BS-1; i >= 0; i--) {
      cplx_PRECISION xi = x[i];
      for (int j = i+1; j < BS; j++)
        xi -= LU[i*BS+j] * x[j];
      x[i] = xi / LU[i*(BS+1)];
    }
    x += BS; b += BS; LU += BS*BS;
  }
}

// One thread per lattice site
template <int BS>
__global__ void LU_fwd_bwd_subs_kernel_PRECISION(
    const int n_sites, cplx_PRECISION *eta,
    const cplx_PRECISION *phi, const cplx_PRECISION *clover)
{
  int site = blockIdx.x * blockDim.x + threadIdx.x;
  if (site < n_sites)
    LU_fwd_bwd_subs_site_PRECISION<BS>(eta    + 2*BS*site,
                                       phi    + 2*BS*site,
                                       clover + 2*BS*BS*site);
}

extern "C"
void cuda_LU_perform_fwd_bwd_subs_PRECISION(
    const int n_sites, const int bs, const size_t elem_bytes,
    void *leta, void *lphi, void *clover)
{
  if (n_sites <= 0) return;

  if (bs != 6 && bs != 12) {
    printf("Error: cuda_LU_perform_fwd_bwd_subs: unsupported block size %d\n", bs);
    return;
  }

  // Sanity check: host complex type must match device complex type
  if (elem_bytes != sizeof(cplx_PRECISION))
    printf("Error: element size mismatch (host %zu, device %zu)\n",
           elem_bytes, sizeof(cplx_PRECISION));

  const size_t vec_bytes    = (size_t)n_sites * 2*bs    * sizeof(cplx_PRECISION);
  const size_t clover_bytes = (size_t)n_sites * 2*bs*bs * sizeof(cplx_PRECISION);

  cplx_PRECISION *d_eta = NULL, *d_phi = NULL, *d_clover = NULL;

  // Allocate device memory
  CK(cudaMalloc(&d_eta,    vec_bytes));
  CK(cudaMalloc(&d_phi,    vec_bytes));
  CK(cudaMalloc(&d_clover, clover_bytes));

  // Copy inputs host -> device (eta is output only, no need to copy it in)
  CK(cudaMemcpy(d_phi,    lphi,   vec_bytes,    cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_clover, clover, clover_bytes, cudaMemcpyHostToDevice));

  // Launch: one thread per site
  int threadsPerBlock = 128;
  int blocksPerGrid   = (n_sites + threadsPerBlock - 1) / threadsPerBlock;
  if (bs == 6)
    LU_fwd_bwd_subs_kernel_PRECISION<6><<<blocksPerGrid, threadsPerBlock>>>(
        n_sites, d_eta, d_phi, d_clover);
  else
    LU_fwd_bwd_subs_kernel_PRECISION<12><<<blocksPerGrid, threadsPerBlock>>>(
        n_sites, d_eta, d_phi, d_clover);
  CK(cudaGetLastError());        // launch errors
  CK(cudaDeviceSynchronize());   // execution errors, wait for the kernel

  // Copy result device -> host
  CK(cudaMemcpy(leta, d_eta, vec_bytes, cudaMemcpyDeviceToHost));

  // Free device memory
  CK(cudaFree(d_eta));
  CK(cudaFree(d_phi));
  CK(cudaFree(d_clover));
}
