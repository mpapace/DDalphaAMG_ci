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

#ifndef CUDA_SCHWARZ_GENERIC_H
#define CUDA_SCHWARZ_GENERIC_H

#include <cuda_runtime.h>
#include <stddef.h>

void cuda_red_black_schwarz_PRECISION(
    const int cycles,
    const int block_thread_start[8], const int block_thread_end[8],
    const int res, const int init_res, const int start, const int end,
    const int method, const int num_blocks, const int *block_list,
    const size_t elem_bytes,
    void *Dphi, void *eta, void *r, void *x, void *latest_iter);

#endif
