/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#if defined(__CUDACC__)

// CUDA only provides 64 bit atomics for `unsigned long long int`, so builds with
// cuopt_int_t == int64_t need these signed wrappers. They are in the global
// namespace so that unqualified calls still consider the CUDA overloads.

__device__ inline int64_t atomicAdd(int64_t* address, int64_t val)
{
  return static_cast<int64_t>(::atomicAdd(reinterpret_cast<unsigned long long int*>(address),
                                          static_cast<unsigned long long int>(val)));
}

__device__ inline int64_t atomicSub(int64_t* address, int64_t val)
{
  return static_cast<int64_t>(::atomicAdd(reinterpret_cast<unsigned long long int*>(address),
                                          0ULL - static_cast<unsigned long long int>(val)));
}

__device__ inline int64_t atomicExch(int64_t* address, int64_t val)
{
  return static_cast<int64_t>(::atomicExch(reinterpret_cast<unsigned long long int*>(address),
                                           static_cast<unsigned long long int>(val)));
}

__device__ inline int64_t atomicCAS(int64_t* address, int64_t compare, int64_t val)
{
  return static_cast<int64_t>(::atomicCAS(reinterpret_cast<unsigned long long int*>(address),
                                          static_cast<unsigned long long int>(compare),
                                          static_cast<unsigned long long int>(val)));
}

// Signed min/max are native for `long long` but not for `long`.
__device__ inline int64_t atomicMin(int64_t* address, int64_t val)
{
  return static_cast<int64_t>(
    ::atomicMin(reinterpret_cast<long long*>(address), static_cast<long long>(val)));
}

__device__ inline int64_t atomicMax(int64_t* address, int64_t val)
{
  return static_cast<int64_t>(
    ::atomicMax(reinterpret_cast<long long*>(address), static_cast<long long>(val)));
}

#endif
