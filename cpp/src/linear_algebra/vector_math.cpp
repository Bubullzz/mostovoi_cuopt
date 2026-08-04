/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/constants.h>
#include <barrier/pinned_host_allocator.hpp>

#include <linear_algebra/vector_math.hpp>
#include <math_optimization/types.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

namespace cuopt::mathematical_optimization {

template <typename i_t, typename f_t, typename Allocator>
f_t vector_norm2_squared(const std::vector<f_t, Allocator>& x)
{
  i_t n   = x.size();
  f_t sum = 0.0;
  for (i_t j = 0; j < n; ++j) {
    sum += x[j] * x[j];
  }
  return sum;
}

template <typename i_t, typename f_t, typename Allocator>
f_t vector_norm2(const std::vector<f_t, Allocator>& x)
{
  return std::sqrt(vector_norm2_squared<i_t, f_t, Allocator>(x));
}

template <typename i_t, typename f_t>
f_t vector_norm1(const std::vector<f_t>& x)
{
  i_t n   = x.size();
  f_t sum = 0.0;
  for (i_t j = 0; j < n; ++j) {
    sum += std::abs(x[j]);
  }
  return sum;
}

template <typename i_t, typename f_t>
f_t dot(const std::vector<f_t>& x, const std::vector<f_t>& y)
{
  assert(x.size() == y.size());
  const i_t n = x.size();
  f_t dot     = 0.0;
  for (i_t k = 0; k < n; ++k) {
    dot += x[k] * y[k];
  }
  return dot;
}

// Work = 3*min(nz_x, nz_y)
template <typename i_t, typename f_t>
f_t sparse_dot(
  i_t const* xind, f_t const* xval, i_t nx, i_t const* yind, i_t ny, f_t const* y_scatter_val)
{
  f_t dot = 0.0;
  for (i_t i = 0, j = 0; i < nx && j < ny;) {
    const i_t p = xind[i];
    const i_t q = yind[j];
    if (p == q) {
      dot += xval[i] * y_scatter_val[q];
      i++;
      j++;
    } else if (p < q) {
      i++;
    } else if (q < p) {
      j++;
    }
  }
  return dot;
}

template <typename i_t, typename f_t>
f_t sparse_dot(i_t* xind, f_t* xval, i_t nx, i_t* yind, f_t* yval, i_t ny)
{
  f_t dot = 0.0;
  for (i_t i = 0, j = 0; i < nx && j < ny;) {
    const i_t p = xind[i];
    const i_t q = yind[j];
    if (p == q) {
      dot += xval[i] * yval[j];
      i++;
      j++;
    } else if (p < q) {
      i++;
    } else if (q < p) {
      j++;
    }
  }
  return dot;
}

template <typename i_t, typename f_t>
f_t sparse_dot(const std::vector<i_t>& xind,
               const std::vector<f_t>& xval,
               const std::vector<i_t>& yind,
               const std::vector<f_t>& yval)
{
  const i_t nx = xind.size();
  const i_t ny = yind.size();
  f_t dot      = 0.0;
  for (i_t i = 0, j = 0; i < nx && j < ny;) {
    const i_t p = xind[i];
    const i_t q = yind[j];
    if (p == q) {
      dot += xval[i] * yval[j];
      i++;
      j++;
    } else if (p < q) {
      i++;
    } else if (q < p) {
      j++;
    }
  }
  return dot;
}

// Computes x = P*b or x=b(p) in MATLAB.
// Work is 3*n
template <typename i_t, typename f_t>
i_t permute_vector(const std::vector<i_t>& p, const std::vector<f_t>& b, std::vector<f_t>& x)
{
  i_t n = p.size();
  assert(x.size() == n);
  assert(b.size() == n);
  for (i_t k = 0; k < n; ++k) {
    x[k] = b[p[k]];
  }
  return 0;
}

// Computes x = P'*b or x(p) = b in MATLAB.
// Work is 3 * n
template <typename i_t, typename f_t>
i_t inverse_permute_vector(const std::vector<i_t>& p,
                           const std::vector<f_t>& b,
                           std::vector<f_t>& x)
{
  i_t n = p.size();
  assert(x.size() == n);
  assert(b.size() == n);
  for (i_t k = 0; k < n; ++k) {
    x[p[k]] = b[k];
  }
  return 0;
}

// Computes pinv from p. Or pinv(p) = 1:n in MATLAB
// Work is 2*n
template <typename i_t>
i_t inverse_permutation(const std::vector<i_t>& p, std::vector<i_t>& pinv)
{
  i_t n = p.size();
  if (pinv.size() != n) { pinv.resize(n); }
  for (i_t k = 0; k < n; ++k) {
    pinv[p[k]] = k;
  }
  return 0;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template double vector_norm_inf<cuopt_int_t, double, std::allocator<double>>(
  const std::vector<double>& x);

template double vector_norm2_squared<cuopt_int_t, double, std::allocator<double>>(
  const std::vector<double, std::allocator<double>>& x);

template double vector_norm2<cuopt_int_t, double, std::allocator<double>>(
  const std::vector<double, std::allocator<double>>& x);

template double vector_norm2_squared<cuopt_int_t, double, barrier::PinnedHostAllocator<double>>(
  const std::vector<double, barrier::PinnedHostAllocator<double>>&);
template double vector_norm2<cuopt_int_t, double, barrier::PinnedHostAllocator<double>>(
  const std::vector<double, barrier::PinnedHostAllocator<double>>&);

template double vector_norm1<cuopt_int_t, double>(const std::vector<double>& x);

template double dot<cuopt_int_t, double>(const std::vector<double>& x,
                                         const std::vector<double>& y);

template double sparse_dot<cuopt_int_t, double>(const std::vector<cuopt_int_t>& xind,
                                                const std::vector<double>& xval,
                                                const std::vector<cuopt_int_t>& yind,
                                                const std::vector<double>& yval);

template double sparse_dot<cuopt_int_t, double>(cuopt_int_t const* xind,
                                                double const* xval,
                                                cuopt_int_t nx,
                                                cuopt_int_t const* yind,
                                                cuopt_int_t ny,
                                                double const* y_scatter_val);

template double sparse_dot<cuopt_int_t, double>(
  cuopt_int_t* xind, double* xval, cuopt_int_t nx, cuopt_int_t* yind, double* yval, cuopt_int_t ny);

template cuopt_int_t permute_vector<cuopt_int_t, double>(const std::vector<cuopt_int_t>& p,
                                                         const std::vector<double>& b,
                                                         std::vector<double>& x);
template cuopt_int_t inverse_permute_vector<cuopt_int_t, double>(const std::vector<cuopt_int_t>& p,
                                                                 const std::vector<double>& b,
                                                                 std::vector<double>& x);
template cuopt_int_t inverse_permutation<cuopt_int_t>(const std::vector<cuopt_int_t>& p,
                                                      std::vector<cuopt_int_t>& pinv);

#endif

}  // namespace cuopt::mathematical_optimization
