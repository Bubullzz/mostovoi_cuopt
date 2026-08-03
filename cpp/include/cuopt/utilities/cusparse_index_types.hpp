/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/constants.h>

#include <cusparse.h>
#include <type_traits>

namespace cuopt::utilities {

template <typename i_t>
inline constexpr cusparseIndexType_t cusparse_index_type_v =
  sizeof(i_t) == 8 ? CUSPARSE_INDEX_64I : CUSPARSE_INDEX_32I;

template <typename i_t>
inline cusparseIndexType_t cusparse_index_type()
{
  static_assert(std::is_integral_v<i_t>, "cusparse index type requires an integral i_t");
  return cusparse_index_type_v<i_t>;
}

}  // namespace cuopt::utilities
