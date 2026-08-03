/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/constants.h>
#include <cuopt/mathematical_optimization/io/parser.hpp>

#include <string_view>

namespace cuopt::test {

inline cuopt::mathematical_optimization::io::mps_data_model_t<cuopt_int_t, double> parse_inline_lp(
  std::string_view lp_text)
{
  return cuopt::mathematical_optimization::io::read_lp_from_string<cuopt_int_t, double>(lp_text);
}

}  // namespace cuopt::test
