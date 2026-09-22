/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstdint>

namespace cuopt {
namespace mathematical_optimization {

#if defined(PDLP_INDEX_64BIT)
using index_t = std::int64_t;
#else
using index_t = std::int32_t;
#endif

}  // namespace mathematical_optimization
}  // namespace cuopt
