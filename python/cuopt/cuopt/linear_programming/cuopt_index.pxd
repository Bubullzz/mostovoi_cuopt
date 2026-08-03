# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, int64_t

IF CUOPT_INDEX_64BIT:
    ctypedef int64_t cuopt_int_t
ELSE:
    ctypedef int32_t cuopt_int_t
