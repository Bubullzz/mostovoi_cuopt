# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
# SPDX-License-Identifier: Apache-2.0

from .data_model cimport *

import warnings

import numpy as np

from libcpp.memory cimport unique_ptr
from ..cuopt_index cimport cuopt_int_t


cdef class DataModel:
    cdef unique_ptr[data_model_view_t[cuopt_int_t, double]] c_data_model_view

    cdef void _set_cpp_quadratic_constraints(
        self, data_model_view_t[cuopt_int_t, double]* c_data_model_view
    )
