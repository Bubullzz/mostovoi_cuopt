# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cuda)
rapids_cuda_init_architectures(CUOPT)

project(
  cuopt-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(CUOPT_INDEX_64BIT "Use 64-bit indices (cuopt_int_t = int64_t) for >INT_MAX nonzeros" OFF)

if(CUOPT_INDEX_64BIT)
  add_compile_definitions(CUOPT_INDEX_64BIT=1 CUOPT_INSTANTIATE_INT64=1 CUOPT_INSTANTIATE_INT32=0)
endif()

find_package(cuopt "${RAPIDS_VERSION}")

include(rapids-cython-core)
rapids_cython_init()

add_subdirectory(cuopt)
add_subdirectory(cuopt/linear_programming/internals/)
