/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuopt/error.hpp>
#include <utilities/logger.hpp>
#include <utilities/macros.cuh>

#include <raft/core/device_setter.hpp>

#include <nccl.h>

#include <memory>

namespace cuopt::mathematical_optimization::pdlp {

// Wraps a NCCL call and throws cuopt::logic_error (RuntimeError) on any non-
// ncclSuccess return code.
#define CUOPT_NCCL_TRY(call)                                          \
  do {                                                                \
    ::ncclResult_t const _cuopt_nccl_status = (call);                 \
    ::cuopt::cuopt_expects(_cuopt_nccl_status == ncclSuccess,         \
                           ::cuopt::error_type_t::RuntimeError,       \
                           "NCCL error at %s:%d: %s",                 \
                           __FILE__,                                  \
                           __LINE__,                                  \
                           ::ncclGetErrorString(_cuopt_nccl_status)); \
  } while (0)

// Non-throwing variant: logs at ERROR level on failure and swallows the error.
// Intended for noexcept contexts (destructors, unique_ptr deleters, teardown)
// where throwing would call std::terminate. Mirrors RAFT_CUDA_TRY_NO_THROW.
#define CUOPT_NCCL_TRY_NO_THROW(call)                                                             \
  do {                                                                                            \
    ::ncclResult_t const _cuopt_nccl_status = (call);                                             \
    if (_cuopt_nccl_status != ncclSuccess) {                                                      \
      CUOPT_LOG_ERROR(                                                                            \
        "NCCL error at %s:%d: %s", __FILE__, __LINE__, ::ncclGetErrorString(_cuopt_nccl_status)); \
    }                                                                                             \
  } while (0)

// RAII deleter for ncclComm_t; pins the right device before ncclCommDestroy
// so the destroy runs on the device the comm was initialized on. Noexcept
// swallow-and-log on failure so it's safe inside unique_ptr destruction.
struct nccl_comm_deleter_t {
  int device_id{-1};
  void operator()(ncclComm* comm) const noexcept
  {
    if (comm == nullptr) return;
    cuopt_assert(device_id >= 0, "nccl_comm_deleter_t: device_id not set");
    raft::device_setter guard(device_id);
    CUOPT_NCCL_TRY_NO_THROW(ncclCommDestroy(comm));
  }
};
using nccl_comm_unique_ptr_t = std::unique_ptr<ncclComm, nccl_comm_deleter_t>;

}  // namespace cuopt::mathematical_optimization::pdlp
