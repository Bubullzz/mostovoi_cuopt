/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuopt/error.hpp>
#include <utilities/logger.hpp>

#include <mpi.h>

#include <memory>

namespace cuopt::mathematical_optimization::pdlp {

enum mpi_status_t {
  MPI_UNUSED = 0,  // not under mpirun / srun; one-process, one-GPU, no MPI calls
  MPI_MASTER = 1,  // under MPI, rank 0
  MPI_SHARD  = 2,  // under MPI, rank > 0
};

class mp_context_t {
 public:
  // Does MPI init (if launched under MPI) and pins the CUDA device.
  mp_context_t(int& argc, char**& argv);

  // Rule of zero: special member functions are compiler-generated. The
  // unique_ptr<mpi_session_t> below carries the "did I call MPI_Init?" flag,
  // and its own move/destroy semantics guarantee MPI_Finalize fires exactly
  // once, by the instance that currently owns the session.
  mp_context_t(const mp_context_t&)                = delete;
  mp_context_t& operator=(const mp_context_t&)     = delete;
  mp_context_t(mp_context_t&&) noexcept            = default;
  mp_context_t& operator=(mp_context_t&&) noexcept = default;

  bool is_master() const noexcept { return status == MPI_MASTER; }
  bool is_mpi() const noexcept { return status != MPI_UNUSED; }
  bool is_root() const noexcept { return is_master() || !is_mpi(); }

  // Public read-only state (kept as fields for ergonomic access at call sites).
  MPI_Comm world_comm{MPI_COMM_NULL};
  int world_size{1};
  int rank{0};
  int device_id{0};  // == rank % cudaGetDeviceCount()
  mpi_status_t status{MPI_UNUSED};

 private:
  // Opaque ownership token. Non-null exactly on the instance that successfully
  // called MPI_Init_thread; its destructor calls MPI_Finalize. On move, the
  // unique_ptr transfers ownership and nulls the source so the moved-from
  // dtor is a no-op.
  struct mpi_session_t {
    ~mpi_session_t() noexcept;
  };
  std::unique_ptr<mpi_session_t> session_;
};

// Wraps an MPI call and throws cuopt::logic_error (RuntimeError) on anything
// other than MPI_SUCCESS. Mirrors CUOPT_NCCL_TRY.
#define CUOPT_MPI_TRY(call)                                           \
  do {                                                                \
    int const _cuopt_mpi_status = (call);                             \
    if (_cuopt_mpi_status != MPI_SUCCESS) {                           \
      char _cuopt_mpi_errstr[MPI_MAX_ERROR_STRING] = {0};             \
      int _cuopt_mpi_errlen                        = 0;               \
      MPI_Error_string(_cuopt_mpi_status, _cuopt_mpi_errstr, &_cuopt_mpi_errlen); \
      ::cuopt::cuopt_expects(false,                                   \
                             ::cuopt::error_type_t::RuntimeError,     \
                             "MPI error at %s:%d: %s",                \
                             __FILE__,                                \
                             __LINE__,                                \
                             _cuopt_mpi_errstr);                      \
    }                                                                 \
  } while (0)

}  // namespace cuopt::mathematical_optimization::pdlp
