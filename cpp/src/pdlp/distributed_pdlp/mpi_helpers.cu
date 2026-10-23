/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pdlp/distributed_pdlp/mpi_helpers.hpp>

#include <cuopt/error.hpp>
#include <utilities/logger.hpp>

#include <raft/core/device_setter.hpp>
#include <raft/core/error.hpp>  // RAFT_CUDA_TRY

#include <mpi.h>

#include <cstdlib>
#include <memory>

namespace cuopt::mathematical_optimization::pdlp {

namespace {

bool launched_under_mpi()
{
  return
    (std::getenv("OMPI_COMM_WORLD_RANK") && std::getenv("OMPI_COMM_WORLD_SIZE")) ||
    (std::getenv("PMI_RANK")             && std::getenv("PMI_SIZE"))             ||
    // PMIX_SIZE is not a standard PMIx env var; PMIX_RANK alone is the marker.
    std::getenv("PMIX_RANK")                                                      ||
    (std::getenv("SLURM_PROCID")         && std::getenv("SLURM_NTASKS"));
}

void pin_device(int device_id)
{
  const int device_count = raft::device_setter::get_device_count();
  cuopt_expects(device_count > 0,
                error_type_t::RuntimeError,
                "mp_context_t: no visible CUDA devices");
  cuopt_expects(device_id >= 0 && device_id < device_count,
                error_type_t::ValidationError,
                "mp_context_t: device_id=%d has no matching CUDA device "
                "(visible device count=%d).",
                device_id,
                device_count);
  RAFT_CUDA_TRY(cudaSetDevice(device_id));
}

}  // namespace

mp_context_t::mpi_session_t::~mpi_session_t() noexcept
{
  int const err = MPI_Finalize();
  if (err != MPI_SUCCESS) {
    CUOPT_LOG_ERROR("~mp_context_t: MPI_Finalize returned %d", err);
  }
}

mp_context_t::mp_context_t(int& argc, char**& argv)
{
  // Fast path: not under mpirun. Single-process, device 0, zero MPI cost.
  // Caller code that gates on `mp.rank == 0` keeps working unchanged.
  if (!launched_under_mpi()) {
    pin_device(0);  // device_id == 0 from default-init
    return;                 // status == MPI_UNUSED, session_ == nullptr
  }

  int provided = 0;
  CUOPT_MPI_TRY(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));
  // simple struct that calls MPI_Finalize() on destruction.
  session_ = std::make_unique<mpi_session_t>();

  world_comm = MPI_COMM_WORLD;
  CUOPT_MPI_TRY(MPI_Comm_size(world_comm, &world_size));
  CUOPT_MPI_TRY(MPI_Comm_rank(world_comm, &rank));

  device_id = rank % raft::device_setter::get_device_count();
  pin_device(device_id);

  status = (rank == 0) ? MPI_MASTER : MPI_SHARD;

  if (is_master()) {
    CUOPT_LOG_INFO(
      "mp_context_t: world_size=%d (rank 0 on device %d)", world_size, device_id);
  }
}

}  // namespace cuopt::mathematical_optimization::pdlp
