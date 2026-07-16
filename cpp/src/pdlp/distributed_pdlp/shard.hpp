/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <pdlp/distributed_pdlp/nccl_helpers.hpp>
#include <pdlp/distributed_pdlp/rank_data.hpp>
#include <utilities/event_handler.cuh>
#include <utilities/macros.cuh>

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/optimization_problem.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>
#include <mip_heuristics/problem/problem.cuh>

#include <raft/core/device_setter.hpp>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

#include <nccl.h>

#include <memory>
#include <optional>
#include <vector>

namespace cuopt::mathematical_optimization::pdlp {

// Forward-declare to break the cyclic include with pdlp.cuh
// (pdlp.cuh -> multi_gpu_engine.hpp -> shard.hpp -> pdlp.cuh).
// Definitions of out-of-line members live in shard.cu, which includes pdlp.cuh.
template <typename i_t, typename f_t>
class pdlp_solver_t;

// RAII deleter for ncclComm_t; sets the right device before destroy.
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

template <typename i_t, typename f_t>
struct pdlp_shard_t {
  // Out-of-line (in shard.cu) because pdlp_solver_t is incomplete here.
  ~pdlp_shard_t();

  // sub worker for distributed pdlp. Owns its own view on scaled problem and unscaled problem
  // Owns necessary multi-gpu data (rank_data, device_id, nccl_comm)
  pdlp_shard_t(int device_id,
               rank_data_t<i_t, f_t>&& rd,
               nccl_comm_unique_ptr_t&& comm,
               io::mps_data_model_t<i_t, f_t> const& mps,
               pdlp_solver_settings_t<i_t, f_t> const& settings);

  pdlp_shard_t(const pdlp_shard_t&)            = delete;
  pdlp_shard_t& operator=(const pdlp_shard_t&) = delete;

  int device_id{-1};
  rmm::cuda_stream stream;
  // Secondary stream used exclusively by multi_gpu_engine_t's
  // distributed_spmv_A/At to run the halo gather+NCCL exchange in parallel
  // with the own-half SpMV on `stream`. Independent from `stream`; sync is
  // done through the spmv_{input,halo}_ready_event pair below.
  rmm::cuda_stream comm_stream;
  // Reusable events for the SpMV compute/comm overlap.
  //   spmv_input_ready_event: recorded on `stream` at overlap-SpMV entry;
  //                           `comm_stream` waits on it before gathering (so
  //                           the caller-produced input is visible).
  //   spmv_halo_ready_event : recorded on `comm_stream` after ncclRecv;
  //                           `stream` waits on it before spmv_halo_into (so
  //                           the halo tail of the input vec is visible).
  // Both are default-created (auto flag == cudaEventDefault) in shard.cu; no
  // per-call allocation.
  std::unique_ptr<cuopt::event_handler_t> spmv_input_ready_event;
  std::unique_ptr<cuopt::event_handler_t> spmv_halo_ready_event;
  raft::handle_t handle;
  nccl_comm_unique_ptr_t comm;
  rank_data_t<i_t, f_t> rank_data;
  optimization_problem_t<i_t, f_t> opt_problem;
  std::optional<mip::problem_t<i_t, f_t>> sub_problem;
  std::unique_ptr<pdlp_solver_t<i_t, f_t>> sub_pdlp;

  // Device-side column-ownership split of the local A / A_T matrices, used by
  // multi_gpu_engine_t::distributed_spmv_A/At for the comm/comp overlap:
  //   own :  cols in [0, owned_*_size)  -- input already valid, run pre-exchange
  //   halo:  cols in [owned_*_size, total_*_size) -- run post-exchange with beta=1
  // Populated in the ctor from rank_data.h_A{,_t}_{own,halo}_*. See rank_data_t
  // comment for the split invariant.
  struct local_csr_t {
    rmm::device_uvector<i_t> row_offsets;
    rmm::device_uvector<i_t> col_indices;
    rmm::device_uvector<f_t> values;

    explicit local_csr_t(rmm::cuda_stream_view s)
      : row_offsets(0, s), col_indices(0, s), values(0, s)
    {
    }
  };
  struct split_matrix_t {
    local_csr_t own;
    local_csr_t halo;

    explicit split_matrix_t(rmm::cuda_stream_view s) : own(s), halo(s) {}
  };
  split_matrix_t A_split;
  split_matrix_t A_T_split;

  // Refresh A_split.{own,halo}.values and A_T_split.{own,halo}.values from
  // the current sub_problem A / A_T on device. Structure (row_offsets and
  // col_indices) is fixed at construction and doesn't change with scaling,
  // so only value arrays need to be re-derived. MUST be called after any
  // in-place mutation of sub_problem's A / A_T values -- notably after
  // multi_gpu_engine_t::distributed_scaling's apply_cummulative_scaling_to_
  // problem, which scales A and A_T on device while leaving our split
  // copies stale (this is the entire bug fixed by wiring this method into
  // distributed_scaling's step 2).
  void sync_split_values_from_parent();

  // var_send_indices_d[peer] : local indices into primal vector to gather and ncclSend
  // var_send_buf_d    [peer] : staging buffer for outgoing variable values
  // cstr_send_indices_d/cstr_send_buf_d : same, for dual vector
  std::vector<rmm::device_uvector<i_t>> var_send_indices_d;
  std::vector<rmm::device_uvector<f_t>> var_send_buf_d;
  std::vector<rmm::device_uvector<i_t>> cstr_send_indices_d;
  std::vector<rmm::device_uvector<f_t>> cstr_send_buf_d;

  // Non-owning bundle of per-axis halo-exchange metadata, indexed by peer.
  // Consumed by multi_gpu_engine_t::halo_exchange_bufs_impl
  struct halo_axis_t {
    std::vector<rmm::device_uvector<i_t>>& send_indices;  // [peer]
    std::vector<rmm::device_uvector<f_t>>& send_buf;      // [peer]
    i_t owned_size;
    std::vector<i_t> const& recv_offsets;  // [peer]
    std::vector<i_t> const& recv_counts;   // [peer]
  };
  halo_axis_t var_halo_axis()
  {
    return {var_send_indices_d,
            var_send_buf_d,
            rank_data.owned_var_size,
            rank_data.var_recv_offsets,
            rank_data.var_recv_counts};
  }
  halo_axis_t cstr_halo_axis()
  {
    return {cstr_send_indices_d,
            cstr_send_buf_d,
            rank_data.owned_cstr_size,
            rank_data.cstr_recv_offsets,
            rank_data.cstr_recv_counts};
  }
};

}  // namespace cuopt::mathematical_optimization::pdlp
