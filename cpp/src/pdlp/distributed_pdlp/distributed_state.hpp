/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <pdlp/distributed_pdlp/nccl_helpers.hpp>
#include <pdlp/distributed_pdlp/rank_data.hpp>

#include <raft/core/copy.hpp>

// Stream handle: use cuda::stream_ref (CCCL) directly, matching the rest of
// cuopt (shard.hpp, pdhg.hpp, barrier/*). The rmm::cuda_stream_view alias has
// been dropped in the rmm version pinned by this cuopt dev env.
#include <cuda/stream>

#include <rmm/device_uvector.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::pdlp {

// Bundle of per-rank distributed context that a pdlp_solver_t needs in
// multi-process mode. Owned long-term by pdlp_solver_t (as
// std::optional<distributed_state_t<i_t,f_t>>): populated when the solver
// was launched under MPI, nullopt in plain single-process single-GPU mode.
//
// Holds everything that only exists in distributed mode:
//   - rank identity (world_size, rank, device_id)
//   - the NCCL comm (lifetime of the solve; destroyed on the right device)
//   - this rank's slice of the problem + halo plan (rank_data)
//   - the per-peer halo exchange device buffers derived from rank_data
//     (var_send_indices_d / var_send_buf_d and the cstr equivalents)
//
// Halo buffer layout, indexed by peer p:
//   var_send_indices_d[p]  : LOCAL primal indices to gather and ncclSend to p
//   var_send_buf_d[p]      : f_t staging for outgoing primal values (same length)
//   cstr_send_indices_d[p] : LOCAL dual indices to gather and ncclSend to p
//   cstr_send_buf_d[p]     : f_t staging for outgoing dual values (same length)
// The self-peer slot (p == rank) is kept but empty (size 0).
//
// Move-only (nccl_comm_unique_ptr_t + rmm::device_uvector are move-only); no
// default ctor because there is no meaningful "default distributed state"
// (rank_data_t itself requires knowing nb_parts up front, and the device
// buffers need a stream to be allocated on).
template <typename i_t, typename f_t>
struct distributed_state_t {
  // Ctor: moves ownership of nccl_comm + rank_data in, then allocates halo
  // staging on `stream` and uploads the send-index plan from rank_data.
  // Synchronizes `stream` at the end so the returned object is immediately
  // usable from any stream (matches pdlp_shard_t's existing behavior).
  distributed_state_t(int world_size_,
                      int rank_,
                      int device_id_,
                      nccl_comm_unique_ptr_t&& nccl_comm_,
                      rank_data_t<i_t, f_t>&& rank_data_,
                      cuda::stream_ref stream)
    : world_size(world_size_),
      rank(rank_),
      device_id(device_id_),
      nccl_comm(std::move(nccl_comm_)),
      rank_data(std::move(rank_data_))
  {
    build_send_plan(rank_data.var_send_per_peer, var_send_indices_d, var_send_buf_d, stream);
    build_send_plan(rank_data.cstr_send_per_peer, cstr_send_indices_d, cstr_send_buf_d, stream);
    // cuda::stream_ref: modern CCCL API is sync() (not synchronize(), which
    // was the old rmm::cuda_stream_view name and has been dropped).
    stream.sync();
  }

  distributed_state_t(const distributed_state_t&)                = delete;
  distributed_state_t& operator=(const distributed_state_t&)     = delete;
  distributed_state_t(distributed_state_t&&) noexcept            = default;
  distributed_state_t& operator=(distributed_state_t&&) noexcept = default;

  int world_size;
  int rank;
  int device_id;
  nccl_comm_unique_ptr_t nccl_comm;
  rank_data_t<i_t, f_t> rank_data;

  // Per-peer halo exchange device buffers. Sizes come straight from
  // rank_data.{var,cstr}_send_per_peer[p].size() at ctor time.
  std::vector<rmm::device_uvector<i_t>> var_send_indices_d;
  std::vector<rmm::device_uvector<f_t>> var_send_buf_d;
  std::vector<rmm::device_uvector<i_t>> cstr_send_indices_d;
  std::vector<rmm::device_uvector<f_t>> cstr_send_buf_d;

 private:
  // Allocate one (indices, staging) pair per peer and H->D copy the indices.
  // send_per_peer[p] is already in LOCAL index space (produced by the
  // partition slicing); no global->local translation needed here.
  static void build_send_plan(std::vector<std::vector<i_t>> const& send_per_peer,
                              std::vector<rmm::device_uvector<i_t>>& indices_d,
                              std::vector<rmm::device_uvector<f_t>>& buf_d,
                              cuda::stream_ref stream)
  {
    const std::size_t n_peers = send_per_peer.size();
    indices_d.reserve(n_peers);
    buf_d.reserve(n_peers);
    for (auto const& send_to_peer : send_per_peer) {
      indices_d.emplace_back(send_to_peer.size(), stream);
      buf_d.emplace_back(send_to_peer.size(), stream);
      if (!send_to_peer.empty()) {
        raft::copy(indices_d.back().data(), send_to_peer.data(), send_to_peer.size(), stream);
      }
    }
  }
};

}  // namespace cuopt::mathematical_optimization::pdlp
