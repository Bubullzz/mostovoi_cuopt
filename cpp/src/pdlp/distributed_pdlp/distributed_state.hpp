/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <pdlp/distributed_pdlp/nccl_helpers.hpp>
#include <pdlp/distributed_pdlp/rank_data.hpp>
#include <pdlp/pdhg.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

// Stream handle: use cuda::stream_ref (CCCL) directly, matching the rest of
// cuopt (shard.hpp, pdhg.hpp, barrier/*). The rmm::cuda_stream_view alias has
// been dropped in the rmm version pinned by this cuopt dev env.
#include <cuda/stream>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cusparse.h>

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
//   - global problem sizes (the local problem only knows owned + halo sizes)
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
// Besides the data, this class holds the MP versions of the communication primitives that
// used to live on multi_gpu_engine_t (halo exchange, allreduce, dot / l2 norm, distributed
// SpMV; defined in distributed_state.cu). In MP every rank IS one shard, so there is no
// master / shard fan-out: each primitive simply runs on this rank and uses `nccl_comm`.
// All ranks must call the same primitives in the same order (they are collectives).
// The higher-level algorithms built on top of them (scaling, power iteration, ...) are
// free functions in mp_algorithms.hpp, so this header does not depend on pdlp_solver_t.
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
                      i_t n_global_vars_,
                      i_t n_global_cstrs_,
                      nccl_comm_unique_ptr_t&& nccl_comm_,
                      rank_data_t<i_t, f_t>&& rank_data_,
                      cuda::stream_ref stream)
    : world_size(world_size_),
      rank(rank_),
      device_id(device_id_),
      n_global_vars(n_global_vars_),
      n_global_cstrs(n_global_cstrs_),
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

  // GLOBAL problem sizes. The rank's local pdlp problem only knows its
  // owned + halo sizes (rank_data.total_*_size).
  i_t n_global_vars;
  i_t n_global_cstrs;

  nccl_comm_unique_ptr_t nccl_comm;
  rank_data_t<i_t, f_t> rank_data;

  // Per-peer halo exchange device buffers. Sizes come straight from
  // rank_data.{var,cstr}_send_per_peer[p].size() at ctor time.
  std::vector<rmm::device_uvector<i_t>> var_send_indices_d;
  std::vector<rmm::device_uvector<f_t>> var_send_buf_d;
  std::vector<rmm::device_uvector<i_t>> cstr_send_indices_d;
  std::vector<rmm::device_uvector<f_t>> cstr_send_buf_d;

  // Non-owning bundle of per-axis halo-exchange metadata, indexed by peer.
  // Same idea as pdlp_shard_t::halo_axis_t: one struct for both variables and
  // constraints so the exchange logic is written once.
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

  // ===========================================================================
  // Primitives (defined in distributed_state.cu)
  // ===========================================================================

  // Halo exchange: refresh the halo tail of `x` from the owners of those entries.
  // `x` layout is [owned | halo]; the owned prefix is the source of truth and is
  // never modified. `_var` is for primal-shaped vectors, `_cstr` for dual-shaped.
  void halo_exchange_var(rmm::device_uvector<f_t>& x, cuda::stream_ref stream);
  void halo_exchange_cstr(rmm::device_uvector<f_t>& x, cuda::stream_ref stream);

  // In-place NCCL sum-allreduce of one device scalar across all ranks.
  void allreduce_sum_inplace(raft::device_scalar_view<f_t> scalar, cuda::stream_ref stream);

  // Global dot product / L2 norm over the OWNED slices (pass spans of the owned
  // prefix only, so halo entries are not double counted). Local cublas dot ->
  // allreduce -> (sqrt for the norm). The result lands in `out` on every rank.
  void distributed_dot(raft::device_span<f_t> a,
                       raft::device_span<f_t> b,
                       raft::device_scalar_view<f_t> out,
                       raft::handle_t const& handle);
  void distributed_dot(raft::device_span<f_t> a,
                       raft::device_span<f_t> b,
                       rmm::device_scalar<f_t>& out,
                       raft::handle_t const& handle);
  void distributed_l2_norm(raft::device_span<f_t> in,
                           raft::device_scalar_view<f_t> out,
                           raft::handle_t const& handle);
  void distributed_l2_norm(raft::device_span<f_t> in,
                           rmm::device_scalar<f_t>& out,
                           raft::handle_t const& handle);

  // out = A^T @ in. `in` is dual-shaped: halo-exchange it first, then run the local SpMV.
  void distributed_spmv_At(pdhg_solver_t<i_t, f_t>& pdhg,
                           rmm::device_uvector<f_t>& in_buf,
                           cusparseDnVecDescr_t in_desc,
                           cusparseDnVecDescr_t out_desc,
                           cuda::stream_ref stream);
  // out = A @ in. `in` is primal-shaped: halo-exchange it first, then run the local SpMV.
  void distributed_spmv_A(pdhg_solver_t<i_t, f_t>& pdhg,
                          rmm::device_uvector<f_t>& in_buf,
                          cusparseDnVecDescr_t in_desc,
                          cusparseDnVecDescr_t out_desc,
                          cuda::stream_ref stream);

  // MP counterparts of pdhg_solver_t::compute_A_x() / compute_At_y(): halo
  // exchange of the pdhg's own input vector, then the pdhg's own SpMV op.
  void distributed_compute_A_x(pdhg_solver_t<i_t, f_t>& pdhg, cuda::stream_ref stream);
  void distributed_compute_At_y(pdhg_solver_t<i_t, f_t>& pdhg, cuda::stream_ref stream);

 private:
  // Shared body of halo_exchange_{var,cstr}: gather what each peer needs into
  // staging, then matched ncclSend/ncclRecv in one NCCL group.
  void halo_exchange_impl(rmm::device_uvector<f_t>& x,
                          halo_axis_t const& axis,
                          cuda::stream_ref stream);

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
