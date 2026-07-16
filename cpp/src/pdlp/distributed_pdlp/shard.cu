/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pdlp/distributed_pdlp/shard.hpp>
#include <pdlp/pdlp.cuh>
#include <pdlp/utils.cuh>

#include <utilities/copy_helpers.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/device_setter.hpp>

#include <cassert>
#include <limits>

namespace cuopt::mathematical_optimization::pdlp {

// This must be done in .cu file because the pdlp_solver_t is not already complete in the hpp file
// This is caused by the problematic cyclic include of pdlp_solver_t
template <typename i_t, typename f_t>
pdlp_shard_t<i_t, f_t>::~pdlp_shard_t() = default;

template <typename i_t, typename f_t>
pdlp_shard_t<i_t, f_t>::pdlp_shard_t(int device_id,
                                     rank_data_t<i_t, f_t>&& rd,
                                     nccl_comm_unique_ptr_t&& comm,
                                     io::mps_data_model_t<i_t, f_t> const& mps,
                                     pdlp_solver_settings_t<i_t, f_t> const& settings)
  : device_id(device_id),
    stream(),
    handle(stream.view()),
    comm(std::move(comm)),
    rank_data(std::move(rd)),
    opt_problem(&handle),
    sub_problem(std::nullopt),
    sub_pdlp(nullptr),
    A_split(stream.view()),
    A_T_split(stream.view())
{
  assert(raft::device_setter::get_current_device() == device_id &&
         "Right device must be set before building the shard");

  // ---- 0. Problem-level scalars, taken straight from the global mps. ----
  // Objective coefficients / offset / scaling factor are passed through
  // unchanged; the max -> min conversion (negating all three) happens once,
  // in mip::problem_t's constructor via convert_to_maximization_problem
  const bool maximize                = mps.get_sense();
  const f_t objective_offset         = mps.get_objective_offset();
  const f_t objective_scaling_factor = mps.get_objective_scaling_factor();

  // Global (unpartitioned) host arrays, indexed by global var / cstr id.
  const std::vector<f_t>& g_obj        = mps.get_objective_coefficients();
  const std::vector<f_t>& g_var_lower  = mps.get_variable_lower_bounds();
  const std::vector<f_t>& g_var_upper  = mps.get_variable_upper_bounds();
  const std::vector<f_t>& g_cstr_lower = mps.get_constraint_lower_bounds();
  const std::vector<f_t>& g_cstr_upper = mps.get_constraint_upper_bounds();

  // ---- 1. Gather per-shard host slices using rank_data's index maps. ----
  // All vectors are sized to TOTAL (owned + halo). Owned slots get real
  // values; halo slots keep defaults because they should not be accessed before getting filled.
  std::vector<f_t> h_obj(rank_data.total_var_size, f_t{0});
  std::vector<f_t> h_var_lower(rank_data.total_var_size, -std::numeric_limits<f_t>::infinity());
  std::vector<f_t> h_var_upper(rank_data.total_var_size, std::numeric_limits<f_t>::infinity());
  std::vector<f_t> h_cstr_lower(rank_data.total_cstr_size, -std::numeric_limits<f_t>::infinity());
  std::vector<f_t> h_cstr_upper(rank_data.total_cstr_size, std::numeric_limits<f_t>::infinity());

  for (i_t i = 0; i < rank_data.owned_var_size; ++i) {
    const auto g   = rank_data.local_to_global_var[i];
    h_obj[i]       = g_obj[g];
    h_var_lower[i] = g_var_lower[g];
    h_var_upper[i] = g_var_upper[g];
  }
  for (i_t i = 0; i < rank_data.owned_cstr_size; ++i) {
    const auto g    = rank_data.local_to_global_cstr[i];
    h_cstr_lower[i] = g_cstr_lower[g];
    h_cstr_upper[i] = g_cstr_upper[g];
  }

  // ---- 2. Populate opt_problem (constructed in init list) on this shard's device. ----
  opt_problem.set_csr_constraint_matrix(rank_data.h_A_values.data(),
                                        static_cast<i_t>(rank_data.h_A_values.size()),
                                        rank_data.h_A_col_indices.data(),
                                        static_cast<i_t>(rank_data.h_A_col_indices.size()),
                                        rank_data.h_A_row_offsets.data(),
                                        static_cast<i_t>(rank_data.h_A_row_offsets.size()));

  // Primal axis: TOTAL (owned + halo)
  opt_problem.set_objective_coefficients(h_obj.data(), rank_data.total_var_size);
  opt_problem.set_variable_lower_bounds(h_var_lower.data(), rank_data.total_var_size);
  opt_problem.set_variable_upper_bounds(h_var_upper.data(), rank_data.total_var_size);

  // Dual axis: TOTAL (owned + halo)
  opt_problem.set_constraint_lower_bounds(h_cstr_lower.data(), rank_data.total_cstr_size);
  opt_problem.set_constraint_upper_bounds(h_cstr_upper.data(), rank_data.total_cstr_size);

  opt_problem.set_maximize(maximize);
  opt_problem.set_objective_offset(objective_offset);
  opt_problem.set_objective_scaling_factor(objective_scaling_factor);
  opt_problem.set_problem_category(problem_category_t::LP);

  // ---- 3. Build problem_t from opt_problem (UNSCALED). ----
  sub_problem.emplace(opt_problem);

  // ---- 4. Override reverse_* with the real local A_T from rank_data. ----
  // problem_t's ctor computes the transpose of the LOCAL A, which is wrong
  // in multi-GPU: A_local is owned_cstr x total_var, and A_t_local is the
  // pre-sliced owned_var x total_cstr matrix we built during partitioning.
  auto stream_view = handle.get_stream();
  sub_problem->reverse_offsets.resize(rank_data.h_A_t_row_offsets.size(), stream_view);
  sub_problem->reverse_constraints.resize(rank_data.h_A_t_col_indices.size(), stream_view);
  sub_problem->reverse_coefficients.resize(rank_data.h_A_t_values.size(), stream_view);
  raft::copy(sub_problem->reverse_offsets.data(),
             rank_data.h_A_t_row_offsets.data(),
             rank_data.h_A_t_row_offsets.size(),
             stream_view);
  raft::copy(sub_problem->reverse_constraints.data(),
             rank_data.h_A_t_col_indices.data(),
             rank_data.h_A_t_col_indices.size(),
             stream_view);
  raft::copy(sub_problem->reverse_coefficients.data(),
             rank_data.h_A_t_values.data(),
             rank_data.h_A_t_values.size(),
             stream_view);
  handle.sync_stream(stream_view);

  // ---- 5. Build sub_pdlp (single-GPU mode). ----
  // is_distributed_sub_pdlp=true has two effects in pdlp_solver_t's ctor:
  //   * skip the CSR/CSC transpose validity check -- A and A_T here are two
  //     independent local slices, not transposes (A has all owned rows and
  //     A_T has all owned columns).
  //   * skip local Ruiz / Pock-Chambolle inside initial_scaling_strategy_'s
  //     ctor -- distributed scaling (multi_gpu_engine_t::distributed_scaling)
  //     runs a cross-shard-coherent scaling later. Local per-shard scaling
  //     would be incoherent across shards.
  sub_pdlp = std::make_unique<pdlp_solver_t<i_t, f_t>>(
    *sub_problem, settings, /*is_legacy_batch_mode=*/false, /*is_distributed_sub_pdlp=*/true);

  // ---- 6. Build per-peer halo-exchange plans ----
  // For each peer p, we precompute:
  //   send_indices_d[p] : local indices to gather (uploaded from host send plan)
  //   send_buf_d[p]     : f_t staging buffer sized to match
  // Self-peer slot is present but empty (size 0).
  auto build_send_plan = [&](std::vector<std::vector<i_t>> const& send_per_peer,
                             std::vector<rmm::device_uvector<i_t>>& indices_d,
                             std::vector<rmm::device_uvector<f_t>>& buf_d) {
    const std::size_t n_peers = send_per_peer.size();
    indices_d.reserve(n_peers);
    buf_d.reserve(n_peers);
    for (auto const& send_to_peer : send_per_peer) {
      indices_d.emplace_back(send_to_peer.size(), stream_view);
      buf_d.emplace_back(send_to_peer.size(), stream_view);
      if (!send_to_peer.empty()) {
        raft::copy(indices_d.back().data(), send_to_peer.data(), send_to_peer.size(), stream_view);
      }
    }
  };
  build_send_plan(rank_data.var_send_per_peer, var_send_indices_d, var_send_buf_d);
  build_send_plan(rank_data.cstr_send_per_peer, cstr_send_indices_d, cstr_send_buf_d);

  // Reusable events for the SpMV compute/comm overlap driven by
  // multi_gpu_engine_t::distributed_spmv_A/At. Cheap (opaque handles) so we
  // create them upfront on the right device rather than on-demand.
  spmv_input_ready_event = std::make_unique<cuopt::event_handler_t>();
  spmv_halo_ready_event  = std::make_unique<cuopt::event_handler_t>();

  // ---- 7. Mirror the column-ownership split of the local host CSRs to device.
  // A_split.own / A_split.halo together carry the same nnz as opt_problem's A;
  // same for A_T_split relative to sub_problem->reverse_*. Consumed by
  // multi_gpu_engine_t::distributed_spmv_A/At via the cusparse descriptors
  // constructed in cusparse_view_t.
  auto upload_csr = [&](std::vector<i_t> const& h_row_offsets,
                        std::vector<i_t> const& h_col_indices,
                        std::vector<f_t> const& h_values,
                        typename pdlp_shard_t<i_t, f_t>::local_csr_t& d_csr) {
    d_csr.row_offsets.resize(h_row_offsets.size(), stream_view);
    d_csr.col_indices.resize(h_col_indices.size(), stream_view);
    d_csr.values.resize(h_values.size(), stream_view);
    if (!h_row_offsets.empty()) {
      raft::copy(
        d_csr.row_offsets.data(), h_row_offsets.data(), h_row_offsets.size(), stream_view);
    }
    if (!h_col_indices.empty()) {
      raft::copy(
        d_csr.col_indices.data(), h_col_indices.data(), h_col_indices.size(), stream_view);
    }
    if (!h_values.empty()) {
      raft::copy(d_csr.values.data(), h_values.data(), h_values.size(), stream_view);
    }
  };
  upload_csr(rank_data.h_A_own_row_offsets,
             rank_data.h_A_own_col_indices,
             rank_data.h_A_own_values,
             A_split.own);
  upload_csr(rank_data.h_A_halo_row_offsets,
             rank_data.h_A_halo_col_indices,
             rank_data.h_A_halo_values,
             A_split.halo);
  upload_csr(rank_data.h_A_t_own_row_offsets,
             rank_data.h_A_t_own_col_indices,
             rank_data.h_A_t_own_values,
             A_T_split.own);
  upload_csr(rank_data.h_A_t_halo_row_offsets,
             rank_data.h_A_t_halo_col_indices,
             rank_data.h_A_t_halo_values,
             A_T_split.halo);

  // ---- 8. Hand the split matrices to sub_pdlp's cusparse_view so cuSPARSE
  //         has descriptors + SpMV workspaces ready for the compute/comm
  //         overlap path in multi_gpu_engine_t::distributed_spmv_A/At.
  //         The parent A / A_T shape is (total_cstr x total_var) / (total_var
  //         x total_cstr); both halves share it (halo cols kept absolute).
  auto& cv = sub_pdlp->get_cusparse_view();
  cv.init_distributed_split(
    /* rows_A     */ rank_data.total_cstr_size,
    /* cols_A     */ rank_data.total_var_size,
    /* nnz_A_own  */ static_cast<int64_t>(A_split.own.values.size()),
    A_split.own.row_offsets.data(),
    A_split.own.col_indices.data(),
    A_split.own.values.data(),
    /* nnz_A_halo */ static_cast<int64_t>(A_split.halo.values.size()),
    A_split.halo.row_offsets.data(),
    A_split.halo.col_indices.data(),
    A_split.halo.values.data(),
    /* rows_A_T     */ rank_data.total_var_size,
    /* cols_A_T     */ rank_data.total_cstr_size,
    /* nnz_A_T_own  */ static_cast<int64_t>(A_T_split.own.values.size()),
    A_T_split.own.row_offsets.data(),
    A_T_split.own.col_indices.data(),
    A_T_split.own.values.data(),
    /* nnz_A_T_halo */ static_cast<int64_t>(A_T_split.halo.values.size()),
    A_T_split.halo.row_offsets.data(),
    A_T_split.halo.col_indices.data(),
    A_T_split.halo.values.data());

  handle.sync_stream(stream_view);
}

// Per-row kernel: walk parent CSR row and dispatch each nnz value into the
// own-half or halo-half output arrays, using the col-ownership boundary. The
// per-row destination base offsets (own_offsets / halo_offsets) were built at
// construction by the host splitter (see create_rank_data_from_parts), and the
// walking order here must match that splitter so the resulting {own,halo}.values
// stay aligned with the pre-built {own,halo}.col_indices. One thread per row is
// enough here: this runs once per solve (after the scaling pass), never in the
// inner PDHG loop.
template <typename i_t, typename f_t>
__global__ void split_csr_values_from_parent_kernel(i_t const* __restrict__ parent_offsets,
                                                    i_t const* __restrict__ parent_col_indices,
                                                    f_t const* __restrict__ parent_values,
                                                    i_t const* __restrict__ own_offsets,
                                                    i_t const* __restrict__ halo_offsets,
                                                    f_t* __restrict__ own_values,
                                                    f_t* __restrict__ halo_values,
                                                    i_t owned_col_boundary,
                                                    i_t n_rows)
{
  const i_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n_rows) return;

  const i_t p_begin = parent_offsets[row];
  const i_t p_end   = parent_offsets[row + 1];
  i_t o_idx         = own_offsets[row];
  i_t h_idx         = halo_offsets[row];
  for (i_t k = p_begin; k < p_end; ++k) {
    const i_t c = parent_col_indices[k];
    const f_t v = parent_values[k];
    if (c < owned_col_boundary) {
      own_values[o_idx++] = v;
    } else {
      halo_values[h_idx++] = v;
    }
  }
}

template <typename i_t, typename f_t>
void pdlp_shard_t<i_t, f_t>::sync_split_values_from_parent()
{
  auto stream_view = handle.get_stream();

  auto sync_one = [&](auto const& parent_offsets,
                      auto const& parent_col_indices,
                      auto const& parent_values,
                      typename pdlp_shard_t<i_t, f_t>::split_matrix_t& split,
                      i_t owned_col_boundary,
                      i_t n_rows) {
    if (n_rows == 0) return;
    constexpr i_t block = 128;
    const i_t grid      = (n_rows + block - 1) / block;
    split_csr_values_from_parent_kernel<i_t, f_t><<<grid, block, 0, stream_view.value()>>>(
      parent_offsets.data(),
      parent_col_indices.data(),
      parent_values.data(),
      split.own.row_offsets.data(),
      split.halo.row_offsets.data(),
      split.own.values.data(),
      split.halo.values.data(),
      owned_col_boundary,
      n_rows);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  };

  // A on this shard is (total_cstr x total_var); nnz in sub_problem->{offsets,
  // variables, coefficients}. A_T is (total_var x total_cstr) in
  // sub_problem->reverse_{offsets, constraints, coefficients} -- we overrode
  // it in the ctor to be the true per-shard local A_T (not the local A's
  // in-place transpose). See shard.cu step 4.
  sync_one(sub_problem->offsets,
           sub_problem->variables,
           sub_problem->coefficients,
           A_split,
           rank_data.owned_var_size,
           rank_data.total_cstr_size);
  sync_one(sub_problem->reverse_offsets,
           sub_problem->reverse_constraints,
           sub_problem->reverse_coefficients,
           A_T_split,
           rank_data.owned_cstr_size,
           rank_data.total_var_size);
}

template struct pdlp_shard_t<int, double>;
template struct pdlp_shard_t<int, float>;

}  // namespace cuopt::mathematical_optimization::pdlp
