/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unordered_map>
#include <vector>

namespace cuopt::mathematical_optimization::pdlp {
// Pure data class representing most of the distributed data needed for mGPU operatiosn
template <typename i_t, typename f_t>
struct rank_data_t {
  rank_data_t(std::size_t nb_parts)
    : var_send_per_peer(nb_parts),
      cstr_send_per_peer(nb_parts),
      var_recv_counts(nb_parts, 0),
      var_recv_offsets(nb_parts, 0),
      cstr_recv_counts(nb_parts, 0),
      cstr_recv_offsets(nb_parts, 0)
  {
  }

  i_t owned_var_size{0};
  i_t total_var_size{0};
  i_t owned_cstr_size{0};
  i_t total_cstr_size{0};

  // === Variable and Constraint indices owned by this shard, in global problem indices ===
  std::vector<i_t> owned_var_indices;
  std::vector<i_t> owned_cstr_indices;

  // === Send plan: each element is a vector of indices to send to associated peer ===
  std::vector<std::vector<i_t>> var_send_per_peer;
  std::vector<std::vector<i_t>> cstr_send_per_peer;

  // === Recv plan: per peer, contiguous slot in halo region ===
  std::vector<i_t> var_recv_counts;
  std::vector<i_t> var_recv_offsets;
  std::vector<i_t> cstr_recv_counts;
  std::vector<i_t> cstr_recv_offsets;

  // === Mappings ===
  // global_to_local_* : full global problem indices to local shard problem indices
  std::unordered_map<i_t, i_t> global_to_local_var;
  std::unordered_map<i_t, i_t> global_to_local_cstr;
  // local_to_global_* : local shard problem indices to full global problem indices
  std::vector<i_t> local_to_global_var;
  std::vector<i_t> local_to_global_cstr;

  // === Local host CSR matrices ===
  // A
  std::vector<i_t> h_A_row_offsets;
  std::vector<i_t> h_A_col_indices;
  std::vector<f_t> h_A_values;
  // A_t
  std::vector<i_t> h_A_t_row_offsets;
  std::vector<i_t> h_A_t_col_indices;
  std::vector<f_t> h_A_t_values;

  // === Column-ownership split of the local host CSR matrices ===
  // `_own`  : cols in [0, owned_var_size)  for A   /  [0, owned_cstr_size)  for A_t
  //           -- SpMV on this half depends only on owned input entries, so it
  //           can run in parallel with the halo NCCL exchange.
  // `_halo` : cols in [owned_var_size, total_var_size)  /  [owned_cstr_size, total_cstr_size)
  //           -- SpMV on this half needs the halo tail of the input vector, so
  //           it must wait for halo_exchange to complete.
  //
  // Column indices in `_halo` remain absolute local indices (NOT remapped to
  // [0, halo_size)) so both SpMVs can share the same total-sized input vector.
  // Both matrices retain the same row count as the parent (total_cstr_size /
  // total_var_size) with owned-rows populated and halo-rows empty; owned+halo
  // nnz per row sums to the parent's row nnz.
  //
  // A_own / A_halo
  std::vector<i_t> h_A_own_row_offsets;
  std::vector<i_t> h_A_own_col_indices;
  std::vector<f_t> h_A_own_values;
  std::vector<i_t> h_A_halo_row_offsets;
  std::vector<i_t> h_A_halo_col_indices;
  std::vector<f_t> h_A_halo_values;
  // A_t_own / A_t_halo
  std::vector<i_t> h_A_t_own_row_offsets;
  std::vector<i_t> h_A_t_own_col_indices;
  std::vector<f_t> h_A_t_own_values;
  std::vector<i_t> h_A_t_halo_row_offsets;
  std::vector<i_t> h_A_t_halo_col_indices;
  std::vector<f_t> h_A_t_halo_values;
};
}  // namespace cuopt::mathematical_optimization::pdlp
