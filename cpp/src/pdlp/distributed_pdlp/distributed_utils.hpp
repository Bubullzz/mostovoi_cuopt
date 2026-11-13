/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <pdlp/distributed_pdlp/rank_data.hpp>

#include <vector>

namespace cuopt::mathematical_optimization::pdlp {

// Slices the data to prepare a split from graph partitioning with halo communication.
// `index_t` walks the global CSR and may be wider than the solver index `i_t`. Each shard's
// nonzero count is checked to fit in `i_t` before any local offset is stored.
template <typename i_t, typename index_t, typename f_t>
std::vector<rank_data_t<i_t, f_t>> create_rank_data_from_parts(
  const std::vector<int>& parts,
  const std::vector<index_t>& A_row_offsets,
  const std::vector<index_t>& A_col_indices,
  const std::vector<f_t>& A_values,
  const std::vector<index_t>& A_t_row_offsets,
  const std::vector<index_t>& A_t_col_indices,
  const std::vector<f_t>& A_t_values,
  int nb_parts,
  i_t nb_cstr,
  i_t nb_vars,
  index_t nnz);

// Multi-process variant: every rank runs the slicing algorithm on the full
// (broadcast) host data and keeps only its own slice. Internally defers to
// create_rank_data_from_parts and returns the keep_rank-th element.
//
// The halo/send-plan computation is intrinsically cross-rank (rank R's
// send_per_peer[P] is populated by P's iteration), so each rank still pays
// the full algorithm cost today. A future optimization can skip the index-
// remap work for non-kept ranks; the memory of the discarded slices is
// released when the returned rank_data_t detaches.
template <typename i_t, typename index_t, typename f_t>
rank_data_t<i_t, f_t> create_rank_data_for_rank(const std::vector<int>& parts,
                                                const std::vector<index_t>& A_row_offsets,
                                                const std::vector<index_t>& A_col_indices,
                                                const std::vector<f_t>& A_values,
                                                const std::vector<index_t>& A_t_row_offsets,
                                                const std::vector<index_t>& A_t_col_indices,
                                                const std::vector<f_t>& A_t_values,
                                                int nb_parts,
                                                i_t nb_cstr,
                                                i_t nb_vars,
                                                index_t nnz,
                                                int keep_rank);

}  // namespace cuopt::mathematical_optimization::pdlp
