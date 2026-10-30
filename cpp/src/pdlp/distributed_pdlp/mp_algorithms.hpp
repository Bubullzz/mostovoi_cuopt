/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <pdlp/distributed_pdlp/distributed_state.hpp>
#include <pdlp/pdlp.cuh>

#include <cuopt/mathematical_optimization/pdlp/pdlp_hyper_params.cuh>

namespace cuopt::mathematical_optimization::pdlp {

// High-level algorithms for the multi-process (MPI + NCCL) path: one rank = one shard.
//
// Each function drives this rank's `solver` and uses `dist` for the cross-rank
// communication (halo exchange, allreduce, ...) and for the global problem sizes.
// They are built on distributed_state_t's primitives (distributed_state.cu).
// Definitions: mp_algorithms.cu. The single-process multi-GPU counterparts are
// multi_gpu_engine_t's members in distributed_algorithms.cu.
//
// Everything runs on the rank's single stream (the pdlp handle's stream), so
// ordering is implicit and host syncs only happen when a value is read back.
// These are collectives: every rank must call them in the same order.

// Refresh the halo copies of the cumulative variable + constraint scalings.
template <typename i_t, typename f_t>
void refresh_halo_cummulative_scalings(distributed_state_t<i_t, f_t>& dist,
                                       pdlp_solver_t<i_t, f_t>& solver);

// Global (allreduced) bound / objective rescaling.
template <typename i_t, typename f_t>
void distributed_bound_objective_rescaling(distributed_state_t<i_t, f_t>& dist,
                                           pdlp_solver_t<i_t, f_t>& solver,
                                           f_t c_scaling_weight);

// Matrix scaling passes (each keeps the halo copies of the scalings fresh).
template <typename i_t, typename f_t>
void distributed_curtis_reid_scaling(distributed_state_t<i_t, f_t>& dist,
                                     pdlp_solver_t<i_t, f_t>& solver,
                                     int num_iter);
template <typename i_t, typename f_t>
void distributed_ruiz_inf_scaling(distributed_state_t<i_t, f_t>& dist,
                                  pdlp_solver_t<i_t, f_t>& solver,
                                  int num_iter);
template <typename i_t, typename f_t>
void distributed_pock_chambolle_scaling(distributed_state_t<i_t, f_t>& dist,
                                        pdlp_solver_t<i_t, f_t>& solver,
                                        f_t alpha);

// Mirrors single-GPU scale_problem(): the passes above + apply + global rescaling.
template <typename i_t, typename f_t>
void distributed_scaling(distributed_state_t<i_t, f_t>& dist,
                         pdlp_solver_t<i_t, f_t>& solver,
                         pdlp_hyper_params_t const& hyper_params,
                         bool inside_mip);

// sigma_max(A)^2 by power iteration on A A^T.
template <typename i_t, typename f_t>
f_t distributed_max_singular_value_squared(distributed_state_t<i_t, f_t>& dist,
                                           pdlp_solver_t<i_t, f_t>& solver,
                                           int max_iterations,
                                           f_t tolerance);

// Writes solver.get_step_size() = scaling_factor / sigma_max(A).
template <typename i_t, typename f_t>
void distributed_compute_initial_step_size(distributed_state_t<i_t, f_t>& dist,
                                           pdlp_solver_t<i_t, f_t>& solver,
                                           pdlp_hyper_params_t const& hyper_params,
                                           f_t scaling_factor,
                                           int max_iterations,
                                           f_t tolerance);

// Writes solver.get_primal_weight() / get_best_primal_weight() = 1.
template <typename i_t, typename f_t>
void distributed_compute_initial_primal_weight(distributed_state_t<i_t, f_t>& dist,
                                               pdlp_solver_t<i_t, f_t>& solver,
                                               pdlp_hyper_params_t const& hyper_params);

}  // namespace cuopt::mathematical_optimization::pdlp
