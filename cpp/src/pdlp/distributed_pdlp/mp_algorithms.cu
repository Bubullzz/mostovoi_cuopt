/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Definitions of the multi-process (MPI + NCCL) algorithms declared in mp_algorithms.hpp
// (scaling, power iteration, initial step size / primal weight). They are built on the
// primitives of distributed_state_t (distributed_state.cu). The single-process multi-GPU
// counterparts live in distributed_algorithms.cu.
#include <pdlp/distributed_pdlp/mp_algorithms.hpp>

#include <pdlp/cusparse_view.hpp>
#include <pdlp/utils.cuh>

#include <raft/core/nvtx.hpp>

#include <utilities/device_scalar_init.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/gather.h>
#include <cub/device/device_transform.cuh>

#include <cmath>
#include <vector>

namespace cuopt::mathematical_optimization::pdlp {

// Same math as the multi_gpu_engine_t versions in distributed_algorithms.cu, but every
// function runs on THIS rank only: no for_each_shard, no master/shard copies, no
// cross-stream events.

// -------- Refresh halo of cumulative scalings (MP) -------------------------
template <typename i_t, typename f_t>
void refresh_halo_cummulative_scalings(distributed_state_t<i_t, f_t>& dist,
                                       pdlp_solver_t<i_t, f_t>& solver)
{
  const cuda::stream_ref stream = solver.get_handle_ptr()->get_stream();
  auto& scaling                 = solver.get_initial_scaling_strategy();
  dist.halo_exchange_var(scaling.get_cummulative_variable_scaling(), stream);
  dist.halo_exchange_cstr(scaling.get_cummulative_constraint_matrix_scaling(), stream);
}

// -------- Bound / objective rescaling (MP) ---------------------------------
// Local raw squared sums over the OWNED slice -> allreduce -> same scalar on every rank.
// Raw squared sums (not norms) because sqrt is not additive across ranks.
template <typename i_t, typename f_t>
void distributed_bound_objective_rescaling(distributed_state_t<i_t, f_t>& dist,
                                           pdlp_solver_t<i_t, f_t>& solver,
                                           f_t c_scaling_weight)
{
  raft::common::nvtx::range scope("mp_distributed_bound_objective_rescaling");

  const cuda::stream_ref stream = solver.get_handle_ptr()->get_stream();
  auto& scaling                 = solver.get_initial_scaling_strategy();
  const auto& scaled            = scaling.get_scaled_op_problem();

  rmm::device_scalar<f_t> d_bound_sq(zero_v<f_t>, stream);
  rmm::device_scalar<f_t> d_obj_sq(zero_v<f_t>, stream);

  compute_sum_bounds_squared(scaled.constraint_lower_bounds,
                             scaled.constraint_upper_bounds,
                             d_bound_sq,
                             stream,
                             static_cast<std::size_t>(dist.rank_data.owned_cstr_size));
  compute_sum_weighted_squares(scaled.objective_coefficients,
                               c_scaling_weight,
                               d_obj_sq,
                               stream,
                               static_cast<std::size_t>(dist.rank_data.owned_var_size));

  dist.allreduce_sum_inplace(raft::make_device_scalar_view<f_t>(d_bound_sq.data()), stream);
  dist.allreduce_sum_inplace(raft::make_device_scalar_view<f_t>(d_obj_sq.data()), stream);
  const f_t global_bound_sq = d_bound_sq.value(stream);
  const f_t global_obj_sq   = d_obj_sq.value(stream);

  const f_t bound_rescaling = rescaling_from_squared_norm_op<f_t>{}(global_bound_sq);
  const f_t obj_rescaling   = rescaling_from_squared_norm_op<f_t>{}(global_obj_sq);

  scaling.set_h_bound_rescaling(bound_rescaling);
  scaling.set_h_objective_rescaling(obj_rescaling);
  scaling.apply_bound_objective_rescaling_to_problem();
}

// -------- Curtis-Reid scaling (MP) -----------------------------------------
// Owned rows of A and owned columns of A_T are complete, so each log-mean is
// local. The other axis's log-scale is read at halo indices, so it is exchanged
// between the row pass and the column pass.
template <typename i_t, typename f_t>
void distributed_curtis_reid_scaling(distributed_state_t<i_t, f_t>& dist,
                                     pdlp_solver_t<i_t, f_t>& solver,
                                     int num_iter)
{
  raft::common::nvtx::range scope("mp_distributed_curtis_reid_scaling");

  const cuda::stream_ref stream = solver.get_handle_ptr()->get_stream();
  auto& scaling                 = solver.get_initial_scaling_strategy();

  scaling.curtis_reid_init();
  for (int it = 0; it < num_iter; ++it) {
    scaling.curtis_reid_row_iteration();
    dist.halo_exchange_cstr(scaling.get_iteration_constraint_matrix_scaling(), stream);

    scaling.curtis_reid_col_iteration();
    dist.halo_exchange_var(scaling.get_iteration_variable_scaling(), stream);
  }
  scaling.curtis_reid_folding();

  // Downstream passes read halo copies of the cumulative scaling.
  refresh_halo_cummulative_scalings(dist, solver);
}

// -------- Ruiz inf-scaling (MP) --------------------------------------------
// Each rank owns its rows AND its columns and stores both complete (h_A = owned
// rows, h_A_t = owned columns), so a Ruiz iteration is local once the halo copies
// of the cumulative scalings are fresh.
template <typename i_t, typename f_t>
void distributed_ruiz_inf_scaling(distributed_state_t<i_t, f_t>& dist,
                                  pdlp_solver_t<i_t, f_t>& solver,
                                  int num_iter)
{
  if (num_iter <= 0 || dist.n_global_vars <= 0) return;
  raft::common::nvtx::range scope("mp_distributed_ruiz_inf_scaling");

  auto& scaling = solver.get_initial_scaling_strategy();
  for (int it = 0; it < num_iter; ++it) {
    refresh_halo_cummulative_scalings(dist, solver);
    scaling.ruiz_iter_local();
  }

  // Final refresh after last iteration
  refresh_halo_cummulative_scalings(dist, solver);
}

// -------- Pock-Chambolle scaling (MP) --------------------------------------
// One pass mirroring single-GPU pock_chambolle_scaling. Row sum-of-powers come from
// the row-major matrix (owned rows) and column sum-of-powers from A_T (owned columns).
template <typename i_t, typename f_t>
void distributed_pock_chambolle_scaling(distributed_state_t<i_t, f_t>& dist,
                                        pdlp_solver_t<i_t, f_t>& solver,
                                        f_t alpha)
{
  if (dist.n_global_vars <= 0) return;
  raft::common::nvtx::range scope("mp_distributed_pock_chambolle_scaling");

  refresh_halo_cummulative_scalings(dist, solver);
  solver.get_initial_scaling_strategy().pock_chambolle_scaling(alpha);
  // Final refresh for downstream consumers.
  refresh_halo_cummulative_scalings(dist, solver);
}

// -------- Scaling orchestration (MP) ---------------------------------------
// Mirrors single-GPU scale_problem().
template <typename i_t, typename f_t>
void distributed_scaling(distributed_state_t<i_t, f_t>& dist,
                         pdlp_solver_t<i_t, f_t>& solver,
                         pdlp_hyper_params_t const& hyper_params,
                         bool inside_mip)
{
  raft::common::nvtx::range scope("mp_distributed_scaling");

  // 1) Matrix scaling passes populate the cumulative row/col scalings.
  //    Curtis-Reid is a prescale and is skipped inside MIP, matching single-GPU.
  if (hyper_params.do_curtis_reid_scaling && !inside_mip) {
    distributed_curtis_reid_scaling(dist, solver, hyper_params.number_of_curtis_reid_iterations);
  }
  if (hyper_params.do_ruiz_scaling) {
    distributed_ruiz_inf_scaling(dist, solver, hyper_params.default_l_inf_ruiz_iterations);
  }
  if (hyper_params.do_pock_chambolle_scaling) {
    distributed_pock_chambolle_scaling(
      dist, solver, static_cast<f_t>(hyper_params.default_alpha_pock_chambolle_rescaling));
  }

  // 2) Apply the accumulated scaling to A, c, variable and constraint bounds.
  //    This is scale_problem() minus its local bound/objective rescaling; the
  //    equivalent global step happens in (3).
  solver.get_initial_scaling_strategy().apply_cummulative_scaling_to_problem();

  // 3) Global bound/objective rescaling (all ranks get the identical scalar).
  if (hyper_params.bound_objective_rescaling) {
    distributed_bound_objective_rescaling(
      dist, solver, static_cast<f_t>(hyper_params.initial_primal_weight_c_scaling));
  }
}

// -------- sigma_max(A)^2 via power iteration (MP) --------------------------
// Each rank holds the owned slice of q / z and the full-size (owned + halo) buffers
// the local SpMVs need. Every cross-rank operation goes through dist's primitives
// (halo exchange, distributed dot / l2 norm), so this function has no NCCL calls.
//
// Every rank takes the same convergence decision: `residual_norm` comes out of an
// allreduce, so it is the same value on all ranks and they all break at the same
// iteration (required, otherwise the collectives would mismatch and hang).
template <typename i_t, typename f_t>
f_t distributed_max_singular_value_squared(distributed_state_t<i_t, f_t>& dist,
                                           pdlp_solver_t<i_t, f_t>& solver,
                                           int max_iterations,
                                           f_t tolerance)
{
  raft::common::nvtx::range scope("mp_distributed_max_singular_value_squared");

  raft::handle_t const& handle  = *solver.get_handle_ptr();
  const cuda::stream_ref stream = handle.get_stream();
  auto& pdhg                    = solver.pdhg_solver_;

  const i_t cstr_total = dist.rank_data.total_cstr_size;
  const i_t var_total  = dist.rank_data.total_var_size;
  const i_t n_owned    = dist.rank_data.owned_cstr_size;

  // ┌──────────────────────────────────────────────────────────────┐
  // │                            Setup                             │
  // └──────────────────────────────────────────────────────────────┘

  // Generate the GLOBAL z[] sequence in cstr-index order (same on every rank),
  // then keep only this rank's owned entries.
  std::vector<f_t> h_global_z =
    make_singular_value_probe<f_t>(static_cast<std::size_t>(dist.n_global_cstrs));

  rmm::device_uvector<f_t> q(static_cast<std::size_t>(cstr_total), stream);
  rmm::device_uvector<f_t> z(static_cast<std::size_t>(cstr_total), stream);
  rmm::device_uvector<f_t> atq(static_cast<std::size_t>(var_total), stream);
  rmm::device_scalar<f_t> sigma_sq(stream);
  rmm::device_scalar<f_t> norm_q(stream);
  rmm::device_scalar<f_t> residual_norm(stream);

  auto q_dn   = make_dnvec<f_t>(static_cast<int64_t>(cstr_total), q.data());
  auto z_dn   = make_dnvec<f_t>(static_cast<int64_t>(cstr_total), z.data());
  auto atq_dn = make_dnvec<f_t>(static_cast<int64_t>(var_total), atq.data());

  // Owned-slice views: halo entries must not be counted in the global dot / norm.
  raft::device_span<f_t> q_owned(q.data(), static_cast<std::size_t>(n_owned));
  raft::device_span<f_t> z_owned(z.data(), static_cast<std::size_t>(n_owned));

  std::vector<f_t> h_owned_z(static_cast<std::size_t>(n_owned));
  thrust::gather(thrust::host,
                 dist.rank_data.local_to_global_cstr.begin(),
                 dist.rank_data.local_to_global_cstr.begin() + n_owned,
                 h_global_z.begin(),
                 h_owned_z.begin());
  raft::copy(z.data(), h_owned_z.data(), static_cast<std::size_t>(n_owned), stream);
  thrust::fill(rmm::exec_policy_nosync(stream), z.data() + n_owned, z.data() + cstr_total, f_t(0));

  // Sync so h_owned_z stays valid through the H2D copy (it goes out of scope at
  // the end of this function, but better to be explicit than rely on that).
  stream.sync();

  // ┌──────────────────────────────────────────────────────────────┐
  // │                        Power iteration                       │
  // └──────────────────────────────────────────────────────────────┘

  // Mirrors single-GPU compute_initial_step_size.
  // copy -> l2 norm -> transform -> SpMV -> SpMV -> dot -> transform -> norm -> convergence check
  for (int it = 0; it < max_iterations; ++it) {
    // q := z on the owned slice (the carried iterate).
    raft::copy(q.data(), z.data(), n_owned, stream);

    // ||q||₂ over the global OWNED cstr slice (one allreduce-sum + sqrt).
    dist.distributed_l2_norm(q_owned, norm_q, handle);

    // q /= ||q||₂ on owned slice (halo gets refreshed by the next exchange).
    cub::DeviceTransform::Transform(q.data(),
                                    q.data(),
                                    n_owned,
                                    divide_by_device_scalar_t<f_t>{norm_q.data()},
                                    stream.get());

    // atq = A^T q  (halo-refresh of q + local SpMV).
    dist.distributed_spmv_At(pdhg, q, q_dn.get(), atq_dn.get(), stream);

    // z = A atq  (halo-refresh of atq + local SpMV).
    dist.distributed_spmv_A(pdhg, atq, atq_dn.get(), z_dn.get(), stream);

    // σ² = q · z over the global OWNED cstr slice (= q^T A A^T q = σ_max²
    // when q is the dominant left-singular vector).
    dist.distributed_dot(q_owned, z_owned, sigma_sq, handle);

    // q := -σ² q + z (owned slice) — residual of the eigen-equation.
    cub::DeviceTransform::Transform(cuda::std::make_tuple(q.data(), z.data()),
                                    q.data(),
                                    n_owned,
                                    residual_fma_neg_scalar_t<f_t>{sigma_sq.data()},
                                    stream.get());

    // Convergence check via global residual norm (value() syncs the stream).
    dist.distributed_l2_norm(q_owned, residual_norm, handle);
    if (residual_norm.value(stream) < tolerance) break;
  }

  // σ_max² is the same on every rank after the last allreduce.
  return sigma_sq.value(stream);
}

// -------- Initial step size (MP) -------------------------------------------
// Mirrors single-GPU compute_initial_step_size (power-iteration branch). Every rank
// ends up with the same step size in solver.get_step_size().
template <typename i_t, typename f_t>
void distributed_compute_initial_step_size(distributed_state_t<i_t, f_t>& dist,
                                           pdlp_solver_t<i_t, f_t>& solver,
                                           pdlp_hyper_params_t const& hyper_params,
                                           f_t scaling_factor,
                                           int max_iterations,
                                           f_t tolerance)
{
  raft::common::nvtx::range scope("mp_distributed_compute_initial_step_size");
  cuopt_expects(hyper_params.initial_step_size_max_singular_value,
                error_type_t::ValidationError,
                "distributed_compute_initial_step_size requires "
                "initial_step_size_max_singular_value = true; the max-abs-value "
                "of A fallback is single-GPU only.");

  const f_t sigma_max_sq =
    distributed_max_singular_value_squared(dist, solver, max_iterations, tolerance);
  const f_t h_step_size = scaling_factor / std::sqrt(sigma_max_sq);

  const cuda::stream_ref stream = solver.get_handle_ptr()->get_stream();
  raft::copy(solver.get_step_size().data(), &h_step_size, 1, stream);
  // h_step_size is a stack local: wait for the H2D copy before returning.
  stream.sync();
}

// -------- Initial primal weight (MP) ---------------------------------------
// Distributed PDLP is restricted to the Stable3-shaped hyper-param profile, for which
// single-GPU compute_initial_primal_weight short-circuits to primal_weight = 1
// (!initial_primal_weight_combined_bounds && bound_objective_rescaling), without
// touching any norm. So there is nothing to reduce: just set the constant locally.
template <typename i_t, typename f_t>
void distributed_compute_initial_primal_weight(distributed_state_t<i_t, f_t>& /*dist*/,
                                               pdlp_solver_t<i_t, f_t>& solver,
                                               pdlp_hyper_params_t const& hyper_params)
{
  raft::common::nvtx::range scope("mp_distributed_compute_initial_primal_weight");
  cuopt_expects(
    !hyper_params.initial_primal_weight_combined_bounds && hyper_params.bound_objective_rescaling,
    error_type_t::ValidationError,
    "distributed_compute_initial_primal_weight: only the Stable3-shaped "
    "short-circuit is supported (initial_primal_weight_combined_bounds=false "
    "and bound_objective_rescaling=true).");

  const f_t h_primal_weight     = f_t(1);
  const cuda::stream_ref stream = solver.get_handle_ptr()->get_stream();
  raft::copy(solver.get_primal_weight().data(), &h_primal_weight, 1, stream);
  raft::copy(solver.get_best_primal_weight().data(), &h_primal_weight, 1, stream);
  // h_primal_weight is a stack local: wait for the H2D copies before returning.
  stream.sync();
}

// ----- Explicit instantiations ---------------------------------------------
#define INSTANTIATE(F_TYPE)                                                                  \
  template void refresh_halo_cummulative_scalings<int, F_TYPE>(distributed_state_t<int, F_TYPE>&, \
                                                               pdlp_solver_t<int, F_TYPE>&);       \
  template void distributed_bound_objective_rescaling<int, F_TYPE>(                          \
    distributed_state_t<int, F_TYPE>&, pdlp_solver_t<int, F_TYPE>&, F_TYPE);                 \
  template void distributed_curtis_reid_scaling<int, F_TYPE>(                                \
    distributed_state_t<int, F_TYPE>&, pdlp_solver_t<int, F_TYPE>&, int);                    \
  template void distributed_ruiz_inf_scaling<int, F_TYPE>(                                   \
    distributed_state_t<int, F_TYPE>&, pdlp_solver_t<int, F_TYPE>&, int);                    \
  template void distributed_pock_chambolle_scaling<int, F_TYPE>(                             \
    distributed_state_t<int, F_TYPE>&, pdlp_solver_t<int, F_TYPE>&, F_TYPE);                 \
  template void distributed_scaling<int, F_TYPE>(distributed_state_t<int, F_TYPE>&,          \
                                                 pdlp_solver_t<int, F_TYPE>&,                \
                                                 pdlp_hyper_params_t const&,                 \
                                                 bool);                                      \
  template F_TYPE distributed_max_singular_value_squared<int, F_TYPE>(                       \
    distributed_state_t<int, F_TYPE>&, pdlp_solver_t<int, F_TYPE>&, int, F_TYPE);            \
  template void distributed_compute_initial_step_size<int, F_TYPE>(                          \
    distributed_state_t<int, F_TYPE>&,                                                       \
    pdlp_solver_t<int, F_TYPE>&,                                                             \
    pdlp_hyper_params_t const&,                                                              \
    F_TYPE,                                                                                  \
    int,                                                                                     \
    F_TYPE);                                                                                 \
  template void distributed_compute_initial_primal_weight<int, F_TYPE>(                      \
    distributed_state_t<int, F_TYPE>&, pdlp_solver_t<int, F_TYPE>&, pdlp_hyper_params_t const&);

INSTANTIATE(double)
INSTANTIATE(float)

#undef INSTANTIATE

}  // namespace cuopt::mathematical_optimization::pdlp
