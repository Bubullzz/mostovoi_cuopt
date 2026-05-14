/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <branch_and_bound/shared_strong_branching_context.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>

#include <pdlp/cusparse_view.hpp>
#include <pdlp/initial_scaling_strategy/initial_scaling.cuh>
#include <pdlp/pdhg.hpp>
#include <pdlp/pdlp_climber_strategy.hpp>
#include <pdlp/restart_strategy/pdlp_restart_strategy.cuh>
#include <pdlp/step_size_strategy/adaptive_step_size_strategy.hpp>
#include <pdlp/swap_and_resize_helper.cuh>
#include <pdlp/termination_strategy/convergence_information.hpp>
#include <pdlp/termination_strategy/termination_strategy.hpp>

#include <mip_heuristics/problem/problem.cuh>

#include <utilities/timer.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <optional>
#include <unordered_set>

namespace cuopt::linear_programming::detail {

/**
 * @brief Abstract base for PDLP-family solvers.
 *
 * Owns all shared state (problem, scaled problem, strategies, iterates, step-size
 * machinery) and the Template Method `run_solver` driver loop. Concrete subclasses
 * implement the divergent operations:
 *   - `single_gpu_solver_t`   : single LP, SpMV path, COL-major layout
 *   - `batch_pdlp_solver_t`   : multi-climber batch, SpMM path, ROW-major layout
 *   - (future) `multi_gpu_pdlp_solver_t` : SpMV + cross-rank reductions / transforms
 *
 * The base ctor does shared initialization only; per-mode setup (e.g. resizing the
 * batched solution storage) lives in the subclass ctors. The base ctor never calls
 * virtual methods (C++ vtable not yet pointing at the derived class during base ctor).
 *
 * @tparam i_t  Data type of indexes
 * @tparam f_t  Data type of the variables and their weights in the equations
 */
template <typename i_t, typename f_t>
class pdlp_solver_base_t {
 public:
  static_assert(std::is_integral<i_t>::value,
                "'pdlp_solver_base_t' accepts only integer types for indexes");
  static_assert(std::is_floating_point<f_t>::value,
                "'pdlp_solver_base_t' accepts only floating point types for weights");

  virtual ~pdlp_solver_base_t() = default;

  optimization_problem_solution_t<i_t, f_t> run_solver(const timer_t& timer);

  f_t get_primal_weight_h(i_t id) const;
  f_t get_step_size_h(i_t id) const;
  i_t get_total_pdhg_iterations() const;
  detail::pdlp_termination_strategy_t<i_t, f_t>& get_current_termination_strategy();

  void swap_context(const thrust::universal_host_pinned_vector<swap_pair_t<i_t>>& swap_pairs);
  void resize_context(i_t new_size);
  void swap_all_context(const thrust::universal_host_pinned_vector<swap_pair_t<i_t>>& swap_pairs);
  void resize_all_context(i_t new_size);
  void resize_and_swap_all_context_loop(
    const std::unordered_set<i_t>& climber_strategies_to_remove);

  void set_problem_ptr(problem_t<i_t, f_t>* problem_ptr_);

  // Interface to let MIP set an initial solution
  // Users will keep on using the optimization_problem to provide an initial solution
  void set_initial_primal_solution(const rmm::device_uvector<f_t>& initial_primal_solution);
  void set_initial_dual_solution(const rmm::device_uvector<f_t>& initial_dual_solution);
  void set_initial_primal_weight(f_t initial_primal_weight);
  void set_initial_step_size(f_t initial_primal_weight);
  void set_initial_k(i_t initial_k);
  void set_relative_primal_tolerance_factor(f_t primal_tolerance_factor);

  using primal_quality_adapter_t =
    typename convergence_information_t<i_t, f_t>::primal_quality_adapter_t;

  const primal_quality_adapter_t& get_best_quality(const primal_quality_adapter_t& current,
                                                   const primal_quality_adapter_t& other);

  void set_inside_mip(bool inside_mip);

  void compute_initial_step_size();
  void compute_initial_primal_weight();

  // Note: `pdhg_solver_` is also part of the public API for direct test access,
  // but it is declared further down (after the state it depends on) so the
  // member-initialization order matches the constructor init list. The class
  // re-opens `public:` access for that one member, then returns to `protected:`.

  void halpern_update();

 protected:
  /**
   * @brief Construct the shared PDLP state.
   *
   * @param op_problem  The optimization problem (single or pre-expanded batch).
   * @param settings    Solver settings.
   */
  pdlp_solver_base_t(problem_t<i_t, f_t>& op_problem,
                     pdlp_solver_settings_t<i_t, f_t> const& settings);

  void print_termination_criteria(const timer_t& timer, bool is_average = false);
  void print_final_termination_criteria(
    const timer_t& timer,
    const convergence_information_t<i_t, f_t>& convergence_information,
    const pdlp_termination_status_t& termination_status,
    bool is_average = false);

  std::optional<optimization_problem_solution_t<i_t, f_t>> check_termination(const timer_t& timer);
  std::optional<optimization_problem_solution_t<i_t, f_t>> check_limits(const timer_t& timer);

  void record_best_primal_so_far(const detail::pdlp_termination_strategy_t<i_t, f_t>& current,
                                 const detail::pdlp_termination_strategy_t<i_t, f_t>& average,
                                 const pdlp_termination_status_t& termination_current,
                                 const pdlp_termination_status_t& termination_average);

  void take_step([[maybe_unused]] i_t total_pdlp_iterations,
                 [[maybe_unused]] bool is_major_iteration);
  void take_adaptive_step(i_t total_pdlp_iterations, bool is_major_iteration);
  void take_constant_step(bool is_major_iteration);


  /**
   * @brief Update current primal & dual solution by setting new solutions and triggering a
   * recomputation of the primal weight and step size
   *
   * @param primal Initial primal solution
   * @param dual Initial dual solution
   */
  void update_primal_dual_solutions(std::optional<const rmm::device_uvector<f_t>*> primal,
                                    std::optional<const rmm::device_uvector<f_t>*> dual);

  // ===================== Hook design (Template Method) =====================
  //
  // `run_solver` is a fixed template-method that drives the iteration loop and
  // calls a small number of COARSE virtual phase hooks. The base class provides
  // the single-LP implementation of every hook; the batch leaf overrides each
  // phase to wrap the base logic with the COL <-> ROW transposes (and to
  // short-circuit termination differently). Future multi-GPU leaves do the
  // same with cross-rank reductions.
  //
  // Beyond the phase hooks, two helper virtuals capture the only remaining
  // truly polymorphic differences:
  //   - `finalize_for_limit_reached`  : how to materialize a solution when a
  //     hard limit (time / iter / concurrent) is hit
  //   - `get_filled_warmed_start_data`: how to pack warm-start data on exit

  /**
   * @brief Pre-loop setup: scaling, initial step size / primal weight, initial
   *        primal/dual application, initial primal projection.
   *
   * Default = single-LP body. Batch override calls the base and then transposes
   * iterates and the scaled problem fields to ROW-major (PDHG works in ROW).
   */
  virtual void setup_initial_state();

  /**
   * @brief Apply the initial-primal projection used at the bottom of
   *        `setup_initial_state`. This is the only sub-step of initial setup
   *        that legitimately differs between single and batch: single-LP
   *        clamps to the scaled variable bounds; batch clamps with per-climber
   *        bound rescaling. Kept as its own hook so the batch override can
   *        replace (not chain to) the base projection.
   *
   * Default = single-LP clamp + restart-to-average buffer clamp.
   */
  virtual void apply_initial_primal_projection();

  /**
   * @brief Major-iteration phase: optionally re-evaluate average solutions,
   *        check termination, and run the restart machinery.
   *
   * Returns `Some(solution)` when PDLP should terminate; otherwise `nullopt`
   * (continue iterating).
   *
   * Default = single-LP body. Batch override does its own batch-specific
   * termination check first (per-climber) and, if not terminating, transposes
   * iterates COL <-> ROW around the base body so the convergence machinery
   * sees COL-major data.
   */
  virtual std::optional<optimization_problem_solution_t<i_t, f_t>> evaluate_and_maybe_restart(
    const timer_t& timer,
    bool is_major_iteration,
    bool artificial_restart_check,
    bool warm_start_was_given,
    std::vector<int>& has_restarted);

  /**
   * @brief Fixed-point-error + Halpern step performed every reflected-PD
   *        iteration. The fixed-error inner work runs only when
   *        `should_compute_fixed_error` is true (i.e. at major step or when an
   *        artificial fixed-error check is requested).
   *
   * Default = single-LP body (SpMV + Halpern). Batch override wraps the
   * fixed-error work with COL <-> ROW transposes and uses the batched
   * (SpMM + reduce) kernel.
   */
  virtual void fixed_error_and_halpern_step(std::vector<int>& has_restarted,
                                            bool should_compute_fixed_error);

  /**
   * @brief Build the final solution when a hard limit (time / iteration /
   * concurrent halt) is hit. Default = single-LP behavior.
   */
  virtual optimization_problem_solution_t<i_t, f_t> finalize_for_limit_reached(
    pdlp_termination_status_t limit_status);

  /**
   * @brief Pack warm-start data for return on a non-trivial termination.
   *
   * Default = single-LP behavior (populated `pdlp_warm_start_data_t`).
   * Batch returns an empty `pdlp_warm_start_data_t` (warm start is not
   * supported in batch).
   */
  virtual pdlp_warm_start_data_t<i_t, f_t> get_filled_warmed_start_data();

  // Building blocks shared between base and batch overrides of
  // `evaluate_and_maybe_restart`. Both leaves do the same pre-termination work
  // (refresh average solutions, unscale) and the same post-termination work
  // (rescale, run restart machinery, post-restart scale). Only the termination
  // check differs (single uses `check_termination`, batch uses
  // `check_batch_termination`). Splitting lets the batch override slot in its
  // own termination check without re-implementing the surrounding logic.
  void evaluate_pre_termination(bool warm_start_was_given);
  void evaluate_post_termination(bool is_major_iteration,
                                 bool artificial_restart_check,
                                 std::vector<int>& has_restarted);

  // Single-LP fixed-point-error computation (SpMV inner). Batch has its own
  // SpMM-based implementation that is invoked from its `fixed_error_and_halpern_step`.
  void compute_fixed_error(std::vector<int>& has_restarted);

  // ===================== Shared state (formerly private) =====================
  //
  // All of the original `pdlp_solver_t` private state lives here so subclasses
  // can read and mutate it directly. Keeping it `protected` (instead of breaking
  // it up by mode) is intentional — the iterates, step sizes, and strategy
  // objects are all naturally shared.

  // Initial number of climbers (derived from settings.fixed_batch_size / settings.new_bounds at
  // ctor time).
  // Stable throughout solving — use this whenever you need the ORIGINAL batch size, since
  // `climber_strategies_` shrinks as climbers finish via resize_and_swap_all_context_loop.
  const size_t original_batch_size_;
  std::vector<pdlp_climber_strategy_t> climber_strategies_;

  raft::handle_t const* handle_ptr_;
  rmm::cuda_stream_view stream_view_;
  // Intentionnaly take a copy to avoid an unintentional modification in the calling context
  const pdlp_solver_settings_t<i_t, f_t> settings_;
  dual_simplex::shared_strong_branching_context_view_t<i_t, f_t> sb_view_{
    settings_.shared_sb_solved};

  problem_t<i_t, f_t>* problem_ptr;
  // Combined bounds in op_problem_scaled_ will only be scaled if
  // compute_initial_primal_weight_before_scaling is false because of compute_initial_primal_weight
  problem_t<i_t, f_t> op_problem_scaled_;

  rmm::device_uvector<f_t> unscaled_primal_avg_solution_;
  rmm::device_uvector<f_t> unscaled_dual_avg_solution_;

  i_t primal_size_h_;
  i_t dual_size_h_;

  rmm::device_uvector<f_t> primal_step_size_;
  rmm::device_uvector<f_t> dual_step_size_;

  /**
  The primal and dual step sizes are parameterized as:
    tau = primal_step_size = step_size / primal_weight
    sigma = dual_step_size = step_size * primal_weight.
  The primal_weight factor is named as such because this parameterization is
  equivalent to defining the Bregman divergences as:
  D_x(x, x bar) = 0.5 * primal_weight ||x - x bar||_2^2, and
  D_y(y, y bar) = 0.5 / primal_weight ||y - y bar||_2^2.

  The parameter primal_weight is adjusted smoothly at each restart; to balance the
  primal and dual distances traveled since the last restart.
  */
  rmm::device_uvector<f_t> primal_weight_;
  rmm::device_uvector<f_t> best_primal_weight_;
  rmm::device_uvector<f_t> step_size_;

  // Step size strategy
  detail::adaptive_step_size_strategy_t<i_t, f_t> step_size_strategy_;

 public:
  // Inner solver. Public for direct test access; do not rely on this externally.
  // Declared here (after step_size_strategy_) so member-initialization order
  // matches the dependencies wired up in the ctor init list.
  detail::pdhg_solver_t<i_t, f_t> pdhg_solver_;

 protected:
  // Initial scaling strategy
  detail::pdlp_initial_scaling_strategy_t<i_t, f_t> initial_scaling_strategy_;

  // For the average evaluation
  detail::cusparse_view_t<i_t, f_t> average_op_problem_evaluation_cusparse_view_;
  detail::cusparse_view_t<i_t, f_t> current_op_problem_evaluation_cusparse_view_;

  // Restart strategy
  detail::pdlp_restart_strategy_t<i_t, f_t> restart_strategy_;
  // Termination strategy
  detail::pdlp_termination_strategy_t<i_t, f_t> average_termination_strategy_;
  detail::pdlp_termination_strategy_t<i_t, f_t> current_termination_strategy_;

  /* Two counters are necessary because of the PDLP warm start data
   *  total_pdlp_iterations_: total, counting potential previous PDLP iterations
   *    Useful for:
   *      - Not triggerring on the min iteration restart
   *      - Not triggering a check_limits without optimality check
   *      - Correct restart information
   * internal_solver_iterations_: only current PDLP object iterations
   *    Useful for:
   *      - Returning the correct amount of iterations in the solution object
   *      - Correct iteration limit trigger
   */
  i_t total_pdlp_iterations_{0};
  i_t internal_solver_iterations_{0};

  // Initial solution
  rmm::device_uvector<f_t> initial_primal_;
  rmm::device_uvector<f_t> initial_dual_;
  // Used in the context of MIP to relaunch PDLP from a pseudo previous state
  std::optional<f_t> initial_primal_weight_;
  std::optional<f_t> initial_step_size_;
  std::optional<i_t> initial_k_;

  const rmm::device_scalar<f_t> reusable_device_scalar_value_1_;
  const rmm::device_scalar<f_t> reusable_device_scalar_value_0_;

  // Only used if save_best_primal_so_far is toggeled
  optimization_problem_solution_t<i_t, f_t> best_primal_solution_so_far;
  primal_quality_adapter_t best_primal_quality_so_far_;
  // Flag to indicate if solver is being called from MIP. No logging is done in this case.
  bool inside_mip_{false};
};

/**
 * @brief Single-GPU, single-LP PDLP solver.
 *
 * Concrete leaf for the SpMV / COL-major / single-climber path. Uses every base
 * default — single-GPU IS the default behavior the base implements.
 */
template <typename i_t, typename f_t>
class single_gpu_solver_t : public pdlp_solver_base_t<i_t, f_t> {
 public:
  single_gpu_solver_t(problem_t<i_t, f_t>& op_problem,
                      pdlp_solver_settings_t<i_t, f_t> const& settings);
};

/**
 * @brief Batched PDLP solver (multi-climber, single GPU).
 *
 * Concrete leaf for the SpMM / ROW-major / multi-climber path. Owns the batched
 * solution-storage buffers and the COL<->ROW transpose machinery. cuPDLPx-style
 * (reflected primal-dual + Halpern + fixed-point-error) is implicitly required —
 * features incompatible with batch are validated in the constructor.
 */
template <typename i_t, typename f_t>
class batch_pdlp_solver_t : public pdlp_solver_base_t<i_t, f_t> {
  // Bring base-class members into the derived class scope so member-function
  // bodies can reference them without `this->` qualification. Required because
  // of two-phase name lookup for class templates with a dependent base.
  using base_t = pdlp_solver_base_t<i_t, f_t>;
  using base_t::average_op_problem_evaluation_cusparse_view_;
  using base_t::average_termination_strategy_;
  using base_t::best_primal_quality_so_far_;
  using base_t::best_primal_solution_so_far;
  using base_t::best_primal_weight_;
  using base_t::climber_strategies_;
  using base_t::current_op_problem_evaluation_cusparse_view_;
  using base_t::current_termination_strategy_;
  using base_t::dual_size_h_;
  using base_t::dual_step_size_;
  using base_t::handle_ptr_;
  using base_t::initial_dual_;
  using base_t::initial_k_;
  using base_t::initial_primal_;
  using base_t::initial_primal_weight_;
  using base_t::initial_scaling_strategy_;
  using base_t::initial_step_size_;
  using base_t::inside_mip_;
  using base_t::internal_solver_iterations_;
  using base_t::op_problem_scaled_;
  using base_t::original_batch_size_;
  using base_t::pdhg_solver_;
  using base_t::primal_size_h_;
  using base_t::primal_step_size_;
  using base_t::primal_weight_;
  using base_t::problem_ptr;
  using base_t::restart_strategy_;
  using base_t::reusable_device_scalar_value_0_;
  using base_t::reusable_device_scalar_value_1_;
  using base_t::sb_view_;
  using base_t::settings_;
  using base_t::step_size_;
  using base_t::step_size_strategy_;
  using base_t::stream_view_;
  using base_t::total_pdlp_iterations_;
  using base_t::unscaled_dual_avg_solution_;
  using base_t::unscaled_primal_avg_solution_;
  // Method names that need to be visible too (for unqualified calls).
  using base_t::get_primal_weight_h;
  using base_t::get_step_size_h;
  using base_t::record_best_primal_so_far;

 public:
  batch_pdlp_solver_t(problem_t<i_t, f_t>& op_problem,
                      pdlp_solver_settings_t<i_t, f_t> const& settings);

 protected:
  // ---- Coarse phase hooks (override base single-LP behavior) ----
  void setup_initial_state() override;
  void apply_initial_primal_projection() override;

  std::optional<optimization_problem_solution_t<i_t, f_t>> evaluate_and_maybe_restart(
    const timer_t& timer,
    bool is_major_iteration,
    bool artificial_restart_check,
    bool warm_start_was_given,
    std::vector<int>& has_restarted) override;

  void fixed_error_and_halpern_step(std::vector<int>& has_restarted,
                                    bool should_compute_fixed_error) override;

  optimization_problem_solution_t<i_t, f_t> finalize_for_limit_reached(
    pdlp_termination_status_t limit_status) override;

  pdlp_warm_start_data_t<i_t, f_t> get_filled_warmed_start_data() override;

 private:
  // Batch-only return buffer. Sized at end of ctor.
  optimization_problem_solution_t<i_t, f_t> batch_solution_to_return_;

  // ---- Layout helpers (PDHG runs in ROW-major; eval/swap want COL-major). ----
  void transpose_problem_to_row();
  void transpose_problem_to_col();
  void transpose_iterates_to_row(rmm::device_uvector<f_t>& primal,
                                 rmm::device_uvector<f_t>& dual,
                                 rmm::device_uvector<f_t>& dual_slack);
  void transpose_iterates_to_col(rmm::device_uvector<f_t>& primal,
                                 rmm::device_uvector<f_t>& dual,
                                 rmm::device_uvector<f_t>& dual_slack);

  // ---- Batch SpMM-based fixed-point-error computation. ----
  // Replaces (does not chain to) base's SpMV-based `compute_fixed_error`.
  void compute_fixed_error_batch(std::vector<int>& has_restarted);

  // ---- Batch termination & solution-collection helpers. ----
  std::optional<optimization_problem_solution_t<i_t, f_t>> check_batch_termination(
    const timer_t& timer);

  // Snapshot the current iterate of climber `i` (batch-local index) into
  // `batch_solution_to_return_` at its `original_index` slot.
  void snapshot_climber_into_return(size_t i);

  // Flush GPU termination stats into `batch_solution_to_return_` and construct
  // the final solution.
  optimization_problem_solution_t<i_t, f_t> finalize_batch_return();
};

/**
 * @brief Factory for the right concrete PDLP solver based on settings.
 *
 * Returns a `batch_pdlp_solver_t` if the settings indicate a batch SpMM run
 * (`fixed_batch_size > 0` or non-empty `new_bounds`), otherwise a
 * `single_gpu_solver_t`. Callers hold the result as a
 * `std::unique_ptr<pdlp_solver_base_t<i_t, f_t>>` and use the abstract API.
 */
template <typename i_t, typename f_t>
std::unique_ptr<pdlp_solver_base_t<i_t, f_t>> make_pdlp_solver(
  problem_t<i_t, f_t>& op_problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings);

}  // namespace cuopt::linear_programming::detail
