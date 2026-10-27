/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <pdlp/distributed_pdlp/mpi_helpers.hpp>

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/solver_settings.hpp>

namespace cuopt::mathematical_optimization::pdlp {

// Multi-process (MPI) PDLP solve entry point. Called from every rank; the body
// branches internally on `mp.is_master()` for the host-side parse/presolve/
// partition steps and participates collectively (NCCL) in the actual solve.
//
// Caller contract:
//   - `mps_data_model` is populated on master and may be empty on shards.
//   - `settings` must already carry the fully-resolved CLI parameters.
//   - `mp` is the per-process MPI context produced by `mp_context_t`.
//
// Returns 0 on success, -1 on recoverable error.
int run_mp_solve(const cuopt::mathematical_optimization::io::mps_data_model_t<int, double>& mps_data_model,
                 cuopt::mathematical_optimization::solver_settings_t<int, double>& settings,
                 mp_context_t& mp);

}  // namespace cuopt::mathematical_optimization::pdlp
