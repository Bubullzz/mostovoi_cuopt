
int run_mp_solve(const std::string& file_name,
    cuopt::mathematical_optimization::io::mps_reader_type_t mps_reader,
    cuopt::mathematical_optimization::solver_settings_t<int, double>& settings,
    mp_context_t& mp)
{
assert(cuopt::mathematical_optimization::get_memory_backend_type() == cuopt::mathematical_optimization::memory_backend_t::GPU);
auto handle_ptr = std::make_unique<raft::handle_t>();
auto problem_interface =
std::make_unique<cuopt::mathematical_optimization::optimization_problem_t<int, double>>(
handle_ptr.get());

auto pdlp_settings_ref = settings.get_pdlp_settings();
if (mp.is_master()) {
// The CLI and the solver library have separate loggers that both write this file.
// Configure the solver's first so its own initializer reuses that configuration rather
// than truncating the file mid-solve; the CLI's own logger then appends to it.
const auto log_file    = settings.get_parameter<std::string>(CUOPT_LOG_FILE);
const auto log_console = settings.get_parameter<bool>(CUOPT_LOG_TO_CONSOLE);

auto solver_log =
cuopt::mathematical_optimization::configure_logging(log_file, log_console, true);
cuopt::init_logger_t log(log_file, log_console, /*truncate=*/false);

auto maybe_mps = read_mps_or_fail(file_path, mps_reader, settings);
if (!maybe_mps) { return -1; }
auto mps_data_model = std::move(*maybe_mps);
const bool run_presolve = settings_resolved.presolver != presolver_t::None;
std::unique_ptr<mip::third_party_presolve_t<i_t, f_t>> presolver_ptr;
std::optional<mip::third_party_presolve_host_result_t<i_t, f_t>> host_res;
[[maybe_unused]] double presolve_time = 0.0;

if (run_presolve) {
// mirroring single-GPU solve.cu
const double presolve_time_limit =
std::max(1.0, std::min(0.1 * lp_timer.remaining_time(), 60.0));

presolver_ptr = std::make_unique<mip::third_party_presolve_t<i_t, f_t>>();
host_res      = presolver_ptr->apply_presolve_from_mps_data(
model,
cuopt::mathematical_optimization::problem_category_t::LP,
settings_resolved.presolver,
settings_resolved.dual_postsolve,
settings_resolved.tolerances.absolute_primal_tolerance,
settings_resolved.tolerances.relative_primal_tolerance,
presolve_time_limit);

if (auto terminal = terminal_solution_from_presolve_status<i_t, f_t>(
host_res->status, handle_ptr->get_stream())) {
return std::move(*terminal);
}

// Presolve completely solved the problem.
if (host_res->reduced_problem.get_n_variables() == 0 &&
host_res->reduced_problem.get_n_constraints() == 0) {
CUOPT_LOG_INFO("Presolve completely solved the problem");
presolve_time = lp_timer.elapsed_time();
CUOPT_LOG_INFO("%s presolve time: %.2fs",
          settings_resolved.presolver == presolver_t::PSLP ? "PSLP" : "Papilo",
          presolve_time);

// Postsolve is host-side here (no reduced GPU problem was ever built);
// bounce the resulting vectors to device to satisfy the solution API.
std::vector<f_t> h_primal, h_dual, h_rc;
presolver_ptr->undo(h_primal,
               h_dual,
               h_rc,
               cuopt::mathematical_optimization::problem_category_t::LP,
               /*status_to_skip=*/false,
               settings_resolved.dual_postsolve);
auto primal_uv = cuopt::device_copy(h_primal, handle_ptr->get_stream());
auto dual_uv   = cuopt::device_copy(h_dual, handle_ptr->get_stream());
auto rc_uv     = cuopt::device_copy(h_rc, handle_ptr->get_stream());
handle_ptr->sync_stream();

return build_presolve_optimal_solution<i_t, f_t>(
primal_uv,
dual_uv,
rc_uv,
host_res->reduced_problem.get_objective_offset(),
presolve_time,
model.get_objective_name(),
model.get_variable_names(),
model.get_row_names());
}

presolve_time = lp_timer.elapsed_time();
CUOPT_LOG_INFO("%s presolve time: %.2fs",
        settings_resolved.presolver == presolver_t::PSLP ? "PSLP" : "Papilo",
        presolve_time);
}
const auto& mps = run_presolve ? host_res->reduced_problem : model;
const i_t n_vars = static_cast<i_t>(mps.get_objective_coefficients().size());
const i_t n_cstr = static_cast<i_t>(mps.get_constraint_lower_bounds().size());
const i_t nnz    = static_cast<i_t>(mps.get_constraint_matrix_values().size());

// A (CSR) — mutable copies for the engine + partitioner consumers below.
std::vector<i_t> h_A_row_offsets = mps.get_constraint_matrix_offsets();
std::vector<i_t> h_A_col_indices = mps.get_constraint_matrix_indices();
std::vector<f_t> h_A_values      = mps.get_constraint_matrix_values();

// ----- 2. Transpose A -> A^T on the host (one-shot CSR transpose) -----
// CSC(A) and CSR(A^T) share the same memory layout, so the CSC produced
// by csr_matrix_t::to_compressed_col IS the CSR of A^T.
// O(nnz + n_vars) counting sort, same as problem_t::compute_transpose.
namespace ds = cuopt::mathematical_optimization;
ds::csr_matrix_t<i_t, f_t> A_csr(n_cstr, n_vars, nnz);
A_csr.row_start = h_A_row_offsets;
A_csr.j         = h_A_col_indices;
A_csr.x         = h_A_values;
ds::csc_matrix_t<i_t, f_t> AT_as_csc(n_vars, n_cstr, nnz);
A_csr.to_compressed_col(AT_as_csc);
std::vector<i_t> h_A_t_row_offsets = std::move(AT_as_csc.col_start);
std::vector<i_t> h_A_t_col_indices = std::move(AT_as_csc.i);
std::vector<f_t> h_A_t_values      = std::move(AT_as_csc.x);

// ----- 3. Partition -----
std::vector<i_t> parts;
partitioner_input_t<i_t, f_t> partition_input;
partition_input.nb_cstr  = n_cstr;
partition_input.nb_vars  = n_vars;
partition_input.nb_parts = distributed_pdlp_num_gpus;

// Resolve which partitioner to use. The public enum is validated at the
// parameter-set layer (int_parameters min/max), so we only need to map
// the three known values to the backend selector.
partitioner_kind_t kind;
switch (settings.multigpu_pdlp_partitioner) {
case multigpu_pdlp_partitioner_t::Auto:
kind = (distributed_pdlp_num_gpus == 1) ? partitioner_kind_t::RoundRobin
                                   : partitioner_kind_t::KaMinPar;
break;
case multigpu_pdlp_partitioner_t::KaMinPar: kind = partitioner_kind_t::KaMinPar; break;
case multigpu_pdlp_partitioner_t::RoundRobin: kind = partitioner_kind_t::RoundRobin; break;
default:
cuopt_expects(false,
         error_type_t::ValidationError,
         "Unknown multigpu_pdlp_partitioner value %d",
         static_cast<int>(settings.multigpu_pdlp_partitioner));
kind = partitioner_kind_t::RoundRobin;  // unreachable; silences -Wmaybe-uninitialized
}
// csr_host_view_t members are std::span<const i_t>, an owning
// std::vector<i_t> converts implicitly.
partition_input.A.row_offsets   = h_A_row_offsets;
partition_input.A.col_indices   = h_A_col_indices;
partition_input.A.num_rows      = n_cstr;
partition_input.A.num_cols      = n_vars;
partition_input.A_t.row_offsets = h_A_t_row_offsets;
partition_input.A_t.col_indices = h_A_t_col_indices;
partition_input.A_t.num_rows    = n_vars;
partition_input.A_t.num_cols    = n_cstr;
// 0 => KaMinPar auto-detects and uses all hardware threads.
partition_input.nb_threads = 0;
CUOPT_LOG_TRACE(
"Partitioning %d constraints + %d variables into %d part(s) using the %s "
"partitioner",
n_cstr,
n_vars,
distributed_pdlp_num_gpus,
(kind == partitioner_kind_t::RoundRobin) ? "round_robin"
: (kind == partitioner_kind_t::KaMinPar) ? "kaminpar"
                                 : "unknown");
auto partitioner = make_partitioner<i_t, f_t>(kind);
parts            = partitioner->partition(partition_input);
}


return 0;
}