#include <pdlp/multi_gpu_handler.hpp>
#include <algorithm>
#include <cassert>
#include <string>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <nccl.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/util/cudart_utils.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/event_handler.cuh>

#define RAFT_NCCL_TRY_NO_THROW(call)                                              \
  do {                                                                            \
    ncclResult_t const status = (call);                                           \
    if (status != ncclSuccess) {                                                  \
      std::fprintf(stderr, "NCCL error at %s:%d: %s\n",                           \
                   __FILE__, __LINE__, ncclGetErrorString(status));               \
    }                                                                             \
  } while (0)

namespace cuopt::linear_programming::detail {

template class multi_gpu_handler_t<int, double>;
template class multi_gpu_handler_t<int, float>;

template <typename i_t, typename f_t>
sub_matrix_t<i_t, f_t>::sub_matrix_t(
    int rank, 
    int device_id,
    const std::vector<i_t>& local_offsets,
    const std::vector<i_t>& local_indices,
    const std::vector<f_t>& local_coeffs,
    size_t n_variables,
    size_t local_n_constraints,
    size_t n_values
)
  : rank(rank),
    device_id(device_id),
    stream(), // Expecting to be set on the right device
    handle(stream),
    offsets(local_n_constraints + 1, stream),
    indices(n_values, stream),
    coefficients(n_values, stream),
    mat_descriptor(),
    vecX_buf(n_variables, stream),
    vecY_buf(local_n_constraints, stream),
    vecX{},
    vecY{},
    alpha(f_t{1.0}, stream),
    beta(f_t{0}, stream),
    external_buffer(0, stream),
    done_event()
{
    int current_device;
    cudaGetDevice(&current_device);
    assert(current_device == device_id && "right device should be set before calling ctr.");

    assert(local_offsets.size() == local_n_constraints + 1 && "local_offsets size should be local_n_constraints + 1.");
    assert(local_indices.size() == n_values && "local_indices size should be n_values.");
    assert(local_coeffs.size() == n_values && "local_coeffs size should be n_values.");

    cuopt::device_copy(offsets, local_offsets, stream.view());
    cuopt::device_copy(indices, local_indices, stream.view());
    cuopt::device_copy(coefficients, local_coeffs, stream.view());
    mat_descriptor.create(local_n_constraints, n_variables, n_values, offsets.data(), indices.data(), coefficients.data());
    vecX.create(n_variables, vecX_buf.data());
    vecY.create(local_n_constraints, vecY_buf.data());

    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
        handle.get_cusparse_handle(),
        CUSPARSE_POINTER_MODE_DEVICE,
        stream.view()));

        size_t buffer_size = 0;
        RAFT_CUSPARSE_TRY(
          raft::sparse::detail::cusparsespmv_buffersize(handle.get_cusparse_handle(),
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        alpha.data(),
                                                        mat_descriptor,
                                                        vecX,
                                                        beta.data(),
                                                        vecY,
                                                        CUSPARSE_SPMV_ALG_DEFAULT,
                                                        &buffer_size,
                                                        stream.view()));

    external_buffer = rmm::device_buffer(buffer_size, stream);
    
    auto float_type = get_cuda_dtype<f_t>();
    RAFT_CUSPARSE_TRY(cusparseSpMV_preprocess(handle.get_cusparse_handle(),
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              alpha.data(),
                                              mat_descriptor,
                                              vecX,
                                              beta.data(),
                                              vecY,
                                              float_type,
                                              CUSPARSE_SPMV_ALG_DEFAULT,
                                              external_buffer.data()));
}

template <typename i_t, typename f_t>
multi_gpu_handler_t<i_t, f_t>::multi_gpu_handler_t(const problem_t<i_t, f_t>& op_problem)
  : multi_gpu_handler_t(op_problem.n_constraints,
                        op_problem.n_variables,
                        cuopt::host_copy(op_problem.offsets, op_problem.handle_ptr->get_stream()),
                        cuopt::host_copy(op_problem.variables, op_problem.handle_ptr->get_stream()),
                        cuopt::host_copy(op_problem.coefficients, op_problem.handle_ptr->get_stream()),
                        cuopt::host_copy(op_problem.reverse_offsets, op_problem.handle_ptr->get_stream()),
                        cuopt::host_copy(op_problem.reverse_constraints, op_problem.handle_ptr->get_stream()),
                        cuopt::host_copy(op_problem.reverse_coefficients, op_problem.handle_ptr->get_stream()),
                        op_problem.handle_ptr->get_stream())
{}

template <typename i_t, typename f_t>
void multi_gpu_handler_t<i_t, f_t>::create_sub_mat(
    int rank,
    size_t rows_per_matrix,
    size_t n_variables,
    const std::vector<i_t>& h_offsets,
    const std::vector<i_t>& h_indices,
    const std::vector<f_t>& h_coefficients,
    std::vector<sub_matrix_owner_t<i_t, f_t>>& mat_vec
)
{
    int start_row_index = rows_per_matrix * rank;
    int end_row_index =
        std::min(int(h_offsets.size() - 1), int(rows_per_matrix * (rank + 1)));

    int start_row = h_offsets[start_row_index];
    int end_row   = h_offsets[end_row_index];
    int nb_values = end_row - start_row;

    // Offsets
    size_t n_copied = end_row_index - start_row_index + 1;
    std::vector<int> local_offsets(rows_per_matrix + 1);
    std::copy(h_offsets.begin() + start_row_index,
              h_offsets.begin() + end_row_index + 1,
              local_offsets.begin());
    // Handle case where len(offsets) % rows_per_matrix != 0 so last gpu needs padding
    if (n_copied < rows_per_matrix + 1)
    {
        assert(rank == nbDevice - 1);
        int last_val = h_offsets[end_row_index];
        std::fill(local_offsets.begin() + n_copied, local_offsets.end(), last_val);
    }
    int first_entry = local_offsets[0];
    std::transform(local_offsets.begin(),
                   local_offsets.end(),
                   local_offsets.begin(),
                   [first_entry](int x) { return x - first_entry; });

    // Indices and coefficients
    std::vector<int> local_indices(nb_values);
    std::copy(h_indices.begin() + start_row, h_indices.begin() + end_row, local_indices.begin());

    std::vector<f_t> local_coeffs(nb_values);
    std::copy(h_coefficients.begin() + start_row,
              h_coefficients.begin() + end_row,
              local_coeffs.begin());

    mat_vec.emplace_back(make_sub_matrix<i_t, f_t>(
    rank,
    devs[rank],
    local_offsets,
    local_indices,
    local_coeffs,
    n_variables,
    rows_per_matrix,
    nb_values));
}


template <typename i_t, typename f_t>
multi_gpu_handler_t<i_t, f_t>::multi_gpu_handler_t(
    i_t n_constraints,
    i_t n_variables,
    const std::vector<i_t>& h_offsets,
    const std::vector<i_t>& h_indices,
    const std::vector<f_t>& h_coefficients,
    const std::vector<i_t>& h_reverse_offsets,
    const std::vector<i_t>& h_reverse_constraints,
    const std::vector<f_t>& h_reverse_coefficients,
    rmm::cuda_stream_view base_stream)
  : base_stream(base_stream)
{
    cudaGetDevice(&base_rank);
    cudaGetDeviceCount(&nbDevice);
    std::cout << "Running in production mode" << std::endl;
    std::cout << "Number of devices: " << nbDevice << std::endl;
    std::cout << "Base rank: " << base_rank << std::endl;
    nbDevice = std::min(nbDevice, n_constraints);
    devs.resize(nbDevice);
    std::iota(devs.begin(), devs.end(), 0);
    comms.resize(nbDevice);

    RAFT_NCCL_TRY(ncclCommInitAll(comms.data(), nbDevice, devs.data()));

    nb_A_rows       = n_constraints;
    nb_A_cols       = n_variables;
    rows_per_matrix_A = ((nb_A_rows - 1) / nbDevice) + 1;

    nb_A_t_rows = nb_A_cols;
    nb_A_t_cols = nb_A_rows;
    rows_per_matrix_A_t = ((nb_A_t_rows - 1) / nbDevice) + 1;

    sub_matrices_A.reserve(nbDevice);
    sub_matrices_A_t.reserve(nbDevice);

    for (int rank = 0; rank < nbDevice; rank++)
    {
        create_sub_mat(rank, rows_per_matrix_A, nb_A_cols, h_offsets, h_indices, h_coefficients, sub_matrices_A);
        create_sub_mat(rank, rows_per_matrix_A_t, nb_A_t_cols, h_reverse_offsets, h_reverse_constraints, h_reverse_coefficients, sub_matrices_A_t);
    }

    // CSR validity checks for each submatrix
    for (int rank = 0; rank < nbDevice; rank++)
    {
        auto& sub = *sub_matrices_A[rank];
        raft::device_setter device_setter(devs[rank]);
        rmm::cuda_stream_view stream_view(sub.stream);
        auto h_offsets_rank = cuopt::host_copy(sub.offsets, stream_view);
        auto h_indices_rank = cuopt::host_copy(sub.indices, stream_view);

        assert(h_indices_rank.size() == sub.coefficients.size() &&
               "A_index and A_values must have same sizes.");
        assert(h_offsets_rank[0] == 0 && "A_offsets first value should be 0.");
        assert(std::is_sorted(h_offsets_rank.begin(), h_offsets_rank.end()) &&
               "A_offsets values must be in increasing order.");
        assert(std::all_of(h_indices_rank.begin(),
                           h_indices_rank.end(),
                           [n_variables](i_t j) { return j >= 0 && j < n_variables; }) &&
               "A_indices values must be in [0, n_variables).");
    }
}

template <typename i_t, typename f_t>
void multi_gpu_handler_t<i_t, f_t>::spmv(cusparseConstDnVecDescr_t vecX,
                                         cusparseDnVecDescr_t vecY,
                                         size_t x_broadcast_size,
                                         size_t y_scatter_size,
                                         std::vector<sub_matrix_owner_t<i_t, f_t>>& sub_matrices)
{
    if (sub_matrices.size() != static_cast<size_t>(nbDevice)) {
        throw std::runtime_error("Requested multi-GPU SpMV for an uninitialized matrix partition.");
    }
    constexpr auto nccl_dtype = get_nccl_dtype<f_t>();
    // Fork base_stream into per-rank streams
    start_spmv_event.record(base_stream);
    for (int rank = 0; rank < nbDevice; rank++) {
        raft::device_setter device_setter(sub_matrices[rank]->device_id);
        start_spmv_event.stream_wait(sub_matrices[rank]->stream.view());
    }

    int64_t x_size = 0, y_size = 0;
    const void* x_ptr = nullptr;
    void*       y_ptr = nullptr;
    cudaDataType_t not_null_type = CUDA_R_8I;
    RAFT_CUSPARSE_TRY(cusparseConstDnVecGet(vecX, &x_size, &x_ptr, &not_null_type));
    RAFT_CUSPARSE_TRY(cusparseDnVecGet(vecY, &y_size, &y_ptr, &not_null_type));

    RAFT_NCCL_TRY(ncclGroupStart());
    for (int rank = 0; rank < nbDevice; rank++)
    {
        auto& sub = *sub_matrices[rank];
        RAFT_NCCL_TRY(ncclBroadcast(
            x_ptr,
            sub.vecX_buf.data(),
            x_broadcast_size,
            nccl_dtype,
            base_rank,
            comms[rank],
            sub.stream.value()));
        RAFT_NCCL_TRY(ncclScatter(
            y_ptr,
            sub.vecY_buf.data(),
            y_scatter_size,
            nccl_dtype,
            base_rank,
            comms[rank],
            sub.stream.value()));
    }
    RAFT_NCCL_TRY(ncclGroupEnd());

    for (int rank = 0; rank < nbDevice; rank++)
    {
        auto& sub = *sub_matrices[rank];
        raft::device_setter device_setter(sub.device_id);
        RAFT_CUSPARSE_TRY(
            raft::sparse::detail::cusparsespmv(sub.handle.get_cusparse_handle(),
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               sub.alpha.data(),
                                               sub.mat_descriptor,
                                               sub.vecX,
                                               sub.beta.data(),
                                               sub.vecY,
                                               CUSPARSE_SPMV_ALG_DEFAULT,
                                               static_cast<f_t*>(sub.external_buffer.data()),
                                               sub.stream.view()));
    }

    RAFT_NCCL_TRY(ncclGroupStart());
    for (int rank = 0; rank < nbDevice; rank++) {
        auto& sub = *sub_matrices[rank];
        RAFT_NCCL_TRY(ncclGather(
            sub.vecY_buf.data(),
            y_ptr,
            y_scatter_size,
            nccl_dtype,
            base_rank,
            comms[rank],
            sub.stream.value()));
    }
    RAFT_NCCL_TRY(ncclGroupEnd());

    for (int rank = 0; rank < nbDevice; ++rank) {
        auto& sub = *sub_matrices[rank];
        raft::device_setter device_setter(sub.device_id);
        sub.done_event.record(sub.stream.view());
    }
    for (int rank = 0; rank < nbDevice; ++rank) {
        sub_matrices[rank]->done_event.stream_wait(base_stream);
    }
}

template <typename i_t, typename f_t>
void multi_gpu_handler_t<i_t, f_t>::spmv_A_x(
    cusparseConstDnVecDescr_t vecX, cusparseDnVecDescr_t vecY)
{
    spmv(vecX, vecY, nb_A_cols, rows_per_matrix_A, sub_matrices_A);
}

template <typename i_t, typename f_t>
void multi_gpu_handler_t<i_t, f_t>::spmv_A_t_y(
    cusparseConstDnVecDescr_t vecX, cusparseDnVecDescr_t vecY)
{
    spmv(vecX, vecY, nb_A_t_cols, rows_per_matrix_A_t, sub_matrices_A_t);
}

template <typename i_t, typename f_t>
multi_gpu_handler_t<i_t, f_t>::~multi_gpu_handler_t()
{
    sub_matrices_A.clear();
    sub_matrices_A_t.clear();
    for (int rank = 0; rank < nbDevice; ++rank) {
        ncclCommDestroy(comms[rank]);
    }
}

template <typename i_t, typename f_t>
void multi_gpu_handler_t<i_t, f_t>::set_alpha_beta(f_t alpha, f_t beta)
{
    for (int rank = 0; rank < nbDevice; ++rank) {
        auto& sub = *sub_matrices_A[rank];
        raft::device_setter device_setter(sub.device_id);
        sub.alpha.set_value_async(alpha, sub.stream.view());
        sub.beta.set_value_async(beta, sub.stream.view());
    }
    for (int rank = 0; rank < nbDevice; ++rank) {
        auto& sub = *sub_matrices_A_t[rank];
        raft::device_setter device_setter(sub.device_id);
        sub.alpha.set_value_async(alpha, sub.stream.view());
        sub.beta.set_value_async(beta, sub.stream.view());
    }
}

template <typename i_t, typename f_t>
void multi_gpu_handler_t<i_t, f_t>::sync_spmv()
{
    for (int rank = 0; rank < nbDevice; rank++) {
        raft::device_setter device_setter(devs[rank]);
        sub_matrices_A[rank]->stream.synchronize();
        sub_matrices_A_t[rank]->stream.synchronize();
    }
}

template <typename i_t, typename f_t>
void multi_gpu_handler_t<i_t, f_t>::print_sub_matrices() const
{
    for (int rank = 0; rank < nbDevice; rank++)
    {
        raft::device_setter device_setter(devs[rank]);

        const auto& sub = *sub_matrices_A[rank];
        rmm::cuda_stream_view stream_view(sub.stream);
        auto h_offsets = cuopt::host_copy(sub.offsets, stream_view);
        auto h_indices = cuopt::host_copy(sub.indices, stream_view);
        auto h_values  = cuopt::host_copy(sub.coefficients, stream_view);

        std::string prefix = "Rank " + std::to_string(rank) + ": ";
        cuopt::print_csr_matrix(static_cast<int>(rows_per_matrix_A),
                               static_cast<int>(nb_A_cols),
                               h_offsets,
                               h_indices,
                               h_values,
                               prefix.c_str());
    }
}

}
