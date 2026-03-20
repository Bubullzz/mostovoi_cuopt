#include <cuda_runtime_api.h>
#include <algorithm>
#include <cassert>
#include <pdlp/multi_gpu_handler.hpp>
#include "cusparse.h"
#include <thrust/host_vector.h>
#include <cuda_runtime.h>

#include <nccl.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/util/cudart_utils.hpp>
#include <utilities/copy_helpers.hpp>

namespace cuopt::linear_programming::detail {

template class multi_gpu_handler_t<int, double>;

template <typename i_t, typename f_t>
multi_gpu_handler_t<i_t, f_t>::multi_gpu_handler_t(const problem_t<i_t, f_t>& op_problem)
  : multi_gpu_handler_t(op_problem.n_constraints,
                        op_problem.n_variables,
                        cuopt::host_copy(op_problem.offsets, op_problem.handle_ptr->get_stream()),
                        cuopt::host_copy(op_problem.variables, op_problem.handle_ptr->get_stream()),
                        cuopt::host_copy(op_problem.coefficients, op_problem.handle_ptr->get_stream()))
{}

template <typename i_t, typename f_t>
multi_gpu_handler_t<i_t, f_t>::multi_gpu_handler_t(
    i_t n_constraints,
    i_t n_variables,
    const std::vector<i_t>& h_offsets,
    const std::vector<i_t>& h_indices,
    const std::vector<f_t>& h_coefficients)
  : sub_mat_descriptors{},
    all_offsets{},
    all_indices{},
    all_coefficients{}
{
    cudaGetDevice(&base_rank);
    cudaGetDeviceCount(&nbDevice);
    if (is_test)
    {
        std::cout << "Running in test mode" << std::endl;
        nbDevice = 4;  // Arbitrary
        std::cout << "Number of dummy devices: " << nbDevice << std::endl;
        devs.resize(nbDevice);
        std::fill(devs.begin(), devs.end(), 0);
    }
    else
    {
        std::cout << "Running in production mode" << std::endl;
        std::cout << "Number of devices: " << nbDevice << std::endl;
        devs.resize(nbDevice);
        std::iota(devs.begin(), devs.end(), 0);
    }
    comms.resize(nbDevice);
    ncclCommInitAll(comms.data(), nbDevice, devs.data());

    sub_mat_descriptors.resize(nbDevice);
    external_buffers.resize(nbDevice);
    streams.resize(nbDevice);
    handles.resize(nbDevice);
    all_offsets.reserve(nbDevice);
    all_indices.reserve(nbDevice);
    all_coefficients.reserve(nbDevice);
    all_vecX_buf.reserve(nbDevice);
    all_vecY_buf.reserve(nbDevice);
    all_vecX.resize(nbDevice);
    all_vecY.resize(nbDevice);

    rows_per_matrix = ((n_constraints - 1) / nbDevice) + 1;
    nb_A_rows       = n_constraints;
    nb_A_cols       = n_variables;

    // Dispatch the matrix
    for (int rank = 0; rank < nbDevice; rank++)
    {
        if (!is_test) cudaSetDevice(rank);
        cudaStreamCreate(&streams[rank]);
        cusparseCreate(&handles[rank]);

        cudaStream_t stream   = streams[rank];
        cusparseHandle_t handle = handles[rank];
        cusparseSetStream(handle, stream);

        int start_row_index = rows_per_matrix * rank;
        int end_row_index =
            std::min(int(h_offsets.size() - 1), int(rows_per_matrix * (rank + 1)));

        int start_row = h_offsets[start_row_index];
        int end_row   = h_offsets[end_row_index];
        int nb_values = end_row - start_row;

        all_offsets.emplace_back(rows_per_matrix + 1, stream);
        all_indices.emplace_back(nb_values, stream);
        all_coefficients.emplace_back(nb_values, stream);
        all_vecX_buf.emplace_back(n_variables, stream);
        all_vecY_buf.emplace_back(rows_per_matrix, stream);

        RAFT_CUSPARSE_TRY(
            cusparseCreateDnVec(&all_vecX[rank], n_variables, all_vecX_buf[rank].data(), CUDA_R_64F));
        RAFT_CUSPARSE_TRY(
            cusparseCreateDnVec(&all_vecY[rank], rows_per_matrix, all_vecY_buf[rank].data(), CUDA_R_64F));

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
        RAFT_CUDA_TRY(cudaMemcpy(all_offsets[rank].data(),
                                 local_offsets.data(),
                                 (rows_per_matrix + 1) * sizeof(int),
                                 cudaMemcpyHostToDevice));

        // Indices and Coefficients
        std::vector<int> local_indices(nb_values);
        std::copy(h_indices.begin() + start_row, h_indices.begin() + end_row, local_indices.begin());

        std::vector<f_t> local_coeffs(nb_values);
        std::copy(h_coefficients.begin() + start_row,
                  h_coefficients.begin() + end_row,
                  local_coeffs.begin());

        RAFT_CUDA_TRY(cudaMemcpy(all_indices[rank].data(),
                                 local_indices.data(),
                                 nb_values * sizeof(i_t),
                                 cudaMemcpyHostToDevice));
        RAFT_CUDA_TRY(cudaMemcpy(all_coefficients[rank].data(),
                                 local_coeffs.data(),
                                 nb_values * sizeof(f_t),
                                 cudaMemcpyHostToDevice));

        // Now creating the actual cuSparse CSR
        cusparseSpMatDescr_t* curr_mat_descr = &(sub_mat_descriptors[rank]);
        int rows = rows_per_matrix;
        int cols = n_variables;
        int nnz  = end_row - start_row;
        cusparseCreateCsr(curr_mat_descr,
                          rows,
                          cols,
                          nnz,
                          all_offsets[rank].data(),
                          all_indices[rank].data(),
                          all_coefficients[rank].data(),
                          CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO,
                          CUDA_R_64F);

        // Preprocessing all of them
        size_t buffer_size = 0;
        double dummy_alpha = 2.0;
        double dummy_beta  = 3.0;
        rmm::device_uvector<f_t> dummy_x(cols, stream);
        rmm::device_uvector<f_t> dummy_y(rows, stream);

        cusparseDnVecDescr_t vecX, vecY;
        RAFT_CUSPARSE_TRY(cusparseCreateDnVec(&vecX, cols, dummy_x.data(), CUDA_R_64F));
        RAFT_CUSPARSE_TRY(cusparseCreateDnVec(&vecY, rows, dummy_y.data(), CUDA_R_64F));

        cusparseSpMV_bufferSize(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &dummy_alpha,
                                *curr_mat_descr,
                                vecX,
                                &dummy_beta,
                                vecY,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_ALG_DEFAULT,
                                &buffer_size);

        cudaMalloc(&(external_buffers[rank]), buffer_size);

        cusparseSpMV_preprocess(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &dummy_alpha,
                                *curr_mat_descr,
                                vecX,
                                &dummy_beta,
                                vecY,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_ALG_DEFAULT,
                                external_buffers[rank]);
    }

    // CSR validity checks for each submatrix (mirrors check_csr_representation)
    for (int rank = 0; rank < nbDevice; rank++)
    {
        if (!is_test) cudaSetDevice(rank);
        rmm::cuda_stream_view stream_view(streams[rank]);
        auto h_offsets_rank  = cuopt::host_copy(all_offsets[rank], stream_view);
        auto h_indices_rank  = cuopt::host_copy(all_indices[rank], stream_view);

        assert(h_indices_rank.size() == all_coefficients[rank].size() &&
               "A_index and A_values must have same sizes.");
        assert(h_offsets_rank[0] == 0 && "A_offsets first value should be 0.");
        assert(std::is_sorted(h_offsets_rank.begin(), h_offsets_rank.end()) &&
               "A_offsets values must be in increasing order.");
        assert(std::all_of(h_indices_rank.begin(),
                          h_indices_rank.end(),
                          [n_variables](i_t j) { return j >= 0 && j < n_variables; }) &&
               "A_indices values must be in [0, n_variables).");
    }

    cudaSetDevice(base_rank);
}

template <typename i_t, typename f_t>
void multi_gpu_handler_t<i_t, f_t>::spmv_A_x(double* alpha, cusparseConstDnVecDescr_t vecX, double *beta, cusparseDnVecDescr_t vecY)
{
    // Assuming vectors/computing is owned by Device(0)
    if (!is_test)
        cudaSetDevice(base_rank); // This call should be useless but eh


    int64_t x_size = 0, y_size = 0;
    const void* x_ptr = nullptr;
    void*       y_ptr = nullptr;
    cudaDataType_t not_null_type = CUDA_R_8I; // just here for no nullptr prblm
    RAFT_CUSPARSE_TRY(cusparseConstDnVecGet(vecX, &x_size, &x_ptr, &not_null_type));
    RAFT_CUSPARSE_TRY(cusparseDnVecGet(vecY, &y_size, &y_ptr, &not_null_type));

    // Optional sanity checks
    assert(x_size == nb_A_cols);
    assert(y_size == nb_A_rows);
    ncclGroupStart();
    // Broadcast VecX and VecY to all devices
    for (int rank = 0; rank < nbDevice; rank++)
    {
        // Vecx.data() is used only if we are on root
        cudaSetDevice(devs[rank]);
        ncclBroadcast(x_ptr, all_vecX_buf[rank].data(), nb_A_cols, ncclFloat64, base_rank, comms[rank], streams[rank]);

        ncclScatter(y_ptr, all_vecY_buf[rank].data(), rows_per_matrix, ncclFloat64, base_rank, comms[rank], streams[rank]);
    }
    ncclGroupEnd();

    // Perform SpMV on each device
    for (int rank = 0; rank < nbDevice; rank++)
    {
        cudaSetDevice(devs[rank]);
        cusparseSpMV(handles[rank],
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            alpha,
            sub_mat_descriptors[rank],
            all_vecX[rank],
            beta,
            all_vecY[rank],
            CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            external_buffers[rank]);
    }

    ncclGroupStart();
    for (int rank = 0; rank < nbDevice; rank++){
        cudaSetDevice(devs[rank]);
        ncclGather(all_vecY_buf[rank].data(), y_ptr, rows_per_matrix, ncclFloat64, base_rank, comms[rank], streams[rank]);
    }
    ncclGroupEnd();
}
}
