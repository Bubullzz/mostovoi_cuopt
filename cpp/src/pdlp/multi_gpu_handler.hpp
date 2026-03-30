#pragma once

#include <nccl.h>
#include <stdexcept>
#include <string>
#include <vector>

#define RAFT_NCCL_TRY(call)                                                       \
  do {                                                                            \
    ncclResult_t const status = (call);                                            \
    if (status != ncclSuccess) {                                                  \
      throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(status)); \
    }                                                                             \
  } while (0)
#include <cusparse_v2.h>
#include <mip_heuristics/problem/problem.cuh>
#include "rmm/device_uvector.hpp"

#include <utilities/event_handler.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class multi_gpu_handler_t {
    public:
        void spmv_A_x(cusparseConstDnVecDescr_t vecX, cusparseDnVecDescr_t vecY);

        /// Debug: print each rank's sub-matrix (host-copies from device)
        void print_sub_matrices() const;

        // Primary constructor - takes raw CSR data (host vectors)
        multi_gpu_handler_t(i_t n_constraints,
                            i_t n_variables,
                            const std::vector<i_t>& h_offsets,
                            const std::vector<i_t>& h_indices,
                            const std::vector<f_t>& h_coefficients,
                            rmm::cuda_stream_view base_stream);

        // Delegating constructor from problem_t
        multi_gpu_handler_t(const problem_t<i_t, f_t>& op_problem);
        std::vector<cudaStream_t>               streams;
        void set_alpha_beta(f_t alpha, f_t beta);
        f_t alpha_h = 1.0;
        f_t beta_h = 0.0;
        ~multi_gpu_handler_t();

    private:
        int nbDevice;
        bool is_test = false;
        // The rank that owns all the pdlp single-gpu data
        int base_rank;

        rmm::cuda_stream_view base_stream;
        std::vector<int> devs;
        std::vector<ncclComm_t> comms;
        size_t rows_per_matrix;

        size_t nb_A_rows;
        size_t nb_A_cols;
        std::vector<cusparseSpMatDescr_t> sub_mat_descriptors;          
        std::vector<void*> external_buffers;
        std::vector<rmm::device_uvector<i_t>> all_offsets;
        std::vector<rmm::device_uvector<i_t>> all_indices;              
        std::vector<rmm::device_uvector<f_t>> all_coefficients;
        std::vector<cusparseHandle_t>           handles;

        std::vector<rmm::device_uvector<f_t>> all_vecX_buf;
        std::vector<rmm::device_uvector<f_t>> all_vecY_buf;
        std::vector<cusparseDnVecDescr_t>     all_vecX;
        std::vector<cusparseDnVecDescr_t>     all_vecY;
        std::vector<rmm::device_scalar<f_t>> all_alpha;
        std::vector<rmm::device_scalar<f_t>> all_beta;

        std::vector<std::unique_ptr<event_handler_t>> done_events;
        event_handler_t start_spmv_event;
};

}