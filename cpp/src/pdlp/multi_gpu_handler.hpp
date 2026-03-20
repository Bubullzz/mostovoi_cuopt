#pragma once

#include <nccl.h>
#include <cusparse_v2.h>
#include <mip_heuristics/problem/problem.cuh>
#include "rmm/device_uvector.hpp"


namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class multi_gpu_handler_t {
    public:
        void spmv_A_x(double* alpha, cusparseConstDnVecDescr_t vecX, double* beta, cusparseDnVecDescr_t vecY);
    
        multi_gpu_handler_t(const problem_t<i_t, f_t>& op_problem);
    private:
        int nbDevice;
        bool is_test;
        // The rank that owns all the pdlp single-gpu data
        int base_rank;

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
        std::vector<cudaStream_t>               streams;
        std::vector<cusparseHandle_t>           handles;

        std::vector<rmm::device_uvector<f_t>> all_vecX_buf;
        std::vector<rmm::device_uvector<f_t>> all_vecY_buf;
        std::vector<cusparseDnVecDescr_t>     all_vecX;
        std::vector<cusparseDnVecDescr_t>     all_vecY;
};

}