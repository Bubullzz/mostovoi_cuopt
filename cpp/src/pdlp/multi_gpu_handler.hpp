#pragma once

#include <nccl.h>
#include <cusparse_v2.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mip_heuristics/problem/problem.cuh>
#include <pdlp/cusparse_view.hpp>
#include <utilities/event_handler.cuh>

#include <raft/core/device_setter.hpp>
#include <raft/core/handle.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <type_traits>


#define RAFT_NCCL_TRY(call)                                                       \
  do {                                                                            \
    ncclResult_t const status = (call);                                            \
    if (status != ncclSuccess) {                                                  \
      throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(status)); \
    }                                                                             \
  } while (0)


namespace cuopt::linear_programming::detail {

template <typename f_t>
constexpr ncclDataType_t get_nccl_dtype()
{
  if constexpr (std::is_same_v<f_t, float>) {
    return ncclFloat32;
  } else {
    return ncclFloat64;
  }
}

template <typename f_t>
constexpr cudaDataType get_cuda_dtype()
{
  if constexpr (std::is_same_v<f_t, float>) {
    return CUDA_R_32F;
  } else {
    return CUDA_R_64F;
  }
}

// Owns all per-rank GPU resources for one row-partition of the sparse matrix A.
template <typename i_t, typename f_t>
struct sub_matrix_t {
    sub_matrix_t() = delete;
    sub_matrix_t(const sub_matrix_t&) = delete;
    sub_matrix_t& operator=(const sub_matrix_t&) = delete;
    sub_matrix_t(sub_matrix_t&&) noexcept = delete;
    sub_matrix_t& operator=(sub_matrix_t&&) noexcept = delete;
    ~sub_matrix_t() = default;
    
      sub_matrix_t(
        int rank, 
        int device_id,
        std::vector<i_t> local_offsets,
        std::vector<i_t> local_indices,
        std::vector<f_t> local_coeffs,
        size_t n_variables,
        size_t local_n_constraints,
        size_t n_values
    );
    int              rank;
    int              device_id;
    rmm::cuda_stream     stream;  
    raft::handle_t handle;

    rmm::device_uvector<i_t> offsets;
    rmm::device_uvector<i_t> indices;
    rmm::device_uvector<f_t> coefficients;

    cusparse_sp_mat_descr_wrapper_t<i_t, f_t> mat_descriptor;

    rmm::device_uvector<f_t> vecX_buf;
    rmm::device_uvector<f_t> vecY_buf;
    cusparse_dn_vec_descr_wrapper_t<f_t> vecX;
    cusparse_dn_vec_descr_wrapper_t<f_t> vecY;
    rmm::device_scalar<f_t>  alpha;
    rmm::device_scalar<f_t>  beta;

    rmm::device_buffer external_buffer;
    event_handler_t done_event;
  };

// Custom deleter to ensure clean Device management
template <typename i_t, typename f_t>
struct sub_matrix_deleter_t {
  int device_id{0};
  void operator()(sub_matrix_t<i_t, f_t>* ptr) const noexcept
  {
    if (ptr == nullptr) { return; }
    raft::device_setter device_setter(device_id);
    delete ptr;
  }
};

template <typename i_t, typename f_t>
using sub_matrix_owner_t =
  std::unique_ptr<sub_matrix_t<i_t, f_t>, sub_matrix_deleter_t<i_t, f_t>>;

template <typename i_t, typename f_t>
sub_matrix_owner_t<i_t, f_t> make_sub_matrix(
    int rank,
    int device_id,
    std::vector<i_t> local_offsets,
    std::vector<i_t> local_indices,
    std::vector<f_t> local_coeffs,
    size_t n_variables,
    size_t local_n_constraints,
    size_t n_values)
{
  raft::device_setter device_setter(device_id);
  return sub_matrix_owner_t<i_t, f_t>(
      new sub_matrix_t<i_t, f_t>(rank,
                                 device_id,
                                 std::move(local_offsets),
                                 std::move(local_indices),
                                 std::move(local_coeffs),
                                 n_variables,
                                 local_n_constraints,
                                 n_values),
      sub_matrix_deleter_t<i_t, f_t>{device_id});
}

template <typename i_t, typename f_t>
class multi_gpu_handler_t {
    public:
        void spmv_A_x(cusparseConstDnVecDescr_t vecX, cusparseDnVecDescr_t vecY);
        void sync_spmv();

        void print_sub_matrices() const;

        multi_gpu_handler_t(i_t n_constraints,
                            i_t n_variables,
                            const std::vector<i_t>& h_offsets,
                            const std::vector<i_t>& h_indices,
                            const std::vector<f_t>& h_coefficients,
                            rmm::cuda_stream_view base_stream);

        // Delegating constructor from problem_t
        multi_gpu_handler_t(const problem_t<i_t, f_t>& op_problem);

        void set_alpha_beta(f_t alpha, f_t beta);
        ~multi_gpu_handler_t();
        
    private:
        int                   nbDevice{0};
        int                   base_rank{0}; // The rank that owns the single-gpu Data

        rmm::cuda_stream_view base_stream;
        std::vector<int>      devs;
        std::vector<ncclComm_t> comms;
        size_t                rows_per_matrix{0};
        size_t                nb_A_rows{0};
        size_t                nb_A_cols{0};
        event_handler_t       start_spmv_event;
        std::vector<sub_matrix_owner_t<i_t, f_t>> sub_matrices;
      };

}
