/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Out-of-line definitions of distributed_state_t's primitive helpers (halo
// exchange, allreduce, dot / l2 norm, distributed SpMV) for the multi-process
// (MPI + NCCL) path. One process = one GPU = one rank, so unlike
// multi_gpu_engine_t there is no loop over shards: every helper below runs on
// THIS rank only and relies on the other ranks calling the same helper.
//
// The algorithm-level helpers (scaling, power iteration, ...) that are built on
// top of these are free functions in mp_algorithms.hpp / mp_algorithms.cu.
#include <pdlp/distributed_pdlp/distributed_state.hpp>
#include <pdlp/pdhg.hpp>

#include <cuopt/error.hpp>

#include <raft/core/cublas_macros.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/gather.h>
#include <cub/device/device_transform.cuh>
#include <cuda/std/cmath>

#include <nccl.h>

namespace cuopt::mathematical_optimization::pdlp {

// -------- Halo exchange ----------------------------------------------------
// Vector layout is [owned | halo]. For each peer we
//   1. gather the owned entries that peer needs into a contiguous staging buffer,
//   2. ncclSend that buffer to the peer,
//   3. ncclRecv the peer's staging buffer straight into our halo tail
//      (at x + owned_size + recv_offsets[peer]).
// Sends and recvs are posted inside one NCCL group so they cannot deadlock.
// Zero-length sends/recvs are still posted so both sides always match.
template <typename i_t, typename f_t>
void distributed_state_t<i_t, f_t>::halo_exchange_impl(rmm::device_uvector<f_t>& x,
                                                       halo_axis_t const& axis,
                                                       cuda::stream_ref stream)
{
  cuopt_expects(static_cast<int>(axis.send_buf.size()) == world_size &&
                  static_cast<int>(axis.recv_counts.size()) == world_size,
                error_type_t::RuntimeError,
                "halo_exchange_impl: per-peer halo metadata must have size == world_size");

  // Step 1: gather owned values that each peer needs into per-peer staging.
  for (int peer = 0; peer < world_size; ++peer) {
    if (peer == rank) continue;
    if (axis.send_indices[peer].size() == 0) continue;
    thrust::gather(rmm::exec_policy_nosync(stream),
                   axis.send_indices[peer].begin(),
                   axis.send_indices[peer].end(),
                   x.data(),
                   axis.send_buf[peer].begin());
  }

  // Step 2: matched send / recv with every peer in one NCCL group.
  CUOPT_NCCL_TRY(ncclGroupStart());
  for (int peer = 0; peer < world_size; ++peer) {
    if (peer == rank) continue;
    CUOPT_NCCL_TRY(ncclSend(axis.send_buf[peer].data(),
                            axis.send_buf[peer].size(),
                            nccl_data_type<f_t>(),
                            peer,
                            nccl_comm.get(),
                            stream.get()));
  }
  for (int peer = 0; peer < world_size; ++peer) {
    if (peer == rank) continue;
    f_t* recv_ptr = x.data() + axis.owned_size + axis.recv_offsets[peer];
    CUOPT_NCCL_TRY(ncclRecv(recv_ptr,
                            static_cast<size_t>(axis.recv_counts[peer]),
                            nccl_data_type<f_t>(),
                            peer,
                            nccl_comm.get(),
                            stream.get()));
  }
  CUOPT_NCCL_TRY(ncclGroupEnd());
}

template <typename i_t, typename f_t>
void distributed_state_t<i_t, f_t>::halo_exchange_var(rmm::device_uvector<f_t>& x,
                                                      cuda::stream_ref stream)
{
  halo_exchange_impl(x, var_halo_axis(), stream);
}

template <typename i_t, typename f_t>
void distributed_state_t<i_t, f_t>::halo_exchange_cstr(rmm::device_uvector<f_t>& x,
                                                       cuda::stream_ref stream)
{
  halo_exchange_impl(x, cstr_halo_axis(), stream);
}

// -------- NCCL allreduce (sum, in place) -----------------------------------
template <typename i_t, typename f_t>
void distributed_state_t<i_t, f_t>::allreduce_sum_inplace(raft::device_scalar_view<f_t> scalar,
                                                          cuda::stream_ref stream)
{
  f_t* p = scalar.data_handle();
  CUOPT_NCCL_TRY(ncclAllReduce(
    p, p, /*count=*/1, nccl_data_type<f_t>(), ncclSum, nccl_comm.get(), stream.get()));
}

// -------- Distributed dot ----------------------------------------
// The spans must cover only the OWNED slice of each vector so every global
// entry is counted exactly once across ranks.
template <typename i_t, typename f_t>
void distributed_state_t<i_t, f_t>::distributed_dot(raft::device_span<f_t> a,
                                                    raft::device_span<f_t> b,
                                                    raft::device_scalar_view<f_t> out,
                                                    raft::handle_t const& handle)
{
  cuopt_expects(a.size() == b.size(),
                error_type_t::RuntimeError,
                "distributed_dot: a and b must have equal size");
  const cuda::stream_ref stream = handle.get_stream();

  // Local partial dot (device pointer mode, result stays on device) ...
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(handle.get_cublas_handle(),
                                                  static_cast<int>(a.size()),
                                                  a.data(),
                                                  1,
                                                  b.data(),
                                                  1,
                                                  out.data_handle(),
                                                  stream.get()));
  // ... then sum the partials over all ranks.
  allreduce_sum_inplace(out, stream);
}

template <typename i_t, typename f_t>
void distributed_state_t<i_t, f_t>::distributed_dot(raft::device_span<f_t> a,
                                                    raft::device_span<f_t> b,
                                                    rmm::device_scalar<f_t>& out,
                                                    raft::handle_t const& handle)
{
  distributed_dot(a, b, raft::make_device_scalar_view<f_t>(out.data()), handle);
}

template <typename i_t, typename f_t>
void distributed_state_t<i_t, f_t>::distributed_l2_norm(raft::device_span<f_t> in,
                                                        raft::device_scalar_view<f_t> out,
                                                        raft::handle_t const& handle)
{
  // sqrt is not additive across ranks, so: allreduce(dot(x, x)) first, sqrt after.
  distributed_dot(in, in, out, handle);
  cub::DeviceTransform::Transform(
    out.data_handle(),
    out.data_handle(),
    1,
    [] __device__(f_t x) { return cuda::std::sqrt(x); },
    handle.get_stream().get());
}

template <typename i_t, typename f_t>
void distributed_state_t<i_t, f_t>::distributed_l2_norm(raft::device_span<f_t> in,
                                                        rmm::device_scalar<f_t>& out,
                                                        raft::handle_t const& handle)
{
  distributed_l2_norm(in, raft::make_device_scalar_view<f_t>(out.data()), handle);
}

// -------- Distributed SpMV -------------------------------------------------
// Local A is (owned rows x total cols) and local A^T is (owned cols x total rows),
// so the INPUT of each product needs its halo entries refreshed first; the
// output is complete on the owned slice.
template <typename i_t, typename f_t>
void distributed_state_t<i_t, f_t>::distributed_spmv_At(pdhg_solver_t<i_t, f_t>& pdhg,
                                                        rmm::device_uvector<f_t>& in_buf,
                                                        cusparseDnVecDescr_t in_desc,
                                                        cusparseDnVecDescr_t out_desc,
                                                        cuda::stream_ref stream)
{
  halo_exchange_cstr(in_buf, stream);  // A^T @ y: y is dual (cstr) shaped
  pdhg.spmv_At_into(in_desc, out_desc);
}

template <typename i_t, typename f_t>
void distributed_state_t<i_t, f_t>::distributed_spmv_A(pdhg_solver_t<i_t, f_t>& pdhg,
                                                       rmm::device_uvector<f_t>& in_buf,
                                                       cusparseDnVecDescr_t in_desc,
                                                       cusparseDnVecDescr_t out_desc,
                                                       cuda::stream_ref stream)
{
  halo_exchange_var(in_buf, stream);  // A @ x: x is primal (var) shaped
  pdhg.spmv_A_into(in_desc, out_desc);
}

// -------- High-level: A @ x and A^T @ y on the pdhg's own buffers ----------
template <typename i_t, typename f_t>
void distributed_state_t<i_t, f_t>::distributed_compute_A_x(pdhg_solver_t<i_t, f_t>& pdhg,
                                                            cuda::stream_ref stream)
{
  halo_exchange_var(pdhg.get_reflected_primal(), stream);
  pdhg.spmvop_A_x();
}

template <typename i_t, typename f_t>
void distributed_state_t<i_t, f_t>::distributed_compute_At_y(pdhg_solver_t<i_t, f_t>& pdhg,
                                                             cuda::stream_ref stream)
{
  halo_exchange_cstr(pdhg.get_dual_solution(), stream);
  pdhg.spmvop_At_y();
}

template struct distributed_state_t<int, double>;
template struct distributed_state_t<int, float>;

}  // namespace cuopt::mathematical_optimization::pdlp
