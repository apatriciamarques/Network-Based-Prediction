#include "dot_prod.hpp"
#include "pipeline_utils.hpp"
#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---------- Error checking ----------
#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::cerr << "CUDA Error: " << cudaGetErrorString(_e) << " at "          \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      std::abort();                                                            \
    }                                                                          \
  } while (0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(expr)                                                     \
  do {                                                                         \
    cublasStatus_t _s = (expr);                                                \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                         \
      std::cerr << "cuBLAS Error: " << _s << " at " << __FILE__ << ":"         \
                << __LINE__ << std::endl;                                      \
      std::abort();                                                            \
    }                                                                          \
  } while (0)
#endif

// ---------- CUDA kernels ----------
// Note: supports different strides for i and j batches
__global__ void compute_dot_products_kernel(
    const float *batch_i_data, const float *batch_j_data,
    const int *batch_i_sizes, const int *batch_j_sizes, float *dot_products,
    int batch_i_size, int batch_j_size, int max_vec_size_i, int max_vec_size_j,
    bool is_diagonal_block) {
  int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_j = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx_i >= batch_i_size || idx_j >= batch_j_size)
    return;

  // Upper triangle only for diagonal blocks
  if (is_diagonal_block && idx_j < idx_i)
    return;

  int len_i = batch_i_sizes[idx_i];
  int len_j = batch_j_sizes[idx_j];
  if (len_i == 0 || len_j == 0)
    return;

  int min_len = len_i < len_j ? len_i : len_j;

  const float *vec_i = batch_i_data + idx_i * max_vec_size_i;
  const float *vec_j = batch_j_data + idx_j * max_vec_size_j;

  float acc = 0.0f;
  // simple loop – you can unroll/tile later if needed
  for (int k = 0; k < min_len; ++k) {
    acc += vec_i[k] * vec_j[k];
  }

  dot_products[idx_i * batch_j_size + idx_j] = acc;
}

__global__ void compute_kernel_values(
    const float *dot_products, float *kernel_values,
    const float *inv_sqrt_factors_i, // only depends on i (matches CPU impl)
    int batch_i_size, int batch_j_size, int nPower, bool is_diagonal_block) {
  int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_j = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx_i >= batch_i_size || idx_j >= batch_j_size)
    return;
  if (is_diagonal_block && idx_j < idx_i)
    return;

  int linear_idx = idx_i * batch_j_size + idx_j;

  float norm_dot = dot_products[linear_idx] * inv_sqrt_factors_i[idx_i];

  // same as CPU: raise (dot * inv_sqrt_i)^(2*nPower)
  // powf handles 0/neg inputs (if any) consistently
  float val = powf(norm_dot, 2.0f * nPower);
  kernel_values[linear_idx] = val;
}

// ---------- Small GPU buffer manager that can grow as needed ----------
struct GPUWorkBuffers {
  // pointers
  float *d_batch_i = nullptr, *d_batch_j = nullptr;
  int *d_sizes_i = nullptr, *d_sizes_j = nullptr;
  float *d_inv_sqrt_i = nullptr;
  float *d_dot = nullptr, *d_out = nullptr;

  // capacities in elements
  size_t cap_batch_i_elems = 0; // floats
  size_t cap_batch_j_elems = 0; // floats
  size_t cap_sizes_i = 0;       // ints
  size_t cap_sizes_j = 0;       // ints
  size_t cap_inv_sqrt_i = 0;    // floats
  size_t cap_dot = 0;           // floats
  size_t cap_out = 0;           // floats

  cudaStream_t stream = nullptr;

  GPUWorkBuffers() { CUDA_CHECK(cudaStreamCreate(&stream)); }
  ~GPUWorkBuffers() {
    if (d_batch_i)
      CUDA_CHECK(cudaFree(d_batch_i));
    if (d_batch_j)
      CUDA_CHECK(cudaFree(d_batch_j));
    if (d_sizes_i)
      CUDA_CHECK(cudaFree(d_sizes_i));
    if (d_sizes_j)
      CUDA_CHECK(cudaFree(d_sizes_j));
    if (d_inv_sqrt_i)
      CUDA_CHECK(cudaFree(d_inv_sqrt_i));
    if (d_dot)
      CUDA_CHECK(cudaFree(d_dot));
    if (d_out)
      CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void ensure_floats(float **ptr, size_t &cap, size_t need) {
    if (need <= cap)
      return;
    if (*ptr)
      CUDA_CHECK(cudaFree(*ptr));
    CUDA_CHECK(cudaMalloc(ptr, need * sizeof(float)));
    cap = need;
  }
  void ensure_ints(int **ptr, size_t &cap, size_t need) {
    if (need <= cap)
      return;
    if (*ptr)
      CUDA_CHECK(cudaFree(*ptr));
    CUDA_CHECK(cudaMalloc(ptr, need * sizeof(int)));
    cap = need;
  }
};

// ---------- Main refactored function ----------
Matrix compute_kernel_block_batched_gpu(
    int c, int i, int s, const std::vector<FirstHopState> &firstHopStates,
    const std::vector<int> &node_ids, const GraphData &data,
    const std::vector<int> &nodeDegrees, const std::vector<int> &nodeDegreesC1,
    int nPower, int batch_size) {
  std::cout << "Compute_kernel_block_batched_gpu (refactored): c=" << c
            << ", i=" << i << std::endl;

  // Keep a conservative default for GPU; you can tune this
  if (batch_size <= 0) {
    batch_size = (c == 1 ? 7000 : 80);
  }

  const int n_nodes = static_cast<int>(node_ids.size());
  Matrix K_block(n_nodes, std::vector<dtype>(n_nodes, 0.0));

  // Per-iteration buffers (reused, grown on demand)
  GPUWorkBuffers buf;

  using clock = std::chrono::high_resolution_clock;
  using duration_ms = std::chrono::duration<double, std::milli>;

  double total_gpu_batch_ms = 0.0;
  double total_outer_loop_ms = 0.0;
  int gpu_batch_count = 0;
  int outer_loop_count = 0;

  // Outer loop over rows
  for (int start_i = 0; start_i < n_nodes; start_i += batch_size) {

    auto outer_t0 = clock::now();

    const int end_i = std::min(start_i + batch_size, n_nodes);
    const int batch_i_size = end_i - start_i;

    // ----- Generate batch-i vectors on CPU (no global cache) -----
    std::vector<std::vector<dtype>> vecs_i(batch_i_size);
    std::vector<int> sizes_i(batch_i_size, 0);
    int max_vec_size_i = 0;

    for (int li = 0; li < batch_i_size; ++li) {
      int v = node_ids[start_i + li];
      auto vec = generate_node_state(v, c, i, s, firstHopStates, data,
                                     nodeDegrees, nodeDegreesC1);
      sizes_i[li] = static_cast<int>(vec.size());
      max_vec_size_i = std::max(max_vec_size_i, sizes_i[li]);
      vecs_i[li] = std::move(vec);
    }

    if (max_vec_size_i == 0) {
      // No work in this i-batch; skip
      continue;
    }

    // Host staging for i-batch (padded)
    std::vector<float> h_batch_i(
        static_cast<size_t>(batch_i_size) * max_vec_size_i, 0.0f);
    std::vector<float> h_inv_sqrt_i(batch_i_size, 0.0f);
    for (int li = 0; li < batch_i_size; ++li) {
      const auto &v = vecs_i[li];
      if (!v.empty()) {
        for (int k = 0; k < sizes_i[li]; ++k) {
          h_batch_i[static_cast<size_t>(li) * max_vec_size_i + k] =
              static_cast<float>(v[k]);
        }
        h_inv_sqrt_i[li] = 1.0f / std::sqrt(static_cast<float>(sizes_i[li]));
      }
    }

    // Ensure GPU buffers for i-batch
    buf.ensure_floats(&buf.d_batch_i, buf.cap_batch_i_elems,
                      static_cast<size_t>(batch_i_size) * max_vec_size_i);
    buf.ensure_ints(&buf.d_sizes_i, buf.cap_sizes_i, batch_i_size);
    buf.ensure_floats(&buf.d_inv_sqrt_i, buf.cap_inv_sqrt_i, batch_i_size);

    // Copy i-batch to GPU
    CUDA_CHECK(cudaMemcpyAsync(buf.d_batch_i, h_batch_i.data(),
                               h_batch_i.size() * sizeof(float),
                               cudaMemcpyHostToDevice, buf.stream));
    CUDA_CHECK(cudaMemcpyAsync(buf.d_sizes_i, sizes_i.data(),
                               batch_i_size * sizeof(int),
                               cudaMemcpyHostToDevice, buf.stream));
    CUDA_CHECK(cudaMemcpyAsync(buf.d_inv_sqrt_i, h_inv_sqrt_i.data(),
                               batch_i_size * sizeof(float),
                               cudaMemcpyHostToDevice, buf.stream));

    // ----- Inner loop over columns -----
    for (int start_j = start_i; start_j < n_nodes; start_j += batch_size) {
      auto batch_t0 = clock::now();
      const int end_j = std::min(start_j + batch_size, n_nodes);
      const int batch_j_size = end_j - start_j;

      std::cout << "GPU Batch: row=" << start_i << ", column=" << start_j
                << std::endl;

      // Generate batch-j vectors
      std::vector<std::vector<dtype>> vecs_j(batch_j_size);
      std::vector<int> sizes_j(batch_j_size, 0);
      int max_vec_size_j = 0;

      for (int lj = 0; lj < batch_j_size; ++lj) {
        int v = node_ids[start_j + lj];
        auto vec = generate_node_state(v, c, i, s, firstHopStates, data,
                                       nodeDegrees, nodeDegreesC1);
        sizes_j[lj] = static_cast<int>(vec.size());
        max_vec_size_j = std::max(max_vec_size_j, sizes_j[lj]);
        vecs_j[lj] = std::move(vec);
      }
      if (max_vec_size_j == 0) {
        // nothing to compute with this j-batch
        continue;
      }

      // Host staging for j-batch (padded)
      std::vector<float> h_batch_j(
          static_cast<size_t>(batch_j_size) * max_vec_size_j, 0.0f);
      for (int lj = 0; lj < batch_j_size; ++lj) {
        const auto &v = vecs_j[lj];
        for (int k = 0; k < sizes_j[lj]; ++k) {
          h_batch_j[static_cast<size_t>(lj) * max_vec_size_j + k] =
              static_cast<float>(v[k]);
        }
      }

      // Ensure GPU buffers for j-batch + pairwise outputs
      buf.ensure_floats(&buf.d_batch_j, buf.cap_batch_j_elems,
                        static_cast<size_t>(batch_j_size) * max_vec_size_j);
      buf.ensure_ints(&buf.d_sizes_j, buf.cap_sizes_j, batch_j_size);

      const size_t pair_elems =
          static_cast<size_t>(batch_i_size) * batch_j_size;
      buf.ensure_floats(&buf.d_dot, buf.cap_dot, pair_elems);
      buf.ensure_floats(&buf.d_out, buf.cap_out, pair_elems);

      // Copy j-batch to GPU
      CUDA_CHECK(cudaMemcpyAsync(buf.d_batch_j, h_batch_j.data(),
                                 h_batch_j.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, buf.stream));
      CUDA_CHECK(cudaMemcpyAsync(buf.d_sizes_j, sizes_j.data(),
                                 batch_j_size * sizeof(int),
                                 cudaMemcpyHostToDevice, buf.stream));

      // Launch kernels
      dim3 block(16, 16);
      dim3 grid((batch_i_size + block.x - 1) / block.x,
                (batch_j_size + block.y - 1) / block.y);

      const bool is_diag = (start_i == start_j);

      compute_dot_products_kernel<<<grid, block, 0, buf.stream>>>(
          buf.d_batch_i, buf.d_batch_j, buf.d_sizes_i, buf.d_sizes_j, buf.d_dot,
          batch_i_size, batch_j_size, max_vec_size_i, max_vec_size_j, is_diag);
      CUDA_CHECK(cudaGetLastError());

      compute_kernel_values<<<grid, block, 0, buf.stream>>>(
          buf.d_dot, buf.d_out, buf.d_inv_sqrt_i, batch_i_size, batch_j_size,
          nPower, is_diag);
      CUDA_CHECK(cudaGetLastError());

      // Copy back
      std::vector<float> h_out(pair_elems);
      CUDA_CHECK(cudaMemcpyAsync(h_out.data(), buf.d_out,
                                 pair_elems * sizeof(float),
                                 cudaMemcpyDeviceToHost, buf.stream));
      CUDA_CHECK(cudaStreamSynchronize(buf.stream));

      // Scatter to K_block (symmetric fill, upper triangle only on-diagonal)
      for (int li = 0; li < batch_i_size; ++li) {
        if (sizes_i[li] == 0)
          continue;
        for (int lj = 0; lj < batch_j_size; ++lj) {
          if (sizes_j[lj] == 0)
            continue;
          if (is_diag && lj < li)
            continue;

          float kv = h_out[static_cast<size_t>(li) * batch_j_size + lj];
          int gi = start_i + li;
          int gj = start_j + lj;
          K_block[gi][gj] = static_cast<dtype>(kv);
          K_block[gj][gi] = static_cast<dtype>(kv);
        }
      }
      auto batch_t1 = clock::now();
      double batch_ms = duration_ms(batch_t1 - batch_t0).count();
      std::cout << "  ↳ GPU batch time: " << batch_ms << " ms" << std::endl;

      total_gpu_batch_ms += batch_ms;
      gpu_batch_count++;
    }
    auto outer_t1 = clock::now();
    double outer_ms = duration_ms(outer_t1 - outer_t0).count();
    std::cout << "Outer loop starting at i=" << start_i << " took " << outer_ms
              << " ms" << std::endl;

    total_outer_loop_ms += outer_ms;
    outer_loop_count++;
  }
  std::cout << "\n=== Timing Summary ===" << std::endl;
  std::cout << "Average GPU batch time: "
            << (gpu_batch_count ? total_gpu_batch_ms / gpu_batch_count : 0.0)
            << " ms over " << gpu_batch_count << " batches" << std::endl;
  std::cout << "Average outer loop time: "
            << (outer_loop_count ? total_outer_loop_ms / outer_loop_count : 0.0)
            << " ms over " << outer_loop_count << " outer loops" << std::endl;
  std::cout << "Total compute time: " << (total_outer_loop_ms) << " ms"
            << std::endl;

  return K_block;
}
