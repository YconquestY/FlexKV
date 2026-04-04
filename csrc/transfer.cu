/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdio>
#include <chrono>

#include "monitoring/metrics_manager.h"
#include "transfer.cuh"

namespace flexkv {

#define FLOAT4_PTR(ptr) reinterpret_cast<float4 *>(ptr)

// Templated CUDA kernel - backend type determined at compile time
template <BackendType Type>
__global__ void transfer_kv_blocks_kernel(
    int num_blocks, int start_layer_id, int num_layers, int64_t *gpu_block_ids,
    GTensorHandler gpu_handler, int64_t gpu_startoff_inside_chunks,
    int64_t *cpu_block_ids, int64_t *cpu_ptr, int64_t cpu_kv_stride,
    int64_t cpu_layer_stride, int64_t cpu_block_stride,
    int64_t cpu_startoff_inside_chunks, int64_t copy_size, bool is_mla,
    bool is_host_to_device) {
  int kv_dim = is_mla ? 1 : 2;
  int num_chunks = num_layers * kv_dim * num_blocks;
  int64_t copy_size_in_float4 = copy_size * sizeof(int64_t) / sizeof(float4);

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int warps_per_block = blockDim.x / 32;
  int total_warps = gridDim.x * warps_per_block;

  for (int chunk_idx = blockIdx.x * warps_per_block + warp_id;
       chunk_idx < num_chunks; chunk_idx += total_warps) {
    int layer_idx = start_layer_id + chunk_idx / (num_blocks * kv_dim);
    int kv_idx = (chunk_idx % (num_blocks * kv_dim)) / num_blocks;
    int gpu_block_idx = gpu_block_ids[chunk_idx % num_blocks];
    int cpu_block_idx = cpu_block_ids[chunk_idx % num_blocks];

    int64_t *cpu_chunk_ptr =
        cpu_ptr + layer_idx * cpu_layer_stride + kv_idx * cpu_kv_stride +
        cpu_block_idx * cpu_block_stride + cpu_startoff_inside_chunks;

    // Use template specialization to compute gpu pointer
    int64_t *gpu_ptr =
        ptr_at<Type>(gpu_handler, layer_idx, kv_idx, gpu_block_idx);
    int64_t *gpu_chunk_ptr =
        reinterpret_cast<int64_t *>(gpu_ptr) + gpu_startoff_inside_chunks;

    int64_t *src_chunk_ptr = is_host_to_device ? cpu_chunk_ptr : gpu_chunk_ptr;
    int64_t *dst_chunk_ptr = is_host_to_device ? gpu_chunk_ptr : cpu_chunk_ptr;

    for (int64_t idx = lane_id; idx < copy_size_in_float4; idx += 32) {
      float4 element;
      asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3},[%4];"
                   : "=f"(element.x), "=f"(element.y), "=f"(element.z),
                     "=f"(element.w)
                   : "l"(&FLOAT4_PTR(src_chunk_ptr)[idx])
                   : "memory");
      asm volatile("st.global.cg.v4.f32 [%0],{%1,%2,%3,%4};" ::"l"(
                       &FLOAT4_PTR(dst_chunk_ptr)[idx]),
                   "f"(element.x), "f"(element.y), "f"(element.z),
                   "f"(element.w)
                   : "memory");
    }
  }
}

// Templated host function
template <BackendType Type>
void transfer_kv_blocks(
    int num_blocks, int start_layer_id, int num_layers, int64_t *gpu_block_ids,
    GTensorHandler gpu_tensor_handler, int64_t gpu_startoff_inside_chunks,
    int64_t *cpu_block_ids, void *cpu_ptr, int64_t cpu_kv_stride_in_bytes,
    int64_t cpu_layer_stride_in_bytes, int64_t cpu_block_stride_in_bytes,
    int64_t cpu_startoff_inside_chunks, int64_t chunk_size_in_bytes,
    cudaStream_t stream, int transfer_num_cta, bool is_host_to_device,
    bool use_ce_transfer, bool is_mla, bool sync) {

  // [DIAG] Log transfer_kv_blocks entry with all parameters
  fprintf(stderr, "[DIAG] transfer_kv_blocks<%d> ENTER: num_blocks=%d, start_layer=%d, num_layers=%d, "
          "gpu_startoff=%lld, cpu_startoff=%lld, chunk_size=%lld, "
          "cpu_kv_stride=%lld, cpu_layer_stride=%lld, cpu_block_stride=%lld, "
          "transfer_num_cta=%d, is_h2d=%d, use_ce=%d, is_mla=%d, sync=%d, "
          "stream=%p, cpu_ptr=%p, gpu_block_ids=%p, cpu_block_ids=%p\n",
          static_cast<int>(Type), num_blocks, start_layer_id, num_layers,
          (long long)gpu_startoff_inside_chunks, (long long)cpu_startoff_inside_chunks,
          (long long)chunk_size_in_bytes,
          (long long)cpu_kv_stride_in_bytes, (long long)cpu_layer_stride_in_bytes,
          (long long)cpu_block_stride_in_bytes,
          transfer_num_cta, (int)is_host_to_device, (int)use_ce_transfer, (int)is_mla, (int)sync,
          (void*)stream, cpu_ptr, (void*)gpu_block_ids, (void*)cpu_block_ids);

  // [DIAG] Dump block_ids
  {
    fprintf(stderr, "[DIAG] transfer_kv_blocks gpu_block_ids (first %d): [", std::min(num_blocks, 10));
    for (int k = 0; k < std::min(num_blocks, 10); k++) {
      if (k > 0) fprintf(stderr, ",");
      fprintf(stderr, "%lld", (long long)gpu_block_ids[k]);
    }
    fprintf(stderr, "]\n");
    fprintf(stderr, "[DIAG] transfer_kv_blocks cpu_block_ids (first %d): [", std::min(num_blocks, 10));
    for (int k = 0; k < std::min(num_blocks, 10); k++) {
      if (k > 0) fprintf(stderr, ",");
      fprintf(stderr, "%lld", (long long)cpu_block_ids[k]);
    }
    fprintf(stderr, "]\n");
  }

  // [DIAG] Check pre-existing error
  {
    cudaError_t pre_err = cudaPeekAtLastError();
    if (pre_err != cudaSuccess) {
      fprintf(stderr, "[DIAG] transfer_kv_blocks: PRE-EXISTING CUDA error: %s (code=%d)\n",
              cudaGetErrorString(pre_err), static_cast<int>(pre_err));
    }
  }

  int block_size = 1024;

  int block_count = transfer_num_cta;

  int64_t *cpu_ptr_int64 = reinterpret_cast<int64_t *>(cpu_ptr);
  int64_t cpu_kv_stride_int64 = cpu_kv_stride_in_bytes / sizeof(int64_t);
  int64_t cpu_block_stride_int64 = cpu_block_stride_in_bytes / sizeof(int64_t);
  int64_t cpu_layer_stride_int64 = cpu_layer_stride_in_bytes / sizeof(int64_t);
  int64_t cpu_startoff_inside_chunks_int64 =
      cpu_startoff_inside_chunks / sizeof(int64_t);
  int64_t gpu_startoff_inside_chunks_int64 =
      gpu_startoff_inside_chunks / sizeof(int64_t);
  int64_t chunk_size_in_int64 = chunk_size_in_bytes / sizeof(int64_t);

  dim3 blockDim(block_size);
  dim3 gridDim(block_count);

  // CE transfer mode (Copy Engine using cudaMemcpyAsync)
  if (use_ce_transfer) {
    int kv_dim = is_mla ? 1 : 2;
    for (int i = 0; i < num_layers; i++) {
      for (int j = 0; j < kv_dim; j++) {
        for (int k = 0; k < num_blocks; k++) {
          int64_t gpu_block_idx = gpu_block_ids[k];
          int64_t cpu_block_idx = cpu_block_ids[k];

          int64_t *cpu_chunk_ptr =
              cpu_ptr_int64 + (i + start_layer_id) * cpu_layer_stride_int64 +
              j * cpu_kv_stride_int64 + cpu_block_idx * cpu_block_stride_int64 +
              cpu_startoff_inside_chunks_int64;

          int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                          i + start_layer_id, j, gpu_block_idx);
          int64_t *gpu_chunk_ptr = reinterpret_cast<int64_t *>(gpu_ptr) +
                                   gpu_startoff_inside_chunks_int64;

          if (is_host_to_device) {
            cudaMemcpyAsync(gpu_chunk_ptr, cpu_chunk_ptr, chunk_size_in_bytes,
                            cudaMemcpyHostToDevice, stream);
          } else {
            cudaMemcpyAsync(cpu_chunk_ptr, gpu_chunk_ptr, chunk_size_in_bytes,
                            cudaMemcpyDeviceToHost, stream);
          }
          // Record transfer metrics after each cudaMemcpyAsync submission
          // Direction convention (from GPU perspective):
          //   - is_host_to_device=true  -> read (CPU->GPU, data flows INTO GPU)
          //   - is_host_to_device=false -> write (GPU->CPU, data flows OUT of
          //   GPU)
          FLEXKV_GPU_CPU_TRANSFER(is_host_to_device, chunk_size_in_bytes);
        }
      }
    }
  } else {
    // [DIAG] Log kernel launch parameters
    fprintf(stderr, "[DIAG] transfer_kv_blocks: launching kernel with gridDim=%d, blockDim=%d, "
            "chunk_size_in_int64=%lld, copy_size_in_float4=%lld\n",
            block_count, block_size,
            (long long)chunk_size_in_int64,
            (long long)(chunk_size_in_int64 * (int64_t)sizeof(int64_t) / (int64_t)sizeof(float4)));

    // [DIAG] Validate GPU pointer for first block before kernel launch
    if (num_blocks > 0) {
      int kv_dim = is_mla ? 1 : 2;
      for (int ki = 0; ki < kv_dim && ki < 1; ki++) {
        int64_t *test_gpu_ptr = ptr_at<Type>(gpu_tensor_handler, start_layer_id, ki, gpu_block_ids[0]);
        int64_t *test_gpu_chunk = reinterpret_cast<int64_t *>(test_gpu_ptr) + gpu_startoff_inside_chunks_int64;
        int64_t *test_cpu_chunk = cpu_ptr_int64 + start_layer_id * cpu_layer_stride_int64
                                  + ki * cpu_kv_stride_int64
                                  + cpu_block_ids[0] * cpu_block_stride_int64
                                  + cpu_startoff_inside_chunks_int64;
        fprintf(stderr, "[DIAG] transfer_kv_blocks: block[0] gpu_ptr=0x%llx, gpu_chunk_ptr=0x%llx, "
                "cpu_chunk_ptr=0x%llx, gpu_block_id=%lld, cpu_block_id=%lld\n",
                (unsigned long long)reinterpret_cast<uintptr_t>(test_gpu_ptr),
                (unsigned long long)reinterpret_cast<uintptr_t>(test_gpu_chunk),
                (unsigned long long)reinterpret_cast<uintptr_t>(test_cpu_chunk),
                (long long)gpu_block_ids[0], (long long)cpu_block_ids[0]);
      }
    }

    // Custom kernel transfer
    transfer_kv_blocks_kernel<Type><<<gridDim, blockDim, 0, stream>>>(
        num_blocks, start_layer_id, num_layers, gpu_block_ids,
        gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids,
        cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64,
        cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
        chunk_size_in_int64, is_mla, is_host_to_device);

    // [DIAG] Check error immediately after kernel launch
    {
      cudaError_t kernel_launch_err = cudaGetLastError();
      if (kernel_launch_err != cudaSuccess) {
        fprintf(stderr, "[DIAG] transfer_kv_blocks: KERNEL LAUNCH ERROR: %s (code=%d)\n",
                cudaGetErrorString(kernel_launch_err), static_cast<int>(kernel_launch_err));
      }
    }

    // Record transfer metrics after kernel launch (cannot record inside kernel)
    // Total bytes = actual_chunk_bytes * num_layers * kv_dim * num_blocks
    // Note: Kernel transfers in float4 units, so we calculate aligned bytes to
    // match Direction convention (from GPU perspective):
    //   - is_host_to_device=true  -> read (CPU->GPU, data flows INTO GPU)
    //   - is_host_to_device=false -> write (GPU->CPU, data flows OUT of GPU)
    int kv_dim = is_mla ? 1 : 2;
    // Calculate actual bytes transferred (aligned to float4, matching kernel
    // logic)
    int64_t actual_chunk_bytes =
        (chunk_size_in_int64 * sizeof(int64_t) / sizeof(float4)) *
        sizeof(float4);
    FLEXKV_GPU_CPU_TRANSFER(
        is_host_to_device,
        actual_chunk_bytes * static_cast<int64_t>(num_layers) *
            static_cast<int64_t>(kv_dim) * static_cast<int64_t>(num_blocks));
  }
  if (sync) {
    auto sync_start = std::chrono::high_resolution_clock::now();
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    auto sync_end = std::chrono::high_resolution_clock::now();
    double sync_ms = std::chrono::duration<double, std::milli>(sync_end - sync_start).count();
    if (sync_err != cudaSuccess) {
      fprintf(stderr, "[DIAG] transfer_kv_blocks: cudaStreamSynchronize FAILED: %s (code=%d), sync_ms=%.3f\n",
              cudaGetErrorString(sync_err), static_cast<int>(sync_err), sync_ms);
    } else {
      fprintf(stderr, "[DIAG] transfer_kv_blocks: sync OK, sync_ms=%.3f\n", sync_ms);
    }
    // [DIAG] Check for any residual error after sync
    cudaError_t post_sync_err = cudaPeekAtLastError();
    if (post_sync_err != cudaSuccess) {
      fprintf(stderr, "[DIAG] transfer_kv_blocks: POST-SYNC residual error: %s (code=%d)\n",
              cudaGetErrorString(post_sync_err), static_cast<int>(post_sync_err));
    }
  } else {
    fprintf(stderr, "[DIAG] transfer_kv_blocks: sync=false, skipping cudaStreamSynchronize\n");
  }
}

// Explicit template instantiations
template void transfer_kv_blocks<BackendType::VLLM>(int, int, int, int64_t *,
                                                    GTensorHandler, int64_t,
                                                    int64_t *, void *, int64_t,
                                                    int64_t, int64_t, int64_t,
                                                    int64_t, cudaStream_t, int,
                                                    bool, bool, bool, bool);

template void transfer_kv_blocks<BackendType::TRTLLM>(
    int, int, int, int64_t *, GTensorHandler, int64_t, int64_t *, void *,
    int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t, int, bool, bool,
    bool, bool);

template void transfer_kv_blocks<BackendType::SGLANG>(
    int, int, int, int64_t *, GTensorHandler, int64_t, int64_t *, void *,
    int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t, int, bool, bool,
    bool, bool);

} // namespace flexkv
