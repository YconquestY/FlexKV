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
#include "tp_transfer_thread_group.h"
#include "transfer.cuh"
#include <stdexcept>
#include <cstdio>
#include <chrono>
#include <sstream>

namespace flexkv {

TPTransferThreadGroup::TPTransferThreadGroup(
    int num_gpus, const std::vector<int64_t> &gpu_block_ptrs_flat,
    int num_tensors_per_gpu, int64_t cpu_blocks_ptr, int dp_group_id,
    int num_layers, const std::vector<int64_t> &gpu_kv_strides_in_bytes,
    const std::vector<int64_t> &gpu_block_strides_in_bytes,
    const std::vector<int64_t> &gpu_layer_strides_in_bytes,
    const std::vector<int64_t> &gpu_chunk_sizes_in_bytes,
    const std::vector<int64_t> &gpu_device_ids) {
  num_gpus_ = num_gpus;
  num_tensors_per_gpu_ = num_tensors_per_gpu;
  dp_group_id_ = dp_group_id;

  gpu_kv_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_block_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_layer_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_chunk_sizes_in_bytes_ = new int64_t[num_gpus];
  for (int i = 0; i < num_gpus; i++) {
    gpu_kv_strides_in_bytes_[i] = gpu_kv_strides_in_bytes[i];
    gpu_block_strides_in_bytes_[i] = gpu_block_strides_in_bytes[i];
    gpu_layer_strides_in_bytes_[i] = gpu_layer_strides_in_bytes[i];
    gpu_chunk_sizes_in_bytes_[i] = gpu_chunk_sizes_in_bytes[i];
  }

  queues_.resize(num_gpus_);
  mtxs_ = std::vector<std::mutex>(num_gpus_);
  cvs_ = std::vector<std::condition_variable>(num_gpus_);

  cudaError_t malloc_err = cudaMallocHost(
      (void **)&gpu_blocks_, num_gpus_ * num_tensors_per_gpu_ * sizeof(void *));
  if (malloc_err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMallocHost failed: ") +
                             cudaGetErrorString(malloc_err));
  }
  for (size_t i = 0; i < gpu_block_ptrs_flat.size(); ++i) {
    gpu_blocks_[i] = reinterpret_cast<void *>(gpu_block_ptrs_flat[i]);
  }

  if (num_tensors_per_gpu_ == 1) {
    backend_type_ = BackendType::TRTLLM;
  } else if (num_tensors_per_gpu_ == num_layers) {
    backend_type_ = BackendType::VLLM;
  } else if (num_tensors_per_gpu_ == num_layers * 2) {
    backend_type_ = BackendType::SGLANG;
  } else {
    throw std::runtime_error("Unsupported GPU block type: " +
                             std::to_string(num_tensors_per_gpu_));
  }

  gpu_tensor_handlers_.reserve(num_gpus_);
  for (int i = 0; i < num_gpus_; i++) {
    int64_t **gpu_blocks_ptr =
        reinterpret_cast<int64_t **>(gpu_blocks_ + i * num_tensors_per_gpu_);
    gpu_tensor_handlers_.emplace_back(
        backend_type_, gpu_blocks_ptr, num_layers, gpu_kv_strides_in_bytes_[i],
        gpu_block_strides_in_bytes_[i], gpu_layer_strides_in_bytes_[i]);
  }

  cpu_blocks_ = reinterpret_cast<void *>(cpu_blocks_ptr);

  gpu_device_ids_.resize(num_gpus_);
  for (int i = 0; i < num_gpus_; ++i) {
    gpu_device_ids_[i] = static_cast<int>(gpu_device_ids[i]);
  }

  streams_.resize(num_gpus_);
  for (int i = 0; i < num_gpus_; i += 1) {
    cudaError_t err = cudaSetDevice(gpu_device_ids_[i]);
    if (err != cudaSuccess)
      throw std::runtime_error(std::string("cudaSetDevice failed: ") +
                               cudaGetErrorString(err));
    err = cudaStreamCreate(&streams_[i]);
    if (err != cudaSuccess)
      throw std::runtime_error(std::string("cudaStreamCreate failed: ") +
                               cudaGetErrorString(err));
  }
  // create the thread pool
  stop_pool_ = false;
  for (int i = 0; i < num_gpus_; ++i) {
    threads_.emplace_back([this, i]() {
      int device_id = gpu_device_ids_[i];
      cudaSetDevice(device_id); // only once

      while (true) {
        Task task;
        {
          std::unique_lock<std::mutex> lk(mtxs_[i]);
          cvs_[i].wait(lk, [&] { return stop_pool_ || !queues_[i].empty(); });
          if (stop_pool_ && queues_[i].empty())
            return;

          task = std::move(queues_[i].front());
          queues_[i].pop();
        }
        task(); //
      }
    });
  }
}

TPTransferThreadGroup::~TPTransferThreadGroup() {
  stop_pool_ = true;
  for (auto &cv : cvs_)
    cv.notify_all();
  for (auto &t : threads_)
    if (t.joinable())
      t.join();

  cudaFreeHost(gpu_blocks_);

  gpu_tensor_handlers_.clear();
  delete[] gpu_kv_strides_in_bytes_;
  delete[] gpu_block_strides_in_bytes_;
  delete[] gpu_layer_strides_in_bytes_;
  delete[] gpu_chunk_sizes_in_bytes_;
}

std::future<void> TPTransferThreadGroup::enqueue_for_gpu(int gpu_idx,
                                                         Task task) {
  auto pkg = std::make_shared<std::packaged_task<void()>>(std::move(task));
  auto fut = pkg->get_future();
  {
    std::lock_guard<std::mutex> lk(mtxs_[gpu_idx]);
    queues_[gpu_idx].emplace([pkg] { (*pkg)(); });
  }
  cvs_[gpu_idx].notify_one();
  return fut;
}

void TPTransferThreadGroup::tp_group_transfer(
    const torch::Tensor &gpu_block_id_tensor,
    const torch::Tensor &cpu_block_id_tensor,
    const int64_t cpu_kv_stride_in_bytes,
    const int64_t cpu_layer_stride_in_bytes,
    const int64_t cpu_block_stride_in_bytes,
    const int64_t cpu_tp_stride_in_bytes, const int transfer_num_cta,
    const bool is_host_to_device, const bool use_ce_transfer,
    const int layer_id, const int layer_granularity, const bool is_mla) {

  // [DIAG] Log entry with all parameters
  int num_blocks_total = gpu_block_id_tensor.numel();
  int64_t *gpu_bids_ptr = static_cast<int64_t *>(gpu_block_id_tensor.data_ptr());
  int64_t *cpu_bids_ptr = static_cast<int64_t *>(cpu_block_id_tensor.data_ptr());

  {
    std::ostringstream oss;
    oss << "[DIAG] tp_group_transfer ENTER: "
        << "num_gpus=" << num_gpus_
        << ", num_blocks=" << num_blocks_total
        << ", layer_id=" << layer_id
        << ", layer_gran=" << layer_granularity
        << ", is_h2d=" << is_host_to_device
        << ", use_ce=" << use_ce_transfer
        << ", is_mla=" << is_mla
        << ", transfer_num_cta=" << transfer_num_cta
        << ", cpu_kv_stride=" << cpu_kv_stride_in_bytes
        << ", cpu_layer_stride=" << cpu_layer_stride_in_bytes
        << ", cpu_block_stride=" << cpu_block_stride_in_bytes
        << ", cpu_tp_stride=" << cpu_tp_stride_in_bytes
        << ", backend=" << static_cast<int>(backend_type_)
        << ", cpu_blocks_ptr=" << reinterpret_cast<uintptr_t>(cpu_blocks_);

    // Dump gpu_block_ids
    oss << ", gpu_block_ids=[";
    for (int k = 0; k < std::min(num_blocks_total, 10); k++) {
      if (k > 0) oss << ",";
      oss << gpu_bids_ptr[k];
    }
    if (num_blocks_total > 10) oss << "...";
    oss << "]";

    // Dump cpu_block_ids
    oss << ", cpu_block_ids=[";
    for (int k = 0; k < std::min(num_blocks_total, 10); k++) {
      if (k > 0) oss << ",";
      oss << cpu_bids_ptr[k];
    }
    if (num_blocks_total > 10) oss << "...";
    oss << "]";

    fprintf(stderr, "%s\n", oss.str().c_str());
  }

  std::atomic<bool> failed{false};
  std::string error_msg;
  std::atomic<int> failed_gpu_idx{-1};
  // threads_.clear();
  // threads_.reserve(num_gpus_);

  // Barrier sync_point(num_gpus_);
  std::vector<std::future<void>> futures;
  futures.reserve(num_gpus_);

  for (int i = 0; i < num_gpus_; ++i) {
    futures.emplace_back(enqueue_for_gpu(i, [&, i]() {
      try {
        // [DIAG] Check for pre-existing CUDA errors on this GPU
        cudaError_t pre_err = cudaGetLastError();
        if (pre_err != cudaSuccess) {
          fprintf(stderr, "[DIAG] GPU %d (device=%d): PRE-EXISTING CUDA error before transfer: %s (code=%d)\n",
                  i, gpu_device_ids_[i], cudaGetErrorString(pre_err), static_cast<int>(pre_err));
        }

        int num_blocks = gpu_block_id_tensor.numel();

        int64_t *gpu_block_ids =
            static_cast<int64_t *>(gpu_block_id_tensor.data_ptr());
        int64_t *cpu_block_ids =
            static_cast<int64_t *>(cpu_block_id_tensor.data_ptr());
        void *cpu_ptr = cpu_blocks_;
        int64_t cpu_startoff_inside_chunks = i * cpu_tp_stride_in_bytes;
        if (is_mla && !is_host_to_device) {
          cpu_startoff_inside_chunks =
              i * gpu_chunk_sizes_in_bytes_[i] / num_gpus_;
        } else if (is_mla && is_host_to_device) {
          cpu_startoff_inside_chunks = 0;
        }
        int64_t gpu_startoff_inside_chunks =
            is_mla && !is_host_to_device
                ? i * gpu_chunk_sizes_in_bytes_[i] / num_gpus_
                : 0;
        // we assume that the chunk size is the same for all gpus,
        // even if they have different number of gpu_blocks
        int64_t chunk_size = is_mla && !is_host_to_device
                                 ? gpu_chunk_sizes_in_bytes_[i] / num_gpus_
                                 : gpu_chunk_sizes_in_bytes_[i];

        // [DIAG] Log per-GPU computed parameters
        fprintf(stderr, "[DIAG] GPU %d (device=%d): cpu_startoff=%lld, gpu_startoff=%lld, "
                "chunk_size=%lld, gpu_chunk_sizes_in_bytes=%lld, "
                "gpu_kv_stride=%lld, gpu_block_stride=%lld, gpu_layer_stride=%lld, "
                "stream=%p, num_tensors_per_gpu=%d\n",
                i, gpu_device_ids_[i],
                (long long)cpu_startoff_inside_chunks,
                (long long)gpu_startoff_inside_chunks,
                (long long)chunk_size,
                (long long)gpu_chunk_sizes_in_bytes_[i],
                (long long)gpu_kv_strides_in_bytes_[i],
                (long long)gpu_block_strides_in_bytes_[i],
                (long long)gpu_layer_strides_in_bytes_[i],
                (void*)streams_[i],
                num_tensors_per_gpu_);

        // [DIAG] Validate GPU tensor handler pointers for this GPU
        {
          int64_t **gpu_blocks_ptr = gpu_tensor_handlers_[i].gpu_tensor_ptrs;
          int kv_dim = is_mla ? 1 : 2;
          int num_ptrs_to_check = std::min(num_tensors_per_gpu_, 4); // check first few
          std::ostringstream ptr_oss;
          ptr_oss << "[DIAG] GPU " << i << " tensor_ptrs (first " << num_ptrs_to_check << "): [";
          for (int p = 0; p < num_ptrs_to_check; p++) {
            if (p > 0) ptr_oss << ", ";
            ptr_oss << "0x" << std::hex << reinterpret_cast<uintptr_t>(gpu_blocks_ptr[p]) << std::dec;
          }
          ptr_oss << "]";
          fprintf(stderr, "%s\n", ptr_oss.str().c_str());

          // [DIAG] Validate block_ids are within reasonable range
          for (int k = 0; k < num_blocks; k++) {
            if (gpu_block_ids[k] < 0) {
              fprintf(stderr, "[DIAG] GPU %d: NEGATIVE gpu_block_id[%d]=%lld!\n",
                      i, k, (long long)gpu_block_ids[k]);
            }
            if (cpu_block_ids[k] < 0) {
              fprintf(stderr, "[DIAG] GPU %d: NEGATIVE cpu_block_id[%d]=%lld!\n",
                      i, k, (long long)cpu_block_ids[k]);
            }
          }

          // [DIAG] For SGLANG backend, validate pointer computation for first block
          if (backend_type_ == BackendType::SGLANG && num_blocks > 0) {
            int64_t test_block_idx = gpu_block_ids[0];
            for (int li = layer_id; li < std::min(layer_id + 1, layer_id + layer_granularity); li++) {
              for (int ki = 0; ki < kv_dim; ki++) {
                int64_t *test_ptr = gpu_blocks_ptr[ki * gpu_tensor_handlers_[i].num_layers + li]
                                    + test_block_idx * gpu_tensor_handlers_[i].gpu_block_stride;
                fprintf(stderr, "[DIAG] GPU %d: SGLANG ptr_at(layer=%d, kv=%d, block=%lld) = 0x%llx, "
                        "base_ptr=0x%llx, block_stride=%lld\n",
                        i, li, ki, (long long)test_block_idx,
                        (unsigned long long)reinterpret_cast<uintptr_t>(test_ptr),
                        (unsigned long long)reinterpret_cast<uintptr_t>(
                            gpu_blocks_ptr[ki * gpu_tensor_handlers_[i].num_layers + li]),
                        (long long)gpu_tensor_handlers_[i].gpu_block_stride);
              }
            }
          }
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        // Dispatch to the appropriate template based on backend type
        switch (backend_type_) {
        case BackendType::VLLM:
          flexkv::transfer_kv_blocks<BackendType::VLLM>(
              num_blocks, layer_id, layer_granularity, gpu_block_ids,
              gpu_tensor_handlers_[i], gpu_startoff_inside_chunks,
              cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes,
              cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
              cpu_startoff_inside_chunks, chunk_size, streams_[i],
              transfer_num_cta, is_host_to_device, use_ce_transfer, is_mla,
              /*sync=*/false);
          break;
        case BackendType::TRTLLM:
          flexkv::transfer_kv_blocks<BackendType::TRTLLM>(
              num_blocks, layer_id, layer_granularity, gpu_block_ids,
              gpu_tensor_handlers_[i], gpu_startoff_inside_chunks,
              cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes,
              cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
              cpu_startoff_inside_chunks, chunk_size, streams_[i],
              transfer_num_cta, is_host_to_device, use_ce_transfer, is_mla,
              /*sync=*/false);
          break;
        case BackendType::SGLANG:
          flexkv::transfer_kv_blocks<BackendType::SGLANG>(
              num_blocks, layer_id, layer_granularity, gpu_block_ids,
              gpu_tensor_handlers_[i], gpu_startoff_inside_chunks,
              cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes,
              cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
              cpu_startoff_inside_chunks, chunk_size, streams_[i],
              transfer_num_cta, is_host_to_device, use_ce_transfer, is_mla,
              /*sync=*/false);
          break;
        }

        // [DIAG] Check error immediately after kernel launch (before sync)
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
          fprintf(stderr, "[DIAG] GPU %d (device=%d): CUDA error AFTER kernel launch (before sync): %s (code=%d)\n",
                  i, gpu_device_ids_[i], cudaGetErrorString(launch_err), static_cast<int>(launch_err));
          failed = true;
          int expected = -1;
          failed_gpu_idx.compare_exchange_strong(expected, i);
          error_msg = std::string("GPU ") + std::to_string(i) + " launch_err: " + cudaGetErrorString(launch_err);
          return; // skip sync since launch already failed
        }

        // [DIAG] Explicitly sync the stream and check sync error separately
        cudaError_t sync_err = cudaStreamSynchronize(streams_[i]);

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        if (sync_err != cudaSuccess) {
          fprintf(stderr, "[DIAG] GPU %d (device=%d): CUDA error AFTER cudaStreamSynchronize: %s (code=%d), "
                  "elapsed_ms=%.3f\n",
                  i, gpu_device_ids_[i], cudaGetErrorString(sync_err), static_cast<int>(sync_err),
                  elapsed_ms);
          failed = true;
          int expected = -1;
          failed_gpu_idx.compare_exchange_strong(expected, i);
          error_msg = std::string("GPU ") + std::to_string(i) + " sync_err: " + cudaGetErrorString(sync_err);
        } else {
          // [DIAG] Check for any post-sync error
          cudaError_t post_err = cudaGetLastError();
          if (post_err != cudaSuccess) {
            fprintf(stderr, "[DIAG] GPU %d (device=%d): CUDA error AFTER sync (via cudaGetLastError): %s (code=%d)\n",
                    i, gpu_device_ids_[i], cudaGetErrorString(post_err), static_cast<int>(post_err));
            failed = true;
            int expected = -1;
            failed_gpu_idx.compare_exchange_strong(expected, i);
            error_msg = std::string("GPU ") + std::to_string(i) + " post_sync_err: " + cudaGetErrorString(post_err);
          } else {
            fprintf(stderr, "[DIAG] GPU %d (device=%d): transfer OK, elapsed_ms=%.3f\n",
                    i, gpu_device_ids_[i], elapsed_ms);
          }
        }
      } catch (const std::exception &e) {
        fprintf(stderr, "[DIAG] GPU %d (device=%d): EXCEPTION in transfer: %s\n",
                i, gpu_device_ids_[i], e.what());
        failed = true;
        int expected = -1;
        failed_gpu_idx.compare_exchange_strong(expected, i);
        error_msg = std::string("GPU ") + std::to_string(i) + " exception: " + e.what();
      }
    }));
  }

  for (int fi = 0; fi < static_cast<int>(futures.size()); fi++) {
    try {
      futures[fi].get();
    } catch (const std::exception &e) {
      fprintf(stderr, "[DIAG] future[%d].get() threw: %s\n", fi, e.what());
      if (!failed) {
        failed = true;
        error_msg = std::string("future[") + std::to_string(fi) + "] exception: " + e.what();
      }
    }
  }

  if (failed) {
    fprintf(stderr, "[DIAG] tp_group_transfer FAILED: failed_gpu_idx=%d, error_msg=%s\n",
            failed_gpu_idx.load(), error_msg.c_str());
    throw std::runtime_error("tp_group_transfer failed: " + error_msg);
  }

  fprintf(stderr, "[DIAG] tp_group_transfer EXIT OK: num_blocks=%d, layer_id=%d, layer_gran=%d\n",
          num_blocks_total, layer_id, layer_granularity);
}

} // namespace flexkv
