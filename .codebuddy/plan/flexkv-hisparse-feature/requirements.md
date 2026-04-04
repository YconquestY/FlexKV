# 需求文档：在 FlexKV 上实现类似 HiSparse 的稀疏注意力分层缓存功能

## 引言

### 背景

SGLang 的 HiSparse 是一种**分层稀疏注意力加速机制**，核心思想是：
- **Indexer KV cache 常驻 GPU 显存**（覆盖所有历史 token，体积小，FP8 量化后每 token 仅 132 bytes）
- **完整 MLA KV cache 大部分存储在 CPU 内存**（每 token 1152 bytes @ bf16）
- Decode 时，Indexer 先扫描 GPU 上的全量 indexer key 计算 TopK，再根据 TopK 结果从 CPU **按需加载**选中 token 的完整 KV cache 到 GPU 的 device buffer 中
- 通过 LRU 热缓存减少实际传输量，连续 decode 步骤的 TopK 重叠率通常 >80%

### 当前 FlexKV 的能力

FlexKV 在 `feat/flexkv_rebase` 分支上已经实现了以下关键能力：

1. **Indexer 存储与传输**：
   - `IndexerCacheConfig`：自动检测 DSA/NSA 模型的 indexer 参数（`qk_rope_head_dim`）
   - `StorageEngine`：为 indexer 独立分配 CPU/SSD/Remote 存储（`_indexer_storage_handles`）
   - `TransferEngine`：独立的 indexer H2D/D2H/Layerwise worker，与主 KV 并行传输
   - `pending_count` 机制：主 KV 和 indexer 传输完成后才标记 op 完成

2. **逐层传输（Layerwise Transfer）**：
   - `LayerwiseTransferGroup`（C++）：支持 SSD→CPU→GPU 的逐层传输
   - `eventfd` 通知机制：每层传输完成后通过 eventfd 通知 SGLang 的 attention 层
   - Triple buffering（`counter_id`）：支持多组 eventfd 实现流水线

3. **分层缓存引擎**：
   - `HierarchyLRCacheEngine`：支持 local/remote radix tree 索引
   - 支持 CPU、SSD、Remote（PCFS）多级存储
   - LRU eviction 策略

### 核心问题

用户的问题是：**如果将 `feat/flexkv_rebase` 分支 rebase 到 main 分支，能否在 FlexKV 上实现类似 HiSparse 的功能？**

答案是**可以**，但需要补充以下关键能力：

| HiSparse 功能 | FlexKV 现有能力 | 差距 |
|---------------|----------------|------|
| Indexer 常驻 GPU（2x 容量） | ✅ Indexer 独立存储 + 传输 | ❌ 缺少 "indexer 不参与 eviction" 的语义 |
| TopK 稀疏选择 | ❌ 无 | ❌ 需要 SGLang 侧提供 TopK indices |
| 按 TopK 结果从 CPU 按需加载 KV | ✅ Layerwise H2D 传输 | ❌ 当前是按 block 全量传输，非按 token 稀疏传输 |
| LRU device buffer 热缓存 | ❌ 无 | ❌ 需要新增 per-request per-layer 的 LRU device buffer |
| CUDA kernel swap-in | ❌ 无 | ❌ 需要类似 `load_cache_to_device_buffer_mla` 的 kernel |
| Indexer 逐层传输 + eventfd 同步 | ✅ 已有 | ✅ 可复用 |
| Staging（prefill→decode 过渡） | ❌ 无 | ❌ 需要新增 staging 机制 |

### 关键洞察：HiSparse 模式对 HiCache 代码改动不大

HiSparse 在 SGLang 中的实现确实没有大幅修改 HiCache 的代码，而是：
1. 新增了 `HiSparseCoordinator`（独立模块）
2. 新增了 `HiSparseNSATokenToKVPool` 和 `HiSparseTokenToKVPoolAllocator`（继承基类）
3. 新增了 `hisparse.cuh` CUDA kernel
4. 在 `nsa_backend.py` 中增加了 HiSparse 分支逻辑

类似地，在 FlexKV 上实现 HiSparse 功能也**不需要大幅修改现有的 HiCache/Layerwise 代码**，而是在现有基础上新增模块。

---

## 需求

### 需求 1：Indexer KV Cache 的 GPU 常驻管理

**用户故事：** 作为一名 FlexKV 集成开发者，我希望 FlexKV 能够管理 Indexer KV cache 的 GPU 常驻策略，以便 Indexer 在 decode 时可以直接从 GPU 显存扫描所有历史 token 的 indexer key，无需任何传输。

#### 验收标准

1. WHEN FlexKV 初始化 HiSparse 模式 THEN FlexKV SHALL 为 indexer 分配 `size × host_to_device_ratio`（默认 2 倍）容量的 GPU buffer，覆盖所有逻辑 token 空间
2. WHEN indexer GPU buffer 已分配 THEN FlexKV SHALL 确保 indexer buffer 不参与任何 eviction 或 swap-out 操作，始终常驻 GPU 显存
3. WHEN 新 token 生成时 THEN FlexKV SHALL 支持 SGLang 侧直接写入 indexer key 到 GPU buffer 的对应逻辑位置，无需经过 FlexKV 的传输路径
4. IF indexer GPU buffer 容量不足以覆盖所有逻辑 token THEN FlexKV SHALL 在初始化时报错并给出明确的配置建议

### 需求 2：基于 TopK 的稀疏 KV Cache 按需加载

**用户故事：** 作为一名 FlexKV 集成开发者，我希望 FlexKV 能够根据 SGLang 提供的 TopK token indices，从 CPU 按需加载选中 token 的完整 KV cache 到 GPU 的 device buffer 中，以便只传输稀疏注意力实际需要的数据，大幅减少 CPU→GPU 带宽消耗。

#### 验收标准

1. WHEN SGLang 在某一层的 Indexer 计算完成并产出 TopK indices THEN FlexKV SHALL 提供接口接收这些 indices 并触发对应 token 的 KV cache 加载
2. WHEN 收到 TopK indices THEN FlexKV SHALL 仅加载 miss 的 token（不在 device buffer 中的），跳过已缓存的 hit token
3. WHEN 加载 miss token 时 THEN FlexKV SHALL 使用 LRU 策略驱逐 device buffer 中最久未使用的 token，为 miss token 腾出空间
4. WHEN 加载完成 THEN FlexKV SHALL 返回 TopK token 在 device buffer 中的物理位置（page table），供 SGLang 的 sparse attention kernel 使用
5. IF 某个 request 的 seq_len ≤ device_buffer_size THEN FlexKV SHALL 将所有 token 保留在 device buffer 中，跳过 LRU swap-in 逻辑（短序列快速路径）

### 需求 3：Per-Request Per-Layer 的 LRU Device Buffer 管理

**用户故事：** 作为一名 FlexKV 集成开发者，我希望 FlexKV 能够为每个 request 的每一层维护独立的 LRU device buffer，以便不同层可以有不同的热点 token 分布，最大化 LRU 命中率。

#### 验收标准

1. WHEN HiSparse 模式初始化 THEN FlexKV SHALL 为每个 request 的每一层分配独立的 device buffer 元数据（缓存了哪些 token、物理位置、LRU 排序）
2. WHEN 某一层的 swap-in 操作更新了 LRU 状态 THEN FlexKV SHALL 仅更新该层的 LRU 状态，不影响其他层
3. WHEN request 完成 THEN FlexKV SHALL 释放该 request 在所有层的 device buffer 资源
4. WHEN device_buffer_size 配置为 N THEN FlexKV SHALL 为每个 request 分配最多 N + page_size 个 slot（N 个 LRU 管理 + 1 个 reserved slot 用于当前 decode token）

### 需求 4：CUDA Kernel 实现 — swap_in_selected_pages

**用户故事：** 作为一名 FlexKV 内核开发者，我希望 FlexKV 提供高效的 CUDA kernel 来执行 TopK 驱动的 LRU swap-in 操作，以便在 decode 的关键路径上最小化延迟。

#### 验收标准

1. WHEN 调用 swap_in kernel THEN FlexKV SHALL 在单个 kernel launch 中完成以下操作：hash 查找 hit/miss、LRU 重排、miss token 的 host→device DMA
2. WHEN TopK token 已在 device buffer 中（hit）THEN kernel SHALL 将其 LRU 排序移至 MRU 端，并直接返回其 device buffer 物理位置
3. WHEN TopK token 不在 device buffer 中（miss）THEN kernel SHALL 驱逐 LRU 端的 slot，通过 `ld.global.nc.b64` / `st.global.cg.b64` 指令执行 host→device DMA，并更新元数据
4. WHEN seq_len ≤ device_buffer_size（短序列）THEN kernel SHALL 走快速路径，直接查表返回物理位置，跳过 hash/LRU 逻辑
5. WHEN 在 CUDA Graph 模式下 THEN kernel SHALL 使用预分配的输出 buffer 和 `num_real_reqs` 标量来支持 padded blocks 的 early exit

### 需求 5：Staging 机制 — Prefill 到 Decode 的过渡

**用户故事：** 作为一名 FlexKV 集成开发者，我希望 FlexKV 支持 staging 机制，在 prefill 完成后异步将所有 KV cache 备份到 CPU，并为 decode 阶段分配 device buffer，以便平滑过渡到 HiSparse decode 模式。

#### 验收标准

1. WHEN prefill 完成 THEN FlexKV SHALL 异步将 prefill 阶段写入 GPU 的所有 KV cache 备份到 CPU（使用独立的 staging stream）
2. WHEN staging 备份完成 THEN FlexKV SHALL 为该 request 分配 decode 用的 device buffer 空间
3. WHEN staging 完成后首次 decode THEN FlexKV SHALL 跳过 "备份上一步 token" 的操作（因为 staging 已备份所有 prefill token）
4. IF staging 过程中 request 被中止 THEN FlexKV SHALL 等待 in-flight DMA 完成后释放所有已分配的 host 和 device 资源
5. IF KV 数据已通过 RDMA 直接写入 host pool（direct-to-host 路径）THEN FlexKV SHALL 跳过 staging DMA，直接为 decode 分配 device buffer

### 需求 6：逐步 Token 备份机制

**用户故事：** 作为一名 FlexKV 集成开发者，我希望 FlexKV 在每步 decode 时将上一步新生成的 token 的 KV cache 备份到 CPU，以便后续 decode 步骤可以从 CPU 按需加载该 token。

#### 验收标准

1. WHEN 每步 decode 开始时 THEN FlexKV SHALL 将上一步（而非当前步）新生成的 token 的 KV cache 从 device buffer 备份到 CPU host pool
2. WHEN 备份时 THEN FlexKV SHALL 一次 kernel 调用备份所有层的 KV cache（`backup_from_device_all_layer`），减少 kernel launch 开销
3. WHEN 备份完成 THEN FlexKV SHALL 更新 `req_to_host_pool` 映射表，记录该 token 在 CPU 中的物理位置
4. IF 是 staging 后的首次 decode THEN FlexKV SHALL 跳过备份操作（设置 `_skip_first_backup` 标志）

### 需求 7：与现有 FlexKV Layerwise Transfer 的集成

**用户故事：** 作为一名 FlexKV 架构师，我希望 HiSparse 功能能够复用现有的 layerwise transfer 基础设施（eventfd 通知、triple buffering、SSD→CPU→GPU 流水线），以便最小化代码改动和维护成本。

#### 验收标准

1. WHEN HiSparse 模式启用 THEN FlexKV SHALL 复用现有的 `LayerwiseTransferGroup` C++ 实现来执行 indexer 的逐层传输（如果需要从 SSD/Remote 加载 indexer）
2. WHEN HiSparse 模式启用 THEN FlexKV SHALL 复用现有的 eventfd 通知机制来同步 indexer 的逐层传输完成状态
3. WHEN HiSparse decode 阶段 THEN FlexKV SHALL **不使用** layerwise transfer 来传输主 KV cache（因为主 KV 是按 TopK 稀疏加载的，不是按层全量加载的）
4. IF indexer 已常驻 GPU THEN FlexKV SHALL 跳过 indexer 的 H2D 传输，仅在初始加载（如从 SSD/Remote 恢复）时使用 layerwise transfer

### 需求 8：双层分配器 — 逻辑空间与物理 Device Buffer 的映射

**用户故事：** 作为一名 FlexKV 内核开发者，我希望 FlexKV 提供双层分配器来管理逻辑 token 索引和物理 device buffer 索引的映射，以便 indexer 可以使用逻辑索引直接写入，而 KV cache 写入需要经过逻辑→物理映射。

#### 验收标准

1. WHEN 分配 token 空间 THEN FlexKV SHALL 同时分配逻辑索引（用于 indexer buffer）和物理索引（用于 KV device buffer）
2. WHEN indexer 写入 key cache THEN FlexKV SHALL 使用逻辑索引直接写入 indexer buffer（因为 indexer buffer 足够大，覆盖所有逻辑空间）
3. WHEN KV cache 写入 device buffer THEN FlexKV SHALL 通过 `full_to_hisparse_device_index_mapping` 将逻辑索引映射为物理 device buffer 索引
4. WHEN 查询可用空间 THEN FlexKV SHALL 返回 `min(logical.available, hisparse.available)`，确保两个空间都有足够容量

### 需求 9：FlexKV 侧的接口设计 — 与 SGLang 的交互协议

**用户故事：** 作为一名 FlexKV 集成开发者，我希望 FlexKV 提供清晰的接口供 SGLang 调用，以便 SGLang 的 `NativeSparseAttnBackend` 可以在每层 decode 时触发稀疏 KV cache 加载。

#### 验收标准

1. WHEN SGLang 初始化 HiSparse 模式 THEN FlexKV SHALL 提供 `init_hisparse(device_buffer_size, top_k, host_to_device_ratio)` 接口来配置 HiSparse 参数
2. WHEN SGLang 每层 decode 时 THEN FlexKV SHALL 提供 `swap_in_selected_pages(req_pool_indices, seq_lens, topk_indices, layer_id) → page_table` 接口
3. WHEN SGLang prefill 完成时 THEN FlexKV SHALL 提供 `admit_request_into_staging(req)` 接口来触发 staging
4. WHEN SGLang 每步 decode 开始时 THEN FlexKV SHALL 提供 `map_last_loc_to_buffer(seq_lens, req_pool_indices, out_cache_loc)` 接口来执行备份和 buffer 映射
5. WHEN SGLang request 完成时 THEN FlexKV SHALL 提供 `release_request(req)` 接口来释放所有资源

---

## 可行性分析

### FlexKV 现有基础设施的复用

| FlexKV 现有组件 | HiSparse 中的对应 | 复用方式 |
|----------------|-------------------|---------|
| `IndexerCacheConfig` | indexer 参数检测 | ✅ 直接复用 |
| `StorageEngine._indexer_storage_handles` | indexer CPU/SSD 存储 | ✅ 直接复用 |
| `TransferEngine._indexer_*_workers` | indexer 传输 worker | ✅ 复用于初始加载 |
| `LayerwiseTransferGroup` + eventfd | 逐层传输 + 通知 | ✅ 复用于 indexer 初始加载 |
| `HierarchyLRCacheEngine` | radix tree 索引 | ✅ 复用于 KV cache 的 CPU/SSD 索引 |
| `pending_count` 机制 | 主 KV + indexer 同步 | ✅ 直接复用 |

### 需要新增的组件

| 新增组件 | 对应 SGLang 代码 | 复杂度 |
|---------|-----------------|--------|
| `HiSparseCoordinator`（FlexKV 版） | `hisparse_coordinator.py` | 高 — 核心调度逻辑 |
| `swap_in_selected_pages` CUDA kernel | `hisparse.cuh` | 高 — 性能关键 |
| 双层分配器 | `HiSparseTokenToKVPoolAllocator` | 中 — 逻辑清晰 |
| Per-request per-layer LRU 元数据 | `HiSparseCoordinator` 内部 | 中 |
| Staging 机制 | `admit_request_into_staging` | 中 |
| SGLang 交互接口 | `nsa_backend.py` 中的调用 | 低 — 接口封装 |

### Rebase 到 main 分支的影响

1. **indexer 存储和传输**：`feat/flexkv_rebase` 分支已实现的 indexer 相关代码（`IndexerCacheConfig`、`StorageEngine` indexer 分配、`TransferEngine` indexer worker）需要 rebase 到 main
2. **layerwise transfer**：已有的 `LayerwiseTransferGroup` 和 eventfd 机制可以直接复用
3. **新增 HiSparse 模块**：在 rebase 后的 main 分支上新增 HiSparse 相关模块，不影响现有代码

### 结论

**在 FlexKV 上实现类似 HiSparse 的功能是完全可行的**，核心原因：
1. FlexKV 已有 indexer 的独立存储和传输基础设施
2. FlexKV 已有 layerwise transfer + eventfd 的逐层同步机制
3. HiSparse 的核心逻辑（LRU device buffer、swap-in kernel、staging）可以作为独立模块新增，不需要大幅修改现有代码
4. 主要工作量在于新增 CUDA kernel（`swap_in_selected_pages`）和 `HiSparseCoordinator` 调度逻辑
