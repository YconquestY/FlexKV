# 稀疏注意力 Indexer Cache 支持 — 设计方案

## 1. 问题背景

DeepSeek V3.2 引入了 Native Sparse Attention（NSA）机制。在 vLLM 中，每个注意力层除了标准的 MLA KV cache 之外，还额外维护一个 `indexer.k_cache` tensor，用于存储压缩后的 key 表示，供稀疏注意力在推理时选择相关 chunk。

**核心挑战**：FlexKV 原有的存储和传输层只处理主 KV cache。若仅卸载主 KV cache 而不同步卸载 indexer cache，则在 prefill 阶段从 CPU/SSD 恢复 KV cache 时，indexer cache 的内容与主 cache 不一致，导致稀疏注意力选择错误的 chunk，产生精度问题。

**约束条件**：
- 两种 cache 共享同一套 block table 和 slot mapping，block ID 一一对应
- FlexKV 的 radix tree 匹配、block 管理、任务调度逻辑不应被修改
- 对不使用 NSA 的模型（Qwen、LLaMA 等）完全透明，零额外开销

---

## 2. 整体设计

### 2.1 核心思路：Shadow Worker 模式

不修改现有的主 KV cache 传输路径，而是为 indexer cache 创建一套**影子（Shadow）**存储和传输资源：

- **GPU_INDEXER / CPU_INDEXER**：新增两个 `DeviceType` 枚举值，在 `StorageEngine` 中独立管理 indexer cache 的 GPU 和 CPU 存储 handle
- **Indexer Worker**：在 `TransferEngine` 中为每个 dp group 创建一个独立的 `GPUCPUTransferWorker`，与主 worker 并行运行
- **同步触发**：当主 worker 收到一个 GPU↔CPU 传输 op 时，indexer worker 同步收到相同的 op（相同的 block IDs），两者并行执行

### 2.2 数据流

**vLLM 路径**：

```
vLLM register_kv_caches(kv_caches_dict)
         │
         ▼
FlexKVWorkerConnector.register_to_server()
         │  按 layer name 中是否含 ".k_cache" 分组
         ├── main_kv_caches  → 主 KV cache layout + handles（LAYERFIRST）
         └── indexer_kv_caches → indexer layout + handles（LAYERFIRST）
         │
         ▼
KVTPClient.register_to_server()
         │  构建 TensorSharedHandle（IPC 共享内存句柄）
         │
         ▼
RegisterTPClientRequest（含 indexer_handles / indexer_layout / indexer_dtype）
```

**TRT-LLM 路径**：

```
TRT-LLM resource_manager 初始化
         │
         ├── FlexKVWorkerConnector.register_kv_caches(kv_cache_tensor)
         │       存储 pending 主 cache（BLOCKFIRST 单 tensor）
         │
         ├── FlexKVWorkerConnector.register_indexer_kv_caches(kv_cache_manager)
         │       逐层调用 get_indexer_k_cache_pool_data(layer_idx)
         │       reshape 为 [num_blocks, block_size, head_size]
         │       存储 pending indexer cache（LAYERFIRST per-layer tensors）
         │
         └── FlexKVWorkerConnector.flush_registration()
                 合并主 cache + indexer cache，一次性发送
                 KVTPClient.register_to_server(...)
         │
         ▼
RegisterTPClientRequest（含 indexer_handles / indexer_layout / indexer_dtype）
```

**公共路径（两种引擎共用）**：

```
RegisterTPClientRequest
         │
         ▼
TransferManager._handle_gpu_blocks_registration()
         │
         ├── StorageEngine.register_gpu_blocks()      → DeviceType.GPU handle
         └── StorageEngine.register_indexer_blocks()  → DeviceType.GPU_INDEXER handle
                                                          + DeviceType.CPU_INDEXER handle（自动分配）
         │
         ▼
TransferEngine.__init__()
         ├── 主 GPUCPUTransferWorker（每个 dp group 一个）
         └── 影子 indexer GPUCPUTransferWorker（每个 dp group 一个）
```

### 2.3 传输触发逻辑

在 `TransferEngine._assign_op_to_worker()` 中：

1. 将 op 提交给主 worker（原有逻辑不变）
2. 若 `_has_indexer == True` 且 op 类型为 `H2D` 或 `D2H`：
   - 在 `pin_buffer.lock` 保护下，将 `op.src_slot_id` 和 `op.dst_slot_id` 的引用计数各加 1，防止主 worker 完成后过早释放 slot
   - 将 op 存入 `op_id_to_indexer_op` 字典
   - 将同一个 op 提交给 indexer worker

3. Indexer worker 完成后，将 `op_id` 放入 `indexer_finished_ops_queue`
4. Scheduler loop 收到 `indexer_finished` 事件后，从 `op_id_to_indexer_op` 取出 op，调用 `free_op_from_buffer` 释放额外持有的引用计数

### 2.4 Pin Buffer 引用计数管理

`SharedOpPool` 使用引用计数管理 slot 复用。正常情况下每个 slot 的引用计数为 1，主 worker 完成后减为 0 并释放。

引入 indexer worker 后，同一个 slot 被两个 worker 使用，需要引用计数为 2：

```
register_op_to_buffer(op)          → slot_ref_count = 1（allocate_slot 内部）
_assign_op_to_worker:
  with pin_buffer.lock:
    slot_ref_count[src_slot_id] += 1  → slot_ref_count = 2

主 worker 完成 → free_op_from_buffer → slot_ref_count = 1（不释放）
indexer worker 完成 → free_op_from_buffer → slot_ref_count = 0（释放）
```

**关键约束**：引用计数的修改必须在 `pin_buffer.lock` 保护下进行，与 `allocate_slot` / `free_slot` 的加锁操作互斥，避免竞态条件。

---

## 3. 存储层设计

### 3.1 新增 DeviceType

```python
class DeviceType(IntEnum):
    GPU         = 0
    CPU         = 1
    SSD         = 2
    REMOTE      = 3
    GPU_INDEXER = 6   # 新增：GPU 上的 indexer cache
    CPU_INDEXER = 7   # 新增：CPU 上的 indexer cache
```

### 3.2 StorageEngine.register_indexer_blocks()

注册 GPU indexer cache blocks，并在 CPU offload 启用时自动分配 CPU_INDEXER buffer：

- GPU_INDEXER：复用 `GPUAllocator.from_raw_data()`，接受 `List[TensorSharedHandle]`
- CPU_INDEXER：复用 `CPUAllocator.allocate()`，layout 与 GPU_INDEXER 一致，block 数量与主 CPU cache 相同

---

## 4. 传输层设计

### 4.1 TransferEngine 扩展

新增参数：
- `indexer_gpu_handles: Optional[Dict[int, List[StorageHandle]]]`：dp_client_id → indexer GPU handles
- `indexer_cpu_handle: Optional[StorageHandle]`：indexer CPU handle

新增成员：
- `indexer_gpucpu_workers: List[WorkerHandle]`：影子 indexer worker 列表
- `indexer_finished_ops_queue: mp.Queue`：indexer worker 专用完成队列
- `op_id_to_indexer_op: Dict[int, TransferOp]`：追踪已提交给 indexer worker 的 op

### 4.2 Worker 创建

Indexer worker 与主 worker 使用相同的 `GPUCPUTransferWorker` / `tpGPUCPUTransferWorker` 实现，区别在于：
- `gpu_blocks` 指向 indexer GPU handle 的 tensor list
- `cpu_blocks` 指向 indexer CPU handle 的 tensor
- `finished_ops_queue` 使用独立的 `indexer_finished_ops_queue`

### 4.3 Scheduler Loop 扩展

在 `selectors` 中额外注册 `indexer_finished_ops_queue._reader`，事件类型为 `"indexer_finished"`。收到事件后：
1. 批量 drain `indexer_finished_ops_queue`
2. 对每个完成的 `op_id`，从 `op_id_to_indexer_op` 取出 op，调用 `free_op_from_buffer` 释放额外引用

---

## 5. 注册流程设计

### 5.1 vLLM 适配层（FlexKVWorkerConnector）

`register_to_server()` 中按 layer name 分组：
- 含 `.k_cache` 的层 → `indexer_kv_caches`
- 其余层 → `main_kv_caches`

分别构建 layout（均为 LAYERFIRST），通过 `KVTPClient.register_to_server()` 一次性发送。

### 5.2 TRT-LLM 适配层（FlexKVWorkerConnector）

**内存布局差异**：TRT-LLM 将所有层的 KV cache 存储为一个连续的 4D tensor（BLOCKFIRST），而 vLLM 提供 per-layer tensor 列表（LAYERFIRST）。FlexKV 的 transfer worker 通过 stride-based 寻址同时支持两种布局，无需额外转换（零拷贝）。

| 引擎 | 主 KV cache 格式 | Layout 类型 |
|---|---|---|
| vLLM | `List[Tensor]`，每层一个 tensor | LAYERFIRST |
| TRT-LLM | 单个 `[num_blocks, num_layers, kv_factor, block_size_dim]` tensor | BLOCKFIRST |
| Indexer cache（两者） | `List[Tensor]`，每层一个 tensor | LAYERFIRST |

**延迟注册（Deferred Registration）模式**：TRT-LLM 的主 cache 和 indexer cache 在不同时机初始化，因此采用三步注册：

1. `register_kv_caches(kv_cache_tensor)` — 存储主 cache 的 pending 状态（BLOCKFIRST 单 tensor）
2. `register_indexer_kv_caches(kv_cache_manager)` — 逐层调用 `get_indexer_k_cache_pool_data(layer_idx)`，reshape 为 `[num_blocks, block_size, head_size]`，存储 pending 状态
3. `flush_registration()` — 将主 cache 和 indexer cache 合并为一个 `RegisterTPClientRequest` 原子发送

这样确保两种 cache 在同一个请求中注册，避免 TransferManager 在 indexer 数据到达前就初始化 TransferEngine。

**无 indexer 时的兼容性**：若 TRT-LLM 未调用 `register_indexer_kv_caches()`，`flush_registration()` 会检测 `_pending_indexer_blocks` 不存在，直接以 `indexer_caches=None` 发送，行为与不支持 NSA 的模型完全一致。

### 5.3 TransferManager 接收

`_handle_gpu_blocks_registration()` 中：
- 若 `req.indexer_handles is not None`，将 indexer 数据存入 `all_indexer_blocks` / `all_indexer_layouts` / `all_indexer_dtypes`

`initialize_transfer_engine()` 中：
- 所有 GPU 注册完成后，调用 `StorageEngine.register_indexer_blocks()` 注册 indexer blocks
- 按 dp_client_id 分组 indexer GPU handles，传给 `TransferEngine`

---

## 6. 向后兼容性

| 场景 | 行为 |
|---|---|
| 不含 `.k_cache` 的模型（Qwen、LLaMA 等） | `indexer_handles=None`，`_has_indexer=False`，所有 indexer 路径跳过 |
| 含 `.k_cache` 但未启用 CPU offload | `indexer_cpu_handle=None`，不创建 indexer worker |
| 含 `.k_cache` 且启用 CPU offload | 完整 shadow worker 路径 |

所有 indexer 相关字段均为可选，默认值为 `None`，不影响现有代码路径。

---

## 7. 已知限制

1. **共享 block IDs**：indexer worker 使用与主 worker 完全相同的 block IDs，这依赖于两种 cache 共享 block table 的假设。若未来 indexer cache 使用独立的 block table，需要额外适配。
2. **无独立失败处理**：indexer worker 的传输失败不会反馈给上层调度器，仅记录日志。
