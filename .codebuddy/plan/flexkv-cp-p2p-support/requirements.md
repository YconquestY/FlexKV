# 需求文档：FlexKV 在 sglang Context Parallelism 模式下支持 P2P 功能

## 引言

sglang 的 Context Parallelism（CP）用于 Prefill 阶段，采用 "Split-Q, Gather-KV" + Zigzag 负载均衡策略。CP 模式下，每个 CP rank 在 Prefill 完成后通过 AllGather + 逆 Zigzag 操作恢复完整的自然顺序 KV cache。这意味着 FlexKV 的 P2P 核心功能（block hash 计算、radix tree 前缀匹配、KV 传输）天然兼容 CP 模式，**不需要修改 FlexKV 核心库**（`/data/workspace/FlexKV`）。

但当前 FlexKV 在 sglang 中的集成代码（`flexkv_radix_cache.py` 和 `scheduler.py`）缺少 CP group 感知，导致 **store（发送端）** 和 **load（接收端）** 两个方向都存在问题。

### 背景：Rank 布局

sglang DSA+CP 模式下的 rank 布局为 `(dp, cp, tp)`：
```
tp_rank = (attn_dp_rank * attn_cp_size + attn_cp_rank) * attn_tp_size + attn_tp_rank
```

例如 `tp_size=8, dp_size=2, cp_size=2`，则 `attn_tp_size=2`：
```
DP0: [GPU0(cp0,atp0), GPU1(cp0,atp1), GPU2(cp1,atp0), GPU3(cp1,atp1)]
DP1: [GPU4(cp0,atp0), GPU5(cp0,atp1), GPU6(cp1,atp0), GPU7(cp1,atp1)]

attn_tp_group: [GPU0,GPU1], [GPU2,GPU3], [GPU4,GPU5], [GPU6,GPU7]
attn_cp_group: [GPU0,GPU2], [GPU1,GPU3], [GPU4,GPU6], [GPU5,GPU7]
```

### 当前问题概览

#### Store 方向（发送端）问题

1. **match_prefix 中 CP group 缺少 broadcast**：`flexkv_hit_length` 和 `flexkv_task_id` 只在 `attn_tp_group` 内 broadcast，CP group 内的其他 rank 收不到匹配结果。
2. **writing_check 中 CP group 缺少 broadcast**：completed/skipped 的 req_ids 只在 `attn_tp_group` 内 broadcast，CP group 内的其他 rank 无法同步 store 完成状态。
3. **冗余 store 问题**：每个 CP rank 都持有完整 KV cache，当前 `rank == 0`（全局 tp_rank）的判断已隐式避免冗余 store，但需要验证。

#### Load 方向（接收端）问题 ⚠️ 关键

4. **slot_mapping 不一致问题**：`init_load_back` 中每个 rank 独立调用 `token_to_kv_pool_allocator.alloc()` 分配 GPU slot，但只有 `rank == 0` 的 `device_indices` 被传给 `kv_manager.launch()` 作为 `slot_mapping`。FlexKV server 根据这个 slot_mapping 往所有已注册的 GPU 写数据。如果 CP rank 0 和 CP rank 1 分配到不同的 slot，CP rank 1 的数据会被写到错误的位置。
   - **当前 TP 模式下为什么没问题**：TP 模式下所有 rank 共享同一个 `token_to_kv_pool_allocator`（通过 NCCL 同步），分配结果一致。但 CP rank 之间的 allocator 是独立的。
5. **register_to_server 的 tp_rank 语义**：`KVTPClient(self.flexkv_config.gpu_register_port, 0, self.rank)` 中 `self.rank` 是全局 `tp_rank`。CP rank 1 的全局 tp_rank（如 GPU2 的 tp_rank=2）注册后，FlexKV server 需要正确识别这是同一个 TP group 内的不同 rank 还是 CP group 内的不同 rank。
6. **Non-layerwise 传输的 barrier 不覆盖 CP group**：`launch_non_layerwise_batch_transfer` 中的 `torch.distributed.barrier(self.tp_group)` 只在 `attn_tp_group` 内同步，CP group 内的 rank 没有 barrier。
7. **Layerwise 传输的 eventfd 不覆盖 CP rank**：`_send_eventfds_to_worker` 只在 `rank == 0` 时有意义（因为 `launch_layerwise_batch_transfer` 只在 rank 0 执行），CP rank > 0 的 layerwise 传输完成通知缺失。
8. **loading_check 缺少 CP 同步**：`loading_check` 中每个 rank 独立检查 `ongoing_load_back` 的完成状态，但 CP rank > 0 没有收到传输完成的通知。

### 涉及文件

| 文件 | 仓库 | 说明 |
|------|------|------|
| `python/sglang/srt/mem_cache/storage/flexkv/flexkv_radix_cache.py` | sglang | FlexKV 在 sglang 中的集成层，包含 `FlexKVRadixCache` 和 `FlexKVConnector` |
| `python/sglang/srt/managers/scheduler.py` | sglang | Scheduler 初始化 FlexKVRadixCache 的入口 |
| `flexkv/integration/config.py` | FlexKV | FlexKVConfig 配置类（可选修改） |

## 需求

---

### 需求 1：FlexKVRadixCache 初始化时传入 CP 参数

**用户故事：** 作为一名 FlexKV 开发者，我希望 FlexKVRadixCache 在初始化时能接收 CP 相关参数（cp_size、cp_rank、cp_group），以便后续的 broadcast 和 barrier 操作能正确覆盖 CP group 内的所有 rank。

#### 验收标准

1. WHEN `scheduler.py` 初始化 `FlexKVRadixCache` THEN 系统 SHALL 传入 `cp_size=self.attn_cp_size`、`cp_rank=self.attn_cp_rank`、`cp_group=self.attn_cp_cpu_group`（当 `attn_cp_size > 1` 时）三个新参数。
2. WHEN `FlexKVRadixCache.__init__` 被调用 THEN 系统 SHALL 保存 `self.cp_size`、`self.cp_rank`、`self.cp_group` 三个属性，并计算 `self.cp_src_rank`（CP group 中 rank 0 的全局 rank）。
3. WHEN `FlexKVConnector.__init__` 被调用 THEN 系统 SHALL 接收并保存 `cp_size`、`cp_rank`、`cp_group` 参数。
4. IF `cp_size == 1`（非 CP 模式）THEN 系统 SHALL 将 `cp_group` 设为 `None`，不影响现有逻辑。

---

### 需求 2：match_prefix 中增加 CP group broadcast（Store 方向）

**用户故事：** 作为一名 FlexKV 开发者，我希望 `match_prefix` 中的 FlexKV 匹配结果能在 CP group 内正确同步，以便所有 CP rank 都能获得一致的 `flexkv_hit_length` 和 `flexkv_task_id`。

#### 验收标准

1. WHEN `match_prefix` 完成 TP broadcast 之后 AND `cp_size > 1` THEN 系统 SHALL 在 CP group 内执行 broadcast，将 `flexkv_hit_length` 和 `flexkv_task_id` 从 CP rank 0 同步到所有 CP rank。
2. WHEN `cp_size == 1` THEN 系统 SHALL 跳过 CP broadcast，不影响现有逻辑。
3. WHEN CP broadcast 完成后 THEN 所有 CP rank 的 `flexkv_hit_length` 和 `flexkv_task_id` SHALL 保持一致。

---

### 需求 3：writing_check 中增加 CP group broadcast（Store 方向）

**用户故事：** 作为一名 FlexKV 开发者，我希望 `writing_check` 中的 store 完成状态能在 CP group 内正确同步，以便所有 CP rank 都能及时释放对应的 TreeNode 锁。

#### 验收标准

1. WHEN `writing_check` 完成 TP broadcast 之后 AND `cp_size > 1` THEN 系统 SHALL 在 CP group 内执行 broadcast，将 completed 和 skipped 的 req_ids 从 CP rank 0 同步到所有 CP rank。
2. WHEN `cp_size == 1` THEN 系统 SHALL 跳过 CP broadcast，不影响现有逻辑。
3. WHEN CP broadcast 完成后 THEN 所有 CP rank SHALL 对相同的 req_ids 执行 `dec_lock_ref` 操作。

---

### 需求 4：确保只有 CP rank 0 执行 store 和 KVManager 操作（Store 方向）

**用户故事：** 作为一名 FlexKV 开发者，我希望在 CP 模式下只有 CP group 中的 rank 0 执行 store/KVManager 操作，以避免冗余的存储和网络开销。

#### 验收标准

1. WHEN CP 模式启用 THEN 只有全局 `tp_rank == 0` 的进程 SHALL 创建 `KVManager` 并执行 `store_kv_async`、`get_match`、`put_match` 操作。（当前已满足，需验证）
2. IF 某个 CP rank 的 `tp_rank != 0` 但 `attn_tp_rank == 0` THEN 该 rank SHALL NOT 创建 KVManager 或执行 store 操作。（当前已满足，因为判断条件是全局 `tp_rank == 0`）

---

### 需求 5：init_load_back 中 slot_mapping 的 CP 一致性（Load 方向）⚠️ 关键

**用户故事：** 作为一名 FlexKV 开发者，我希望 `init_load_back` 中分配的 GPU slot（`device_indices`）在所有 CP rank 之间保持一致，以便 FlexKV server 能将 KV cache 数据正确写入每个 CP rank 的 GPU。

#### 背景

当前 `init_load_back` 流程：
1. 每个 rank 独立调用 `self.token_to_kv_pool_allocator.alloc(host_hit_length)` 分配 GPU slot
2. 只有 `rank == 0` 的 `device_indices` 被传给 `kv_manager.launch()` 作为 `slot_mapping`
3. FlexKV server 根据 `slot_mapping` 往所有已注册的 GPU 写数据（包括 CP rank 1 的 GPU）
4. 如果 CP rank 0 分配到 slot [100, 101, 102]，而 CP rank 1 分配到 slot [200, 201, 202]，FlexKV server 会把数据写到 CP rank 1 的 slot [100, 101, 102]（使用 rank 0 的 mapping），但 CP rank 1 的 radix tree 记录的是 slot [200, 201, 202]，导致数据错位。

#### 验收标准

1. WHEN `init_load_back` 被调用 AND `cp_size > 1` THEN 系统 SHALL 确保所有 CP rank 分配到相同的 `device_indices`。
2. 方案 A（推荐）：WHEN `cp_size > 1` THEN CP rank 0 先分配 `device_indices`，然后通过 CP group broadcast 将 `device_indices` 同步到所有 CP rank，其他 CP rank 使用 rank 0 的分配结果而非独立分配。
3. 方案 B（备选）：WHEN `cp_size > 1` THEN 每个 CP rank 独立分配，但 `launch` 时每个 CP rank 各自提交自己的 `slot_mapping`，FlexKV server 按 rank 分别写入。（需要修改 FlexKV 核心库，不推荐）
4. WHEN `init_load_back` 完成后 THEN 所有 CP rank 的 `device_indices` 和 radix tree 中记录的 slot SHALL 保持一致。
5. WHEN `init_load_back` 中 `device_indices` 通过 CP broadcast 同步后 THEN CP rank > 0 的 `token_to_kv_pool_allocator` 状态 SHALL 正确反映这些 slot 已被占用（需要调用 allocator 的 reserve 或等效操作）。

---

### 需求 6：register_to_server 的 CP 兼容性（Load 方向）

**用户故事：** 作为一名 FlexKV 开发者，我希望 `register_to_server` 在 CP 模式下能正确注册所有 CP rank 的 GPU buffer，以便 FlexKV server 能将 KV cache 数据写入每个 CP rank 的 GPU。

#### 验收标准

1. WHEN CP 模式启用 THEN 所有 CP rank SHALL 各自调用 `register_to_server` 注册自己的 GPU KV cache buffer。
2. WHEN `KVTPClient` 初始化时 THEN 系统 SHALL 传入正确的 `tp_rank` 参数，使 FlexKV server 能区分不同 CP rank 的 GPU buffer。
3. IF 需求 5 采用方案 A（所有 CP rank 使用相同的 slot_mapping）THEN FlexKV server SHALL 能根据 slot_mapping 正确写入所有已注册 GPU 的对应位置。（当前 FlexKV server 已支持按 slot_mapping 写入所有注册的 GPU，需验证 CP 场景下行为正确）

---

### 需求 7：Non-layerwise 传输的 CP barrier（Load 方向）

**用户故事：** 作为一名 FlexKV 开发者，我希望 non-layerwise 传输模式下的 barrier 能覆盖 CP group，以便所有 CP rank 在传输完成后同步继续执行。

#### 验收标准

1. WHEN `launch_non_layerwise_batch_transfer` 完成传输 AND `cp_size > 1` THEN 系统 SHALL 在 CP group 内也执行 barrier，确保所有 CP rank 同步完成。
2. WHEN `cp_size == 1` THEN 系统 SHALL 只执行 TP barrier，不影响现有逻辑。
3. WHEN barrier 完成后 THEN 所有 CP rank 的 GPU 上 SHALL 拥有相同的完整 KV cache 数据。

---

### 需求 8：Layerwise 传输的 CP 兼容性（Load 方向）

**用户故事：** 作为一名 FlexKV 开发者，我希望 layerwise 传输模式在 CP 模式下能正确通知所有 CP rank 传输完成，以便 `loading_check` 能正确释放锁。

#### 背景

当前 layerwise 传输流程：
1. `rank == 0` 调用 `kv_manager.launch(layerwise_transfer=True, counter_id=producer_id)`
2. FlexKV worker 完成每层传输后，通过 eventfd 通知 rank 0
3. `layer_done_counter.events[producer_id]._finished` 被设为 True
4. `loading_check` 检查 `_finished` 状态并释放锁

CP 模式下的问题：
- 只有 rank 0 收到 eventfd 通知
- CP rank > 0 的 `layer_done_counter` 永远不会被更新
- CP rank > 0 的 `loading_check` 永远不会释放锁，导致内存泄漏

#### 验收标准

1. WHEN layerwise 传输完成 AND `cp_size > 1` THEN 系统 SHALL 确保所有 CP rank 的 `loading_check` 能检测到传输完成。
2. 方案 A（推荐）：WHEN `loading_check` 被调用 AND `cp_size > 1` THEN CP rank 0 通过 CP group broadcast 将已完成的 `node_id` 列表同步到所有 CP rank。
3. 方案 B（备选）：WHEN `cp_size > 1` THEN 非 rank 0 的 CP rank 在 `ready_to_load_host_cache` 中将 `producer_id` 设为 `-1`（标记为 non-layerwise 模式），在 non-layerwise barrier 完成后直接标记为完成。
4. WHEN `loading_check` 释放锁后 THEN 所有 CP rank 的 `ongoing_load_back` 状态 SHALL 保持一致。

---

### 需求 9：cache_finished_req 和 cache_unfinished_req 的 CP 兼容性（Store 方向）

**用户故事：** 作为一名 FlexKV 开发者，我希望 `cache_finished_req` 和 `cache_unfinished_req` 在 CP 模式下能正确工作，确保 store 操作不会重复执行。

#### 验收标准

1. WHEN `cache_finished_req` 被调用 AND CP 模式启用 THEN 系统 SHALL 确保 `store_kv_async` 只在全局 `tp_rank == 0` 时执行（当前已满足，`FlexKVConnector.store_kv_async` 内部有 `rank != 0` 的 early return）。
2. WHEN `cache_finished_req` 被调用 THEN 所有 CP rank SHALL 各自维护 `inflight_reqid2node` 映射和 TreeNode 锁引用计数。
3. WHEN `writing_check` 释放锁时 THEN 所有 CP rank SHALL 同步释放相同的 req_ids 对应的锁（依赖需求 3 的 CP broadcast）。

---

### 需求 10：evict 操作的 CP 兼容性

**用户故事：** 作为一名 FlexKV 开发者，我希望 `evict` 操作在 CP 模式下能正确工作，确保所有 CP rank 的 eviction 行为一致。

#### 验收标准

1. WHEN `evict` 被调用 AND CP 模式启用 THEN 系统 SHALL 确保 `writing_check` 的 CP broadcast 正确执行（依赖需求 3）。
2. WHEN `evict` 中 block-wait 剩余 FlexKV tasks 时 THEN 只有 `rank == 0` 的进程 SHALL 执行 `wait_task`，其他 rank 直接释放锁。（当前已满足）
3. WHEN eviction 完成后 THEN 所有 CP rank 的 `inflight_reqid2node` 状态 SHALL 保持一致。

---

### 需求 11：FlexKVConfig 传递 CP 信息（可选优化）

**用户故事：** 作为一名 FlexKV 开发者，我希望 FlexKVConfig 能感知 CP 配置，以便 FlexKV server 端在日志和监控中展示正确的并行信息。

#### 验收标准

1. WHEN `post_init_from_sglang_config` 被调用 THEN 系统 SHALL 接受可选的 `cp_size` 参数并保存到 `self.model_config.cp_size`。
2. IF `cp_size` 未传入 THEN 系统 SHALL 默认 `cp_size=1`，不影响现有逻辑。

---

## 边界情况与技术限制

### 边界情况

1. **cp_size=1（非 CP 模式）**：所有新增逻辑应被跳过，不影响现有行为。
2. **cp_size > 1 且 attn_tp_size=1**：此时每个 DP group 内只有 CP 并行，没有 TP 并行。FlexKV 的 TP broadcast 会被跳过，但 CP broadcast 仍需执行。
3. **PP + CP 组合**：PP sync（`all_reduce MIN`）在 TP rank 0 上执行，CP broadcast 在 PP sync 之后、TP broadcast 之后执行。需确保三者的执行顺序正确。
4. **不同 CP 配置的实例间 P2P**：由于每个 CP rank 都持有完整 KV cache，不同 `cp_size` 的实例间 P2P 匹配和传输应能正常工作。
5. **token_to_kv_pool_allocator 的 CP 一致性**：如果采用需求 5 方案 A（broadcast slot_mapping），需要确保 CP rank > 0 的 allocator 能正确标记这些 slot 为已占用，避免后续分配冲突。

### 技术限制

1. **不修改 FlexKV 核心库**：所有改动限制在 sglang 集成层（`flexkv_radix_cache.py` 和 `scheduler.py`），除非需求 5 采用方案 B。
2. **broadcast 开销**：CP broadcast 会增加额外的通信开销，但由于传输的数据量很小（几个 int/list/tensor），影响可忽略。
3. **CP group 的 ProcessGroup**：使用 sglang 已有的 `attn_cp_cpu_group`（gloo backend），无需创建新的 group。
4. **allocator 的 reserve 能力**：需要确认 `token_to_kv_pool_allocator` 是否支持 "指定 slot 分配" 或 "reserve" 操作，如果不支持则需求 5 方案 A 需要额外适配。

### 成功标准

1. 在 `cp_size=1` 时，所有现有测试通过，行为不变。
2. 在 `cp_size=2, attn_tp_size=2, dp_size=2`（总 tp_size=8）配置下，FlexKV P2P 的 store 和 load 操作正确执行。
3. 所有 CP rank 的 radix tree 状态保持一致。
4. 无冗余 store 操作（只有全局 tp_rank=0 执行 store）。
5. Load 操作后每个 CP rank 的 GPU 上都有完整的 KV cache，且数据位于正确的 slot 位置。
6. Layerwise 和 non-layerwise 两种传输模式在 CP 模式下都能正确工作。
7. `loading_check` 在所有 CP rank 上都能正确检测传输完成并释放锁。
