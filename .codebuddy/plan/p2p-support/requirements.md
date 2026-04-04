# 需求文档：FlexKV P2P（点对点）功能支持

## 引言

FlexKV 是一个面向大规模 LLM 推理场景的分布式 KV 缓存管理系统，当前已支持三级缓存（GPU → CPU → SSD/远程存储）、分布式 KVCache 复用（基于 Mooncake Transfer Engine + Redis 元数据管理），以及与 vLLM 和 TensorRT-LLM 的集成。

本需求旨在为 FlexKV 添加完整的 **P2P（点对点）通信功能**，使分布式节点间能够直接进行 KVCache 数据传输，而无需经过中心化的存储中间层。核心目标是：

1. **优先完成与 sglang 推理引擎的深度集成**，利用 sglang 已有的 `FlexKVRadixCache` 集成点，扩展 P2P 分布式能力。
2. **设计统一的推理引擎适配层**，使 P2P 功能能够以最小改动兼容 vLLM 和 TensorRT-LLM。
3. **确保 P2P 功能在多种部署场景下的稳定性和可靠性**，包括多节点、多实例、TP/DP 并行等场景。

### 现有架构分析

- **FlexKV 已有 P2P 基础设施**：`PEER2CPUTransferWorker`、`TransferType.PEERH2H`/`PEERSSD2H`、`enable_p2p_cpu`/`enable_p2p_ssd` 配置项、`RedisMetaChannel` + `DistributedRadixTree`/`LocalRadixTree` 分布式索引、`MoonCakeTransferEngineWrapper` RDMA 传输。
- **sglang 已有 FlexKV 集成**：`FlexKVRadixCache`（继承 `RadixCache`）已在 sglang 中实现，支持 KV 缓存的 get/put 操作和层级加载事件。sglang 还有独立的 disaggregation 框架（`BaseKVManager`/`BaseKVSender`/`BaseKVReceiver`）。
- **vLLM 已有 FlexKV 集成**：`FlexKVConnectorV1` 通过 vLLM 的 `KVConnector` 接口实现，已合入 vLLM 主线。
- **TensorRT-LLM 已有 FlexKV 集成**：通过 `trtllm_adapter.py` 实现。

### 关键技术约束

- P2P 传输依赖 Mooncake Transfer Engine（RDMA），需要 `FLEXKV_ENABLE_P2P=1` 编译选项。
- 分布式元数据管理依赖 Redis（`RedisMetaChannel`、`RedisNodeInfo`）。
- 不同推理引擎的 KV 缓存布局（`KVCacheLayout`）可能不同（vLLM 格式、SGLang 格式、TRT-LLM 格式），P2P 传输需处理布局转换。
- sglang 使用 `RadixCache` 进行前缀匹配，FlexKV 使用 `CRadixTreeIndex`，两者需要协调。

---

## 需求

### 需求 1：P2P 网络拓扑管理与节点发现

**用户故事：** 作为一名分布式推理系统运维人员，我希望 FlexKV 能够自动发现和管理 P2P 网络中的节点，以便在多节点部署时无需手动配置每个节点的连接信息。

#### 验收标准

1. WHEN 一个新的 FlexKV 节点启动 AND P2P 功能已启用 THEN 系统 SHALL 自动向 Redis 注册节点信息（包括 node_id、IP 地址、Mooncake 引擎地址、ZMQ 地址、CPU/SSD 缓冲区指针）。
2. WHEN 一个 FlexKV 节点正常关闭或异常退出 THEN 系统 SHALL 自动从 Redis 注销该节点信息，并通过 pub/sub 通知其他节点更新活跃节点列表。
3. WHEN 系统检测到某个远程节点不可达（心跳超时或连接失败） THEN 系统 SHALL 将该节点标记为不活跃，并在后续的 P2P 传输中跳过该节点。
4. IF 多个 FlexKV 实例运行在同一物理节点上 THEN 系统 SHALL 为每个实例分配唯一的 node_id，并正确管理各实例的 P2P 连接。
5. WHEN 节点拓扑发生变化（新增或移除节点） THEN 系统 SHALL 在 `FLEXKV_REBUILD_INTERVAL_MS`（默认 10 秒）内完成拓扑更新，且不中断正在进行的 P2P 传输。

### 需求 2：P2P 数据传输协议与序列化

**用户故事：** 作为一名推理引擎开发者，我希望 FlexKV 的 P2P 传输能够高效地在节点间直接传输 KVCache 数据，以便减少跨节点推理的延迟。

#### 验收标准

1. WHEN 本地节点需要从远程节点获取 KVCache 数据 AND 远程节点的数据在 CPU 内存中 THEN 系统 SHALL 通过 Mooncake RDMA 引擎直接读取远程 CPU 内存中的数据到本地 CPU 内存（`PEERH2H` 传输类型）。
2. WHEN 本地节点需要从远程节点获取 KVCache 数据 AND 远程节点的数据在 SSD 中 THEN 系统 SHALL 通过 ZMQ 通知远程节点将数据从 SSD 加载到 CPU，然后通过 RDMA 传输到本地（`PEERSSD2H` 传输类型）。
3. WHEN 源节点和目标节点的 KV 缓存布局不同 THEN 系统 SHALL 在传输过程中自动进行布局转换，确保数据在目标节点上可直接使用。
4. WHEN P2P 传输操作提交后 THEN 系统 SHALL 支持异步执行，传输时间可与推理计算重叠。
5. IF 单次 P2P 传输涉及多个不连续的块 THEN 系统 SHALL 将连续块合并为批量传输（`batch_transfer_sync_read`），以减少 RDMA 操作次数。
6. WHEN P2P 传输完成 THEN 系统 SHALL 通过 `CompletedOp` 回调通知上层，包含传输的块数、字节数和传输类型信息。

### 需求 3：分布式 KVCache 索引与前缀匹配

**用户故事：** 作为一名推理引擎开发者，我希望 FlexKV 能够在分布式环境中高效地查找和复用其他节点上的 KVCache，以便最大化缓存命中率。

#### 验收标准

1. WHEN 一个新的推理请求到达 AND 本地缓存未命中 THEN 系统 SHALL 查询 `DistributedRadixTree` 中的远程索引，找到拥有最长前缀匹配的远程节点。
2. WHEN 本地节点成功存储新的 KVCache 块 THEN 系统 SHALL 通过 `LocalRadixTree.insert_and_publish` 将块元数据异步发布到 Redis，供其他节点发现。
3. WHEN 远程索引查询返回多个候选节点 THEN 系统 SHALL 选择匹配长度最长的节点，若匹配长度相同则优先选择网络距离最近的节点。
4. IF 远程节点上的 KVCache 块正在被驱逐（`NODE_STATE_ABOUT_TO_EVICT`） THEN 系统 SHALL 通过 Lease 机制确保在传输完成前数据不被实际删除。
5. WHEN `DistributedRadixTree` 后台刷新线程运行时 THEN 系统 SHALL 以 `FLEXKV_REFRESH_BATCH_SIZE`（默认 256）为批次从 Redis 拉取远程节点的块元数据，并重建本地的远程索引快照。
6. WHEN 本地节点执行缓存驱逐 THEN 系统 SHALL 批量更新 Redis 中对应块的状态为 `NODE_STATE_EVICTED`，并从远程索引中移除。

### 需求 4：与 sglang 推理引擎的深度集成

**用户故事：** 作为一名使用 sglang 的推理服务开发者，我希望 FlexKV 的 P2P 功能能够无缝集成到 sglang 中，以便在多节点 sglang 部署中实现跨节点 KVCache 共享。

#### 验收标准

1. WHEN sglang 的 `FlexKVRadixCache` 初始化时 AND P2P 功能已启用 THEN 系统 SHALL 初始化 `DistributedRadixTree` 和 `LocalRadixTree`，并启动 Redis 元数据通道和后台刷新线程。
2. WHEN sglang 调用 `match_prefix` 进行前缀匹配时 THEN 系统 SHALL 同时查询本地 RadixTree 和分布式 RadixTree，返回包含远程匹配信息（node_id、匹配长度）的结果。
3. WHEN sglang 的调度器决定从远程节点加载 KVCache 时 THEN 系统 SHALL 生成包含 `PEERH2H` 或 `PEERSSD2H` 操作的 `TransferOpGraph`，并提交给 `TransferEngine` 执行。
4. WHEN P2P 传输完成后 THEN 系统 SHALL 通过 `FlexKVLayerLoadingEvent` 的 eventfd 机制通知 sglang 的 attention 层，支持层级粒度的流水线加载。
5. IF sglang 使用 MLA（Multi-head Latent Attention）架构 THEN 系统 SHALL 正确处理 MLA 特有的 KV 缓存格式（单 KV head、latent + rope 维度拼接）。
6. WHEN sglang 的 `FlexKVRadixCache` 执行 `put` 操作存储新的 KVCache 时 THEN 系统 SHALL 在本地存储完成后，通过 `LocalRadixTree.insert_and_publish` 将元数据发布到分布式索引。
7. WHEN sglang 配置了 TP（张量并行）> 1 THEN 系统 SHALL 确保所有 TP rank 的 GPU 块都正确注册到 `TransferEngine`，且 P2P 传输覆盖所有 TP rank 的数据。

### 需求 5：统一推理引擎适配层

**用户故事：** 作为一名 FlexKV 维护者，我希望 P2P 功能的实现能够通过统一的适配层对接不同的推理引擎，以便减少代码重复并简化后续维护。

#### 验收标准

1. WHEN FlexKV 的 P2P 功能需要与新的推理引擎集成时 THEN 系统 SHALL 提供一个抽象的 `P2PAdapter` 接口，定义 `init_p2p`、`match_remote`、`transfer_remote`、`notify_transfer_complete` 等标准方法。
2. WHEN `FlexKVConfig` 从不同推理引擎初始化时（`post_init_from_sglang_config`、`post_init_from_vllm_config`、`post_init_from_trt_config`） THEN 系统 SHALL 自动检测并配置 P2P 相关参数（`enable_p2p_cpu`、`enable_p2p_ssd`、Redis 连接信息、Mooncake 配置）。
3. IF 推理引擎使用不同的 KV 缓存布局类型 THEN 系统 SHALL 在 `P2PAdapter` 中提供布局转换钩子，允许每个引擎适配器定义自己的布局映射逻辑。
4. WHEN vLLM 的 `FlexKVConnectorV1` 需要使用 P2P 功能时 THEN 系统 SHALL 通过现有的 `KVManager` 接口透明地支持 P2P 传输，无需修改 vLLM 侧的代码。
5. WHEN TensorRT-LLM 的 `trtllm_adapter` 需要使用 P2P 功能时 THEN 系统 SHALL 通过现有的 `KVManager` 接口透明地支持 P2P 传输，无需修改 TRT-LLM 侧的代码。

### 需求 6：负载均衡与容错处理

**用户故事：** 作为一名分布式推理系统运维人员，我希望 FlexKV 的 P2P 功能能够在节点故障或网络异常时自动恢复，以便保证推理服务的高可用性。

#### 验收标准

1. WHEN P2P 传输操作超时（默认 5 秒） THEN 系统 SHALL 将该传输标记为失败，并回退到本地重新计算 KVCache 的路径。
2. WHEN 远程节点在 P2P 传输过程中宕机 THEN 系统 SHALL 检测到连接断开，取消未完成的传输操作，并通知上层调度器重新调度。
3. IF 多个远程节点都拥有所需的 KVCache 前缀 THEN 系统 SHALL 基于以下因素进行负载均衡选择：匹配长度（优先最长）、节点负载（优先低负载）、网络距离（优先近距离）。
4. WHEN Redis 元数据服务暂时不可用 THEN 系统 SHALL 使用本地缓存的远程索引快照继续提供服务，并在 Redis 恢复后自动重新同步。
5. WHEN Mooncake RDMA 引擎初始化失败（例如缺少 RDMA 设备） THEN 系统 SHALL 优雅降级，禁用 P2P 功能并记录警告日志，不影响本地缓存功能的正常运行。
6. WHEN 系统检测到某个远程节点的 Lease 已过期（超过 `FLEXKV_LEASE_TTL_MS`） THEN 系统 SHALL 自动清理该节点在本地远程索引中的所有条目，避免引用过期数据。

### 需求 7：性能优化与监控

**用户故事：** 作为一名性能工程师，我希望 FlexKV 的 P2P 功能提供详细的性能指标和优化手段，以便我能够监控和调优分布式 KVCache 共享的效率。

#### 验收标准

1. WHEN P2P 功能启用且 `FLEXKV_ENABLE_METRICS=1` THEN 系统 SHALL 通过 Prometheus 暴露以下指标：P2P 传输次数、P2P 传输字节数、P2P 传输延迟分布、远程缓存命中率、远程索引大小。
2. WHEN P2P 传输操作执行时 THEN 系统 SHALL 通过 NVTX 标记传输的各个阶段（匹配、传输、完成），支持 Nsight Systems 性能分析。
3. IF 用户配置了 `FLEXKV_ENABLE_TRACE=1` THEN 系统 SHALL 记录每次 P2P 传输的详细日志，包括源节点、目标节点、块数量、传输时间、传输类型。
4. WHEN 连续块可以合并传输时 THEN 系统 SHALL 自动合并为批量 RDMA 操作，减少网络往返次数。
5. WHEN P2P 传输与本地 GPU↔CPU 传输同时进行时 THEN 系统 SHALL 确保两者不互相阻塞，通过独立的 worker 进程/线程实现并行执行。

### 需求 8：配置与部署

**用户故事：** 作为一名推理服务部署人员，我希望 FlexKV 的 P2P 功能能够通过简单的配置启用，以便快速在现有部署中开启跨节点 KVCache 共享。

#### 验收标准

1. WHEN 用户在 FlexKV 配置文件中设置 `enable_p2p_cpu: true` THEN 系统 SHALL 启用基于 CPU 内存的 P2P 传输功能。
2. WHEN 用户在 FlexKV 配置文件中设置 `enable_p2p_ssd: true` THEN 系统 SHALL 启用基于 SSD 的 P2P 传输功能（需要同时启用 SSD 缓存）。
3. WHEN P2P 功能启用时 THEN 系统 SHALL 要求用户配置 Redis 连接信息（`redis_host`、`redis_port`、`redis_password`）和 Mooncake 引擎配置（通过 `MOONCAKE_CONFIG_PATH` 环境变量）。
4. WHEN 编译 FlexKV 时未设置 `FLEXKV_ENABLE_P2P=1` AND 用户尝试启用 P2P 功能 THEN 系统 SHALL 抛出明确的错误信息，指导用户重新编译。
5. IF 用户使用 sglang 部署 THEN 系统 SHALL 支持通过 sglang 的 `--hicache-config` 参数传递 FlexKV P2P 配置。
6. IF 用户使用 vLLM 部署 THEN 系统 SHALL 支持通过 `FLEXKV_CONFIG_PATH` 环境变量传递 P2P 配置，无需修改 vLLM 启动参数。
7. IF 用户使用 TensorRT-LLM 部署 THEN 系统 SHALL 支持通过 `flexkv_config.json` 配置文件传递 P2P 配置。

---

## 边界情况与技术限制

1. **RDMA 硬件依赖**：P2P 传输依赖 RDMA 网络设备（InfiniBand/RoCE），在没有 RDMA 硬件的环境中 P2P 功能不可用，系统应优雅降级。
2. **跨数据中心场景**：当前 P2P 设计主要面向同一数据中心内的节点间通信，跨数据中心的高延迟场景暂不在本期范围内。
3. **内存安全**：P2P 传输涉及远程内存直接访问，需要确保 Lease 机制正确防止 use-after-free 问题。
4. **版本兼容性**：P2P 功能需要与 sglang 最新版本、vLLM v0.17.2+、TensorRT-LLM v1.1.0+ 兼容。
5. **多租户隔离**：在多租户场景下，P2P 传输应尊重 FlexKV 的 namespace 隔离机制，不同 namespace 的 KVCache 不应跨节点共享。
