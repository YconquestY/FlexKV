# 实施计划

- [ ] 1. PP模式下 `num_layers` 一致性校验与防护
   - 在 `flexkv/common/config.py` 的 `ModelConfig` 中新增 `num_local_layers` 属性（或 property），当 `pp_size > 1` 时返回 `num_layers // pp_size`，否则返回 `num_layers`
   - 在 `flexkv/transfer/layerwise.py` 的 `LayerwiseTransferWorker.__init__` 中添加断言：验证 `gpu_kv_layouts[0].num_layer` 与实际 GPU blocks 数量的一致性，当 `pp_size > 1` 时确保 `num_layers` 为本地层数而非总层数
   - 在 `flexkv/cache/cache_engine.py` 的 `get()` 和 `put()` 方法中，当 `layer_num == -1` 时使用 `model_config.num_layers`（需确保调用方已将其设置为本地层数），添加防御性检查：若 `pp_size > 1` 且 `num_layers` 大于预期本地层数则报错
   - _需求：1.1、1.2、1.3、1.4_

- [ ] 2. Indexer Layerwise Worker 同步机制验证与修复
   - 在 `flexkv/transfer/transfer_engine.py` 的 `_indexer_layerwise_workers` 创建处（约第 L480 行），确认 `enable_eventfd=False` 时 `LayerwiseTransferGroup` 构造函数接收空的 `layer_eventfds_tensor`，验证 C++ 层面 `layerwise.cpp` 中对空 eventfd tensor 的处理逻辑正确
   - 在 `flexkv/transfer/transfer_engine.py` 的 `_assign_op_to_worker` 方法中，验证当 `_has_indexer=True` 且 `transfer_type == TransferType.LAYERWISE` 时，`pending_count` 正确递增为 2（主KV + Indexer），确保 `_finalize_op` 在两者都完成后才触发
   - 在 `csrc/layerwise.cpp` 中检查当 `eventfds` tensor 为空时，`layerwise_transfer` 函数是否跳过 eventfd 通知逻辑，确保不会出现段错误或死锁
   - _需求：2.1、2.3、2.4_

- [ ] 3. Indexer 在 TP 模式下的 Worker 类型与布局验证
   - 在 `flexkv/transfer/transfer_engine.py` 中验证：当 `tp_size > 1` 时，Indexer H2D/D2H workers 使用 `tpGPUCPUTransferWorker`（当前代码已正确区分），确认 `gpu_blocks` 参数传递了所有 TP rank 的 handle list
   - 在 `flexkv/transfer/transfer_engine.py` 中验证 Indexer Layerwise worker 的 `tp_group_size` 参数与主 KV 的 Layerwise worker 一致，且 `gpu_kv_layouts` 列表长度等于 `tp_size`
   - 编写单元测试：模拟 `tp_size=2` 场景，验证 Indexer 的 `gpu_kv_layouts` 中 `is_mla=True`、`num_kv_heads=1`、`head_size=qk_rope_head_dim` 参数正确，且 CPU 布局的 `tp_stride` 设置正确
   - _需求：3.1、3.2、3.3、3.4_

- [ ] 4. PP 模式下 Namespace 跨 Rank 一致性保障
   - 在 `flexkv/cache/cache_engine.py` 的 `get()` 和 `put()` 方法中，确认 `namespace` 参数正确传递到 `SequenceMeta` 构造函数，并最终影响 `block_hashes` 的计算
   - 在 `flexkv/common/block.py` 的 `SequenceMeta.gen_hashes()` 中验证 namespace 参与 hash 计算的方式，确保不同 PP rank 对相同 `(token_ids, namespace)` 组合生成相同的 hash 值
   - 添加防御性日志：当 `pp_size > 1` 且 namespace 非空时，记录当前 `pp_rank` 和 namespace 信息，便于跨 rank 一致性调试
   - _需求：4.1、4.2、4.3_

- [ ] 5. Indexer 与主 KV 的 Eviction 一致性保障
   - 在 `flexkv/cache/cache_engine.py` 中分析 `CacheEngineV2.take()` 和 `CacheEngineV1.take()` 的 eviction 逻辑，确认 eviction 操作同时释放主 KV 和 Indexer 的 block（当前 Indexer 和主 KV 共享相同的 block ID 映射）
   - 在 `flexkv/transfer/transfer_engine.py` 的 `_finalize_op` 方法中验证：当 `pending_count` 从 2 降到 0 时（主 KV + Indexer 都完成），才释放 pin buffer 和通知上层，确保 eviction 不会在传输未完成时释放 block
   - 添加断言：在 eviction 路径中，若 `_has_indexer=True`，确保被淘汰的 block 在 Indexer 的存储中也被正确标记或释放
   - _需求：5.1、5.2、5.3_

- [ ] 6. Indexer TP 模式下的 GPU 注册与传输正确性测试
   - 在 `flexkv/common/storage.py` 的 `StorageEngine.register_gpu_blocks` 方法中，验证 `indexer_gpu_blocks` 和 `indexer_gpu_layout` 参数在 `tp_size > 1` 时正确处理每个 TP rank 的指针和布局
   - 编写集成测试：模拟 `tp_size=2` 且 `indexer is not None` 的场景，执行 H2D 和 D2H 传输，验证每个 TP rank 的 Indexer 数据传输后与源数据一致
   - 在测试中验证：多 GPU 同时执行 Indexer 传输时 `pending_count` 正确归零，且各 GPU 的传输操作互不干扰
   - _需求：6.1、6.2、6.3、6.4_

- [ ] 7. Redis Key 中注入 `pp_rank` 实现 PP 模式下的元数据隔离
   - [ ] 7.1 修改 `flexkv/cache/hie_cache_engine.py` 的 `start()` 方法
      - 将 block key 前缀从固定的 `"CPUB"` / `"SSDB"` / `"PCFSB"` 改为包含 `pp_rank` 的格式（如 `"CPUB:pp0"` / `"SSDB:pp1"`）
      - 需要在 `HierarchyLRCacheEngine.__init__` 或 `start()` 中接收 `pp_rank` 和 `pp_size` 参数
      - 当 `pp_size == 1` 时保持原有 key 格式不变，确保向后兼容
      - _需求：7.1、7.2、7.6_
   - [ ] 7.2 修改 `flexkv/cache/redis_meta.py` 中的 Key 构造
      - 在 `RedisMetaChannel.__init__` 中增加可选的 `pp_rank` 参数，将其编码到 `blocks_key` 前缀中
      - 修改 `RedisMeta.regist_node_meta()` 的 key 格式：从 `meta:{node_id}` 改为 `meta:{node_id}:pp{pp_rank}`（当 `pp_size > 1` 时）
      - 修改 `RedisMeta.regist_buffer()` 的 key 格式：从 `buffer:{node_id}:{ptr}` 改为 `buffer:{node_id}:pp{pp_rank}:{ptr}`（当 `pp_size > 1` 时）
      - 修改 `RedisMeta.register_node()` 中 `node:{node_id}` 的 hash fields，增加 `pp_rank` 字段
      - _需求：7.3、7.4_
   - [ ] 7.3 修改 `flexkv/cache/redis_meta.py` 中的清理逻辑
      - 修改 `RedisMeta.unregister_node()` 和 `RedisNodeInfo.unregister_node()`，确保只清理当前 `pp_rank` 对应的 Redis key
      - 修改 `RedisNodeInfo.scan_active_nodes()` 以正确处理包含 `pp_rank` 的 key 格式
      - _需求：7.5、7.6_
   - [ ] 7.4 修改 `flexkv/cache/cache_engine.py` 中 Redis 初始化调用链
      - 在 `CacheEngine` 初始化 `HierarchyLRCacheEngine` 时，将 `model_config.pp_rank` 和 `model_config.pp_size` 传递到 `from_cache_config` 和 `start()` 方法
      - 确保 `get_redis_meta_channel()` 调用时传入包含 `pp_rank` 的 `blocks_key`
      - _需求：7.1、7.2_

- [ ] 8. 端到端集成测试
   - 编写 PP + P2P 场景的集成测试：模拟 `pp_size=2`、`enable_p2p_cpu=True`，验证两个 PP rank 的 Redis key 完全隔离，互不干扰
   - 编写 PP + Layerwise 场景的集成测试：模拟 `pp_size=2`、`enable_layerwise_transfer=True`，验证每个 PP rank 只传输本地层数的数据
   - 编写 DSA Indexer + TP + Eviction 场景的集成测试：模拟 `tp_size=2`、`indexer is not None`，验证 eviction 时主 KV 和 Indexer 数据一致性
   - _需求：1、2、3、4、5、6、7_
