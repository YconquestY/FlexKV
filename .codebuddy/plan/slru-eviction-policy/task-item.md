# 实施计划

- [ ] 1. C++ 层：新增 SLRU 枚举与 `protected_threshold` 成员
   - 在 `csrc/radix_tree.h` 的 `EvictionPolicy` 枚举中新增 `SLRU` 值
   - 在 `parse_eviction_policy()` 函数中新增 `"slru"` 字符串到 `EvictionPolicy::SLRU` 的映射，并更新错误提示中的策略列表
   - 在 `CRadixTreeIndex` 类中新增 `int protected_threshold` 成员变量，构造函数新增 `protected_threshold` 参数（默认值 2），并在构造函数体中初始化
   - 新增 `get_protected_threshold()` 访问方法供 `CRadixNode` 使用
   - _需求：1.1、1.2、1.3_

- [ ] 2. C++ 层：实现 SLRU 的比较与优先级逻辑
   - 在 `csrc/radix_tree.cpp` 的 `CRadixNode::Compare::operator()` 中新增 `EvictionPolicy::SLRU` 分支：先比较段（`hit_count >= protected_threshold` 为 Protected 段），Protected 段优先级更高（不易被淘汰）；同段内按 `last_access_time` 升序排列（越小越先淘汰）
   - 在 `CRadixNode::get_priority()` 中新增 `EvictionPolicy::SLRU` 分支，返回能反映分段优先级的值（例如 `is_protected * LARGE_VALUE + last_access_time`）
   - _需求：1.4、1.5_

- [ ] 3. Python 层：实现 SLRU 淘汰优先级逻辑
   - 在 `flexkv/cache/radixtree.py` 的 `RadixTreeIndex.__init__()` 中新增 `protected_threshold` 参数（默认值 2）并存储为实例属性
   - 在 `RadixTreeIndex._get_eviction_priority()` 中新增 `"slru"` 分支，返回 `(is_protected, last_access_time)` 元组，其中 `is_protected = 1 if node.hit_count >= self.protected_threshold else 0`
   - 更新 `_get_eviction_priority()` 末尾的错误提示，将 `"slru"` 加入支持的策略列表
   - _需求：2.1、2.2、2.3_

- [ ] 4. 配置层：新增 SLRU 相关配置项
   - 在 `flexkv/common/config.py` 的 `GLOBAL_CONFIG_FROM_ENV` 中新增 `slru_protected_threshold=int(os.getenv('FLEXKV_SLRU_PROTECTED_THRESHOLD', 2))` 配置项
   - _需求：3.2、3.3、3.4、3.6_

- [ ] 5. CacheEngine 层：注册 SLRU 策略并传递 `protected_threshold`
   - 在 `flexkv/cache/cache_engine.py` 的 `_VALID_EVICTION_POLICIES` 集合中添加 `'slru'`
   - 在 `CacheEngineAccel.__init__()` 中，当 `eviction_policy == "slru"` 时，从 `GLOBAL_CONFIG_FROM_ENV` 获取 `slru_protected_threshold` 并传递给 `CRadixTreeIndex` 构造函数
   - 在 `CacheEngine.__init__()` 中，当 `eviction_policy == "slru"` 时，从 `GLOBAL_CONFIG_FROM_ENV` 获取 `slru_protected_threshold` 并传递给 `RadixTreeIndex` 构造函数
   - _需求：3.1、3.5_

- [ ] 6. pybind11 绑定层：适配 `protected_threshold` 参数
   - 在 `csrc/bindings.cpp` 中修改 `CRadixTreeIndex` 的 Python 绑定 lambda，新增 `protected_threshold` 参数（默认值 2），并传递给 C++ 构造函数
   - 确保 `LocalRadixTree` 的绑定也能正确传递 `protected_threshold`（如果其构造函数继承了该参数）
   - _需求：4.1、4.2、4.3_

- [ ] 7. 文档更新：中英文淘汰策略文档
   - 在 `docs/eviction_policy/README_zh.md` 的"支持的驱逐策略"表格中新增 SLRU 行：`slru` | 分段最近最少使用 | 将节点分为试用段和保护段，优先驱逐试用段中最久未访问的节点
   - 在"驱逐相关配置项"表格中新增 `slru_protected_threshold` 行
   - 在"各策略详解"章节中新增 SLRU 详解小节，说明 Probationary/Protected 分段机制、优先级计算方式和适用场景
   - 在 `docs/eviction_policy/README_en.md` 中做对应的英文更新
   - 更新所有涉及可选值列表的文本（如环境变量说明中的 `lru, lfu, fifo, mru, filo` → `lru, lfu, slru, fifo, mru, filo`）
   - _需求：5.1、5.2、5.3、5.4_

- [ ] 8. 单元测试：SLRU 策略核心行为验证
   - 在 `tests/test_cache_engine.py` 中新增 SLRU 相关测试用例（同时覆盖 `CacheEngine` 和 `CacheEngineAccel`）：
     - 测试 SLRU 策略初始化：验证 `eviction_policy='slru'` 能正常创建引擎
     - 测试 Protected 节点保留：插入多个节点，对部分节点多次访问使其 `hit_count >= protected_threshold`，触发淘汰后验证高频节点被保留、低频节点被淘汰
     - 测试同段内 LRU 排序：在同一段（Probationary 或 Protected）内，验证 `last_access_time` 更早的节点先被淘汰
     - 测试 `protected_threshold` 自定义值：验证设置不同的 `protected_threshold` 值时行为正确
   - 确保现有的 LRU、LFU 等策略测试用例不受影响（回归测试）
   - _需求：6.1、6.2、6.3、6.4、6.5_
