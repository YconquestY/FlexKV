# 实施计划

- [ ] 1. 在 `radix_tree.h` 中扩展 `EvictionPolicy` 枚举和 `CRadixTreeIndex` 类
  - 在 `EvictionPolicy` 枚举中新增 `SLRU` 值
  - 在 `parse_eviction_policy()` 函数中新增 `"slru"` 的解析分支，并更新错误提示信息包含 `'slru'`
  - 在 `CRadixTreeIndex` 类中新增 `int protected_threshold` 成员变量
  - 在 `CRadixTreeIndex` 构造函数中新增 `int protected_threshold = 2` 参数，并在初始化列表中赋值
  - 新增 `int get_protected_threshold() { return protected_threshold; }` 公有方法
  - 文件：`csrc/radix_tree.h`
  - _需求：1.1, 1.2, 1.3, 2.1, 2.2, 2.3_

- [ ] 2. 在 `radix_tree.cpp` 的 `Compare::operator()` 中新增 SLRU 分支
  - 在 `switch (policy)` 中新增 `case EvictionPolicy::SLRU` 分支
  - 实现逻辑：先通过 `hit_count >= protected_threshold` 判断两个节点是否在同一段
  - 不同段时：Probationary 段（`is_protected` 为 false）的节点优先被淘汰
  - 同段时：按 `last_access_time` 升序淘汰（与 LRU 一致）
  - 代码风格与现有 LFU 分支完全一致，使用简单的 if-else，不引入额外结构体
  - 参考实现：
    ```cpp
    case EvictionPolicy::SLRU: {
      int threshold = a->get_index()->get_protected_threshold();
      bool a_protected = a->get_hit_count() >= threshold;
      bool b_protected = b->get_hit_count() >= threshold;
      if (a_protected != b_protected) {
        return a_protected < b_protected;  // Probationary first
      }
      return a->get_last_access_time() > b->get_last_access_time();  // Same segment: LRU
    }
    ```
  - 文件：`csrc/radix_tree.cpp`
  - _需求：3.1, 3.2, 3.3, 3.4_

- [ ] 3. 在 `radix_tree.cpp` 的 `get_priority()` 中新增 SLRU 分支
  - 在 `switch (policy)` 中新增 `case EvictionPolicy::SLRU` 分支
  - 返回 `(double)last_access_time`，Protected 段节点可额外加一个大偏移量以区分段
  - 注意：`get_priority()` 仅用于展示/调试，真正的淘汰排序由 `Compare::operator()` 决定，因此实现可以简单
  - 文件：`csrc/radix_tree.cpp`
  - _需求：4.1, 4.2, 4.3_

- [ ] 4. 适配 `LocalRadixTree` 构造函数传递 `protected_threshold`
  - 在 `local_radix_tree.h` 的 `LocalRadixTree` 构造函数声明中新增 `int protected_threshold = 2` 参数
  - 在 `local_radix_tree.cpp` 的构造函数实现中接收该参数，并传递给基类 `CRadixTreeIndex` 的构造函数
  - 文件：`csrc/dist/local_radix_tree.h`、`csrc/dist/local_radix_tree.cpp`
  - _需求：5.1, 5.2_

- [ ] 5. 验证现有单元测试覆盖 SLRU C++ 索引
  - 确认 `tests/test_cache_engine.py` 中已有的 SLRU 测试用例能覆盖 C++ 索引（`CRadixTreeIndex`）
  - 已有测试包括：`test_slru_protected_node_retained`、`test_slru_same_segment_lru_order`、`test_slru_custom_protected_threshold`、`test_slru_batch_eviction_cross_segment`
  - 如果现有测试的 `engine_cls` 参数化已包含 C++ 索引路径，则无需新增测试；否则需补充
  - 文件：`tests/test_cache_engine.py`
  - _需求：6.1, 6.2, 6.3, 6.4_
