# 实施计划：SLRU 优先级比较重构 — 消除魔法值

## 背景

当前代码中存在两套优先级机制：
- `Compare::operator()` — 使用 `switch-case` + 多字段比较，SLRU 分支已正确实现，**无魔法值**
- `get_priority()` — 返回 `double` 标量，SLRU 分支使用 `1e15` 魔法数字，且在 C++ 层**无任何调用方**

重构策略：引入策略多态体系，将各淘汰策略的比较逻辑封装为独立的策略类，统一 `Compare` 和 `get_priority` 的实现路径。

---

- [ ] 1. 定义淘汰策略基类接口 `IEvictionStrategy`
   - 在 `radix_tree.h` 中新增抽象基类 `IEvictionStrategy`
   - 定义纯虚方法 `bool compare(CRadixNode *a, CRadixNode *b) const` — 用于 priority_queue 比较，返回 true 表示 a 的优先级高于 b（a 更不应该被淘汰）
   - 定义纯虚方法 `std::string priority_repr(CRadixNode *node) const` — 返回人类可读的优先级表示（替代原 `get_priority()` 的调试用途），避免将多维优先级压缩为 double
   - _需求：2.3、1.1_

- [ ] 2. 实现各具体策略类
   - 在 `radix_tree.h`（或新建 `eviction_strategy.h`）中实现以下策略类，每个类继承 `IEvictionStrategy`：
     - `LRUStrategy`：compare 比较 `grace_time`；priority_repr 返回 grace_time 字符串
     - `LFUStrategy`：compare 先比 `hit_count`，再比 `last_access_time`；priority_repr 返回 `(hit_count, last_access_time)` 字符串
     - `FIFOStrategy`：compare 比较 `creation_time`；priority_repr 返回 creation_time 字符串
     - `MRUStrategy`：compare 反向比较 `last_access_time`；priority_repr 返回 `-last_access_time` 字符串
     - `FILOStrategy`：compare 反向比较 `creation_time`；priority_repr 返回 `-creation_time` 字符串
     - `SLRUStrategy`：接收 `protected_threshold` 参数，compare 先比 segment（hit_count >= threshold 为 Protected），同 segment 内比 `last_access_time`；priority_repr 返回 `(segment, last_access_time)` 字符串
   - 每个策略类的 compare 逻辑直接从当前 `Compare::operator()` 的对应 case 分支迁移，确保行为完全一致
   - _需求：2.1、2.3、4.1、4.2、4.3_

- [ ] 3. 添加策略工厂函数
   - 实现 `std::unique_ptr<IEvictionStrategy> create_eviction_strategy(EvictionPolicy policy, int protected_threshold = 2)` 工厂函数
   - 根据 `EvictionPolicy` 枚举值创建对应的策略实例
   - SLRU 策略需要传入 `protected_threshold` 参数
   - _需求：2.1_

- [ ] 4. 在 `CRadixTreeIndex` 中持有策略实例
   - 在 `CRadixTreeIndex` 构造函数中调用工厂函数创建策略实例，存储为 `std::unique_ptr<IEvictionStrategy> strategy_`
   - 新增 `const IEvictionStrategy* get_strategy() const` 方法供 `CRadixNode` 访问
   - 保留现有的 `get_eviction_policy()` 和 `get_protected_threshold()` 公共 API 不变
   - _需求：3.3、3.1_

- [ ] 5. 重构 `Compare::operator()` 委托给策略对象
   - 将 `CRadixNode::Compare::operator()` 中的 `switch-case` 替换为对策略对象的委托调用：`return a->get_index()->get_strategy()->compare(a, b)`
   - 删除原有的 `switch-case` 分支代码
   - 确保 `priority_queue` 的行为与重构前完全一致
   - _需求：3.1、2.2_

- [ ] 6. 重构 `get_priority()` 为 `get_priority_repr()`
   - 将 `CRadixNode::get_priority()` 重命名为 `get_priority_repr()`，返回类型从 `double` 改为 `std::string`
   - 实现委托给策略对象：`return index->get_strategy()->priority_repr(this)`
   - 删除原有的 `switch-case` 分支代码和 `1e15` 魔法数字
   - 由于 `get_priority()` 在 C++ 层无任何调用方且未暴露给 Python，此变更不影响任何现有功能
   - 同步更新 `radix_tree.h` 中的声明
   - _需求：1.1、1.2、3.4_

- [ ] 7. 验证现有单元测试通过
   - 编译项目，确保无编译错误
   - 运行现有的所有单元测试，确保全部通过
   - 重点验证 SLRU 相关的驱逐行为测试（如果存在）
   - _需求：3.2_
