# 需求文档：SLRU 优先级比较重构 — 消除魔法值

## 引言

FlexKV 的 C++ 层（`radix_tree.h` / `radix_tree.cpp`）实现了多种缓存淘汰策略（LRU、LFU、SLRU、FIFO、MRU、FILO）。当前存在两套优先级机制：

1. **`CRadixNode::Compare::operator()`** — 用于 `std::priority_queue` 的比较函数，是驱逐决策的**唯一真实来源**。SLRU 分支已正确实现多字段比较（先比 segment，再比 time），**无魔法值**。
2. **`CRadixNode::get_priority()`** — 返回 `double` 标量，用于展示/调试。SLRU 分支使用了 `1e15` 魔法数字来将多维优先级压缩为单一标量，这是不可接受的。

此外，Python 层（sglang 参考实现和 FlexKV 的 `radixtree.py`）的 `get_priority()` 返回 `Tuple[int, float]`，天然支持多字段比较，不存在此问题。

**核心问题**：C++ 的 `get_priority()` 返回 `double`，无法自然表达 SLRU 等多维优先级策略，导致使用魔法数字。用户明确要求：
- **禁止使用魔法值**（如 `1e15`、`1e18`）
- **禁止使用位移技巧**（如 `<< 62`）
- **要求使用自定义比较函数或多态**来实现

## 现状分析

### 当前代码结构

```
CRadixNode
├── Compare::operator()   ← 驱逐的唯一真实来源（priority_queue 比较器）
│   └── switch(policy) { case SLRU: 先比 segment，再比 time }  ✅ 已正确
├── get_priority() → double  ← 展示/调试用途
│   └── switch(policy) { case SLRU: 1e15 * segment + time }  ❌ 魔法值
```

### 关键约束

- `get_priority()` 当前返回 `double`，在 C++ 层**未被任何代码调用**，也未通过 pybind11 暴露给 Python
- `Compare::operator()` 是驱逐逻辑的唯一消费者，已正确实现
- Python 层有独立的 `_get_eviction_priority()` 实现，返回 tuple，不依赖 C++ 的 `get_priority()`

## 需求

### 需求 1：消除 `get_priority()` 中的魔法值

**用户故事：** 作为一名 FlexKV 开发者，我希望 `get_priority()` 不使用任何魔法数字或位移技巧，以便代码可读性和可维护性得到保障。

#### 验收标准

1. WHEN SLRU 策略的 `get_priority()` 被调用 THEN 系统 SHALL 不使用任何硬编码的大数字（如 `1e15`、`1e18`）或位移操作（如 `<< 62`）来区分 segment
2. WHEN 任何策略的 `get_priority()` 被调用 THEN 系统 SHALL 返回的值能正确反映节点的淘汰优先级顺序（值越小越先被淘汰）

### 需求 2：采用自定义比较函数或多态设计

**用户故事：** 作为一名 FlexKV 开发者，我希望优先级比较逻辑采用面向对象的设计（自定义比较函数或多态），以便不同策略的比较逻辑清晰解耦、易于扩展。

#### 验收标准

1. WHEN 新增一种淘汰策略 THEN 开发者 SHALL 能够通过实现一个明确的接口/基类来定义该策略的比较逻辑，而无需修改 `switch-case` 中的魔法数字
2. WHEN 多维优先级策略（如 SLRU、LFU）需要比较两个节点 THEN 系统 SHALL 通过结构化的多字段比较来实现，而非将多维信息压缩为单一标量
3. IF 采用多态方案 THEN 系统 SHALL 定义一个策略基类/接口，每种淘汰策略实现自己的比较方法
4. IF 采用自定义比较函数方案 THEN 系统 SHALL 为每种策略提供独立的比较函数对象（functor 或 lambda），通过配置注入到比较逻辑中

### 需求 3：保持与现有架构的兼容性

**用户故事：** 作为一名 FlexKV 开发者，我希望重构不破坏现有的驱逐逻辑和 API 接口，以便现有功能和测试继续正常工作。

#### 验收标准

1. WHEN 重构完成后 THEN `Compare::operator()` 的行为 SHALL 与重构前完全一致（所有策略的淘汰顺序不变）
2. WHEN 重构完成后 THEN 现有的所有单元测试 SHALL 继续通过
3. WHEN 重构完成后 THEN `CRadixTreeIndex` 的公共 API（构造函数参数、`get_eviction_policy()`、`get_protected_threshold()`）SHALL 保持不变
4. IF `get_priority()` 的返回类型发生变化 THEN 系统 SHALL 确保不影响任何现有调用方（当前 C++ 层无调用方，Python 层有独立实现）

### 需求 4：与 sglang 参考实现保持语义一致

**用户故事：** 作为一名 FlexKV 开发者，我希望 C++ 层的 SLRU 实现与 sglang 的 Python 参考实现保持语义一致，以便两者的淘汰行为可预测且可对比。

#### 验收标准

1. WHEN SLRU 策略比较两个节点 THEN 系统 SHALL 遵循与 sglang `SLRUStrategy.get_priority()` 相同的语义：先按 segment（Probationary=0, Protected=1）排序，同 segment 内按 `last_access_time` 排序
2. WHEN 节点的 `hit_count >= protected_threshold` THEN 该节点 SHALL 被归入 Protected segment（segment=1）
3. WHEN 节点的 `hit_count < protected_threshold` THEN 该节点 SHALL 被归入 Probationary segment（segment=0）

## 技术约束与边界条件

1. **性能要求**：`Compare::operator()` 在驱逐热路径上被频繁调用，重构后不应引入显著的性能开销（如虚函数调用的间接开销需评估）
2. **编译兼容性**：代码需兼容 C++17 标准
3. **`get_priority()` 的定位**：如果 `get_priority()` 当前无调用方，可以考虑：(a) 改变其返回类型以支持多维比较；(b) 将其重构为与 `Compare` 统一的机制；(c) 如确认无用则移除
4. **`switch-case` vs 多态**：当前 `Compare::operator()` 使用 `switch-case`，重构可以选择保留 `switch-case`（但消除魔法值）或改为多态分发，需权衡代码复杂度与可扩展性
