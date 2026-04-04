# 需求文档：C++ 层 SLRU 策略简洁实现

## 引言

本需求描述在 FlexKV 的 C++ 层（`radix_tree.h` / `radix_tree.cpp`）中，以**简洁直观的方式**实现 SLRU（Segmented LRU）缓存淘汰策略。

### 背景

FlexKV 的 Python 层已经完成了 SLRU 策略的实现（`radixtree.py` 中的 `_get_eviction_priority()`），配置层（`cache_engine.py`、`config.py`）和绑定层（`bindings.cpp`）也已适配完毕，但 C++ 核心层（`radix_tree.h` / `radix_tree.cpp`）尚未实现 SLRU。

### 设计原则

**采用与现有 LFU 策略完全一致的实现风格**：直接在 `Compare::operator()` 的 `switch-case` 中新增 `case EvictionPolicy::SLRU` 分支，使用简单的 if-else 逻辑实现分段比较。不引入任何新的结构体（如 `EvictionKey`）、位编码、魔法数字或多态设计。

### 参考：现有 LFU 的实现模式

```cpp
case EvictionPolicy::LFU:
  if (a->get_hit_count() != b->get_hit_count()) {
    return a->get_hit_count() > b->get_hit_count();
  }
  return a->get_last_access_time() > b->get_last_access_time();
```

LFU 先按 `hit_count` 比较，相同时再按 `last_access_time` 比较。SLRU 应采用完全相同的模式：先按"是否在保护段"比较，相同段内再按 `last_access_time` 比较。

### SLRU 算法逻辑

- **Probationary 段**：`hit_count < protected_threshold`（默认 2）
- **Protected 段**：`hit_count >= protected_threshold`
- **淘汰优先级**：Probationary 段节点优先被淘汰；同段内按 `last_access_time` 升序淘汰（越早访问的越先淘汰）
- 这与 SGLang 的 `SLRUStrategy` 行为完全一致

## 需求

### 需求 1：在 EvictionPolicy 枚举中新增 SLRU

**用户故事：** 作为一名 FlexKV 开发者，我希望 C++ 层的 `EvictionPolicy` 枚举包含 `SLRU` 值，以便 C++ 代码能识别 SLRU 策略。

#### 验收标准

1. WHEN `EvictionPolicy` 枚举被定义时 THEN 枚举 SHALL 包含 `SLRU` 值
2. WHEN `parse_eviction_policy()` 函数接收字符串 `"slru"` THEN 函数 SHALL 返回 `EvictionPolicy::SLRU`
3. WHEN `parse_eviction_policy()` 的错误提示被展示时 THEN 提示信息 SHALL 包含 `'slru'`

### 需求 2：在 CRadixTreeIndex 中存储 protected_threshold

**用户故事：** 作为一名 FlexKV 开发者，我希望 `CRadixTreeIndex` 能存储 `protected_threshold` 参数，以便 SLRU 策略在比较节点时能获取该阈值。

#### 验收标准

1. WHEN `CRadixTreeIndex` 构造函数被调用时 THEN 构造函数 SHALL 接受 `protected_threshold` 参数（默认值为 2）
2. WHEN `CRadixNode` 需要获取 `protected_threshold` 时 THEN 节点 SHALL 能通过 `get_index()->get_protected_threshold()` 获取该值
3. IF `eviction_policy` 不是 SLRU THEN `protected_threshold` 参数 SHALL 被存储但不影响其他策略的行为

### 需求 3：在 Compare::operator() 中实现 SLRU 比较逻辑

**用户故事：** 作为一名 FlexKV 开发者，我希望 `CRadixNode::Compare::operator()` 中有一个 SLRU 分支，采用与 LFU 相同的 if-else 风格实现分段比较，以便淘汰时能正确区分 Probationary 和 Protected 段。

#### 验收标准

1. WHEN `Compare::operator()` 在 SLRU 策略下比较两个节点时 THEN 系统 SHALL 先判断两个节点是否在同一段（通过 `hit_count` 与 `protected_threshold` 的比较）
2. IF 两个节点不在同一段 THEN 系统 SHALL 让 Probationary 段的节点排在前面（优先被淘汰），即 `is_protected_a < is_protected_b` 时返回 `true`
3. IF 两个节点在同一段 THEN 系统 SHALL 按 `last_access_time` 升序排列（越早访问的越先淘汰），即 `a->get_last_access_time() > b->get_last_access_time()` 时返回 `true`（因为是最大堆，堆顶先弹出）
4. WHEN 实现代码被审查时 THEN 代码风格 SHALL 与现有 LFU 分支保持一致，不引入额外的结构体、位编码或魔法数字

### 需求 4：在 get_priority() 中实现 SLRU 优先级

**用户故事：** 作为一名 FlexKV 开发者，我希望 `CRadixNode::get_priority()` 在 SLRU 策略下返回合理的标量值，以便用于日志、调试等展示场景。

#### 验收标准

1. WHEN `get_priority()` 在 SLRU 策略下被调用时 THEN 系统 SHALL 返回一个 `double` 值，能大致反映节点的淘汰优先级
2. WHEN 返回值被使用时 THEN Protected 段节点的返回值 SHALL 大于 Probationary 段节点的返回值（表示更不容易被淘汰）
3. WHEN 实现方式被审查时 THEN 实现 SHALL 简洁明了（注意：`get_priority()` 仅用于展示，真正的淘汰排序由 `Compare::operator()` 决定）

### 需求 5：LocalRadixTree 构造函数适配

**用户故事：** 作为一名 FlexKV 开发者，我希望 `LocalRadixTree` 构造函数能将 `protected_threshold` 参数传递给基类 `CRadixTreeIndex`，以便分布式场景下也能使用 SLRU 策略。

#### 验收标准

1. WHEN `LocalRadixTree` 构造函数被调用时 THEN 构造函数 SHALL 接受 `protected_threshold` 参数并传递给基类
2. WHEN `LocalRadixTree` 的 pybind11 绑定已经支持 `protected_threshold` 参数时 THEN 该参数 SHALL 被正确传递到 C++ 层

### 需求 6：单元测试验证

**用户故事：** 作为一名 FlexKV 开发者，我希望有单元测试验证 SLRU 策略在 C++ 层的正确性，以便确保实现与 Python 层行为一致。

#### 验收标准

1. WHEN 运行 SLRU 单元测试时 THEN 测试 SHALL 验证 Protected 段节点（`hit_count >= protected_threshold`）在淘汰时被保留
2. WHEN 运行 SLRU 单元测试时 THEN 测试 SHALL 验证 Probationary 段节点优先被淘汰
3. WHEN 运行 SLRU 单元测试时 THEN 测试 SHALL 验证同一段内按 `last_access_time` 排序（LRU 规则）
4. WHEN 运行 SLRU 单元测试时 THEN 测试 SHALL 覆盖 C++ 索引（`CRadixTreeIndex`）的实现
