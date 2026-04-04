# 需求文档：FlexKV 新增 SLRU（Segmented LRU）缓存淘汰策略

## 引言

本需求描述将 SGLang 中的 SLRU（Segmented Least Recently Used）缓存淘汰策略迁移到 FlexKV 项目中。SLRU 是经典 LRU 的改进版本，通过将缓存逻辑上分为 **Probationary（试用段）** 和 **Protected（保护段）** 两个段，结合了 LRU 的时间敏感性和 LFU 的频率感知能力，能有效抵抗缓存扫描污染，适用于混合工作负载场景。

FlexKV 目前支持 5 种淘汰策略（LRU、LFU、FIFO、MRU、FILO），本次需求在此基础上新增 SLRU 策略。FlexKV 的缓存索引有 Python（`RadixTreeIndex`）和 C++（`CRadixTreeIndex`）两套实现，两套实现均需支持 SLRU。

### 参考实现

SGLang 中 SLRU 的核心逻辑（`sglang/srt/mem_cache/evict_policy.py`）：
- 节点 `hit_count < protected_threshold`（默认 2）→ Probationary 段（segment=0）
- 节点 `hit_count >= protected_threshold` → Protected 段（segment=1）
- 淘汰优先级：`(segment, last_access_time)`，元组字典序比较，segment 小的先淘汰，同段内按 LRU

## 需求

### 需求 1：C++ 层 SLRU 策略实现

**用户故事：** 作为一名 FlexKV 开发者，我希望在 C++ 层的 Radix Tree 索引中新增 SLRU 淘汰策略，以便使用 C++ 加速索引（`index_accel=True`）时也能使用 SLRU 策略。

#### 验收标准

1. WHEN `EvictionPolicy` 枚举被定义时 THEN 系统 SHALL 包含 `SLRU` 枚举值
2. WHEN `parse_eviction_policy` 函数接收到字符串 `"slru"` THEN 系统 SHALL 返回 `EvictionPolicy::SLRU`
3. WHEN `CRadixTreeIndex` 使用 SLRU 策略构造时 THEN 系统 SHALL 接受并存储 `protected_threshold` 参数（默认值为 2）
4. WHEN `CRadixNode::Compare::operator()` 在 SLRU 策略下比较两个节点时 THEN 系统 SHALL 先按段（`hit_count >= protected_threshold` 为 Protected 段，否则为 Probationary 段）排序，同段内按 `last_access_time` 排序（越小越先淘汰）
5. WHEN `CRadixNode::get_priority()` 在 SLRU 策略下被调用时 THEN 系统 SHALL 返回能正确反映分段优先级的值

### 需求 2：Python 层 SLRU 策略实现

**用户故事：** 作为一名 FlexKV 开发者，我希望在 Python 层的 Radix Tree 索引中新增 SLRU 淘汰策略，以便使用纯 Python 索引（`index_accel=False`）时也能使用 SLRU 策略。

#### 验收标准

1. WHEN `RadixTreeIndex._get_eviction_priority()` 在 SLRU 策略下被调用时 THEN 系统 SHALL 返回 `(is_protected, last_access_time)` 元组，其中 `is_protected` 为 1（`hit_count >= protected_threshold`）或 0
2. WHEN `RadixTreeIndex` 使用 `"slru"` 策略构造时 THEN 系统 SHALL 接受并存储 `protected_threshold` 参数（默认值为 2）
3. WHEN 节点的 `hit_count` 从低于 `protected_threshold` 增长到达到或超过 `protected_threshold` THEN 该节点 SHALL 在下次淘汰时被视为 Protected 段节点，优先级高于 Probationary 段节点

### 需求 3：配置与参数传递

**用户故事：** 作为一名 FlexKV 用户，我希望通过配置文件或环境变量来启用 SLRU 策略并设置保护阈值，以便灵活控制缓存淘汰行为。

#### 验收标准

1. WHEN 用户设置 `eviction_policy` 为 `"slru"` THEN 系统 SHALL 在 `CacheEngine` 和 `CacheEngineAccel` 中使用 SLRU 策略
2. WHEN 用户设置环境变量 `FLEXKV_EVICTION_POLICY=slru` THEN 系统 SHALL 使用 SLRU 策略
3. WHEN 用户设置环境变量 `FLEXKV_SLRU_PROTECTED_THRESHOLD=N`（N 为正整数）THEN 系统 SHALL 使用 N 作为 SLRU 的保护阈值
4. IF 用户未设置 `FLEXKV_SLRU_PROTECTED_THRESHOLD` THEN 系统 SHALL 使用默认值 2
5. WHEN `_VALID_EVICTION_POLICIES` 集合被引用时 THEN 系统 SHALL 包含 `"slru"`
6. WHEN 用户在配置文件中设置 `slru_protected_threshold` THEN 系统 SHALL 将该值传递到 C++ 和 Python 索引层

### 需求 4：pybind11 绑定层适配

**用户故事：** 作为一名 FlexKV 开发者，我希望 C++ 层的 SLRU 策略能通过 pybind11 正确暴露给 Python 层，以便 `CacheEngineAccel` 能正确使用 C++ 加速的 SLRU 策略。

#### 验收标准

1. WHEN `CRadixTreeIndex` 的 Python 绑定被调用时 THEN 系统 SHALL 支持传入 `protected_threshold` 参数
2. WHEN `eviction_policy` 参数为 `"slru"` 且 `protected_threshold` 被指定时 THEN `CRadixTreeIndex` 构造函数 SHALL 正确接收并使用该阈值
3. IF `eviction_policy` 不是 `"slru"` THEN `protected_threshold` 参数 SHALL 被忽略，不影响其他策略的行为

### 需求 5：文档更新

**用户故事：** 作为一名 FlexKV 用户，我希望在文档中看到 SLRU 策略的说明和配置方式，以便了解如何使用该策略。

#### 验收标准

1. WHEN 用户查阅 `docs/eviction_policy/README_zh.md` THEN 文档 SHALL 包含 SLRU 策略的说明、行为描述、配置方式和适用场景
2. WHEN 用户查阅 `docs/eviction_policy/README_en.md` THEN 文档 SHALL 包含对应的英文版 SLRU 策略说明
3. WHEN 文档中的"支持的驱逐策略"表格被展示时 THEN 表格 SHALL 包含 SLRU 策略行
4. WHEN 文档中的"驱逐相关配置项"表格被展示时 THEN 表格 SHALL 包含 `slru_protected_threshold` 配置项

### 需求 6：单元测试

**用户故事：** 作为一名 FlexKV 开发者，我希望有完善的单元测试覆盖 SLRU 策略的核心行为，以便确保策略的正确性和回归安全。

#### 验收标准

1. WHEN 运行 SLRU 单元测试时 THEN 测试 SHALL 验证高频访问节点（`hit_count >= protected_threshold`）在淘汰时被保留
2. WHEN 运行 SLRU 单元测试时 THEN 测试 SHALL 验证低频访问节点（`hit_count < protected_threshold`）优先被淘汰
3. WHEN 运行 SLRU 单元测试时 THEN 测试 SHALL 验证同一段内按 LRU 规则（`last_access_time` 越小越先淘汰）淘汰
4. WHEN 运行 SLRU 单元测试时 THEN 测试 SHALL 同时覆盖 Python 索引（`RadixTreeIndex`）和 C++ 索引（`CRadixTreeIndex`）两套实现
5. WHEN SLRU 策略与其他策略（如 LRU、LFU）共存时 THEN 其他策略的行为 SHALL 不受影响（回归测试）
