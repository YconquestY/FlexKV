# 稀疏注意力 Indexer Cache 支持

## 概述

FlexKV 现已支持 **原生稀疏注意力（NSA, Native Sparse Attention）** indexer cache 的卸载，适用于 **DeepSeek V3.2** 等模型。该功能使 FlexKV 能够在主 MLA KV cache 之外，同时管理和搬运稀疏注意力 indexer 的 `k_cache`，确保 GPU、CPU、SSD 之间的 block 搬运操作是原子性的。

## 背景

### 什么是原生稀疏注意力（NSA）？

DeepSeek V3.2 引入了 NSA，这是一种硬件对齐的稀疏注意力机制，通过仅选择与每个 query 最相关的 KV cache chunk 来降低注意力计算开销。NSA indexer 维护一个紧凑的 `k_cache`，用于存储压缩后的 key 表示，以确定哪些 chunk 是相关的。

### 两种 KV Cache

在启用了 NSA 的 DeepSeek V3.2 模型中，每个注意力层管理两种不同的 cache tensor：

| Cache 类型 | 用途 |
|---|---|
| **主 MLA KV Cache** | 标准注意力的压缩 KV，使用模型默认 dtype（如 bfloat16） |
| **Indexer k_cache** | 用于稀疏 chunk 选择的压缩 key，使用 uint8 dtype |

两种 cache 在 vLLM 中共享相同的 block table 和 slot mapping，FlexKV 的 radix tree 匹配和 block 管理逻辑无需修改。

## 设计原则

### 1. 向后兼容
所有 indexer 相关字段均为 **可选**，默认值为 `None`。不使用稀疏注意力的模型（如 Qwen、LLaMA）完全不受影响。

### 2. 共享 Block Table
Indexer cache 与主 KV cache 共享 vLLM 的 block table 和 `slot_mapping`，block ID 一一对应。FlexKV 的 radix tree match/put 逻辑 **无需任何修改**。

### 3. Shadow Worker 模式
Indexer 传输 worker 作为主 worker 的"影子"运行。当一个 block 被搬运（GPU ↔ CPU）时，相同 block ID 的主 KV cache 数据和 indexer cache 数据会被并行搬运。

### 4. 自动 CPU 分配
注册 GPU indexer cache blocks 时，如果启用了 CPU 卸载，会自动分配对应的 CPU 端 buffer，确保 GPU ↔ CPU 双向卸载能力。

### 5. 最小化开销
Indexer cache 远小于主 KV cache，因此 indexer 传输总是先于主传输完成，不会成为瓶颈。

## 运行测试

```bash
# 运行稀疏注意力 indexer cache 单测（CUDA 测试会自动跳过，如无 GPU）
pytest tests/test_sparse_attention_indexer.py -v

# 运行全部测试
bash run_tests.sh
```

## 配置

无需额外配置。当 vLLM 注册的 `kv_caches_dict` 中包含 `.k_cache` layer name 时，FlexKV 会自动检测并处理 indexer cache 层。现有的 FlexKV 配置（CPU cache 大小、SSD 设置等）同时适用于主 cache 和 indexer cache。

```bash
# 标准配置即可用于稀疏注意力模型
export FLEXKV_CPU_CACHE_GB=32
```
