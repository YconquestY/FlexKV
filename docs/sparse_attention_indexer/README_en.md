# Sparse Attention Indexer Cache Support

## Overview

FlexKV now supports **Native Sparse Attention (NSA)** indexer cache offloading for models like **DeepSeek V3.2**. This feature enables FlexKV to manage and transfer the sparse attention indexer's `k_cache` alongside the main MLA KV cache, ensuring that both caches are offloaded and restored atomically when blocks are transferred between GPU, CPU, and SSD.

## Background

### What is Native Sparse Attention (NSA)?

DeepSeek V3.2 introduces NSA, a hardware-aligned sparse attention mechanism that reduces attention computation cost by selecting only the most relevant KV cache chunks for each query. The NSA indexer maintains a compact `k_cache` that stores compressed key representations used to determine which chunks are relevant.

### The Two Types of KV Cache

In a DeepSeek V3.2 model with NSA enabled, each attention layer manages two distinct cache tensors:

| Cache Type | Purpose |
|---|---|
| **Main MLA KV Cache** | Compressed KV for standard attention, using the model's default dtype (e.g., bfloat16) |
| **Indexer k_cache** | Compressed keys for sparse chunk selection, using uint8 dtype |

Both caches share the same block table and slot mapping in vLLM. FlexKV's radix tree matching and block management logic works unchanged.

## Design Principles

### 1. Backward Compatibility
All indexer-related fields are **optional** with `None` defaults. Models without sparse attention (e.g., Qwen, LLaMA) are completely unaffected.

### 2. Shared Block Table
The indexer cache shares vLLM's block table and `slot_mapping` with the main KV cache. Block IDs have a 1:1 correspondence. FlexKV's radix tree match/put logic requires **zero modifications**.

### 3. Shadow Worker Pattern
Indexer transfer workers operate as "shadows" of the main workers. When a block is transferred (GPU ↔ CPU), both the main KV cache data and the indexer cache data for the same block IDs are transferred in parallel.

### 4. Automatic CPU Allocation
When GPU indexer cache blocks are registered, the corresponding CPU buffer is automatically allocated if CPU offloading is enabled, ensuring bidirectional GPU ↔ CPU offloading capability.

### 5. Minimal Overhead
The indexer cache is significantly smaller than the main KV cache, so indexer transfers complete faster than main transfers and do not become a bottleneck.

## Running Tests

```bash
# Run sparse attention indexer cache tests (CUDA tests auto-skip if no GPU)
pytest tests/test_sparse_attention_indexer.py -v

# Run all tests
bash run_tests.sh
```

## Configuration

No additional configuration is needed. FlexKV automatically detects indexer cache layers when vLLM registers `kv_caches_dict` containing `.k_cache` layer names. The existing FlexKV configuration (CPU cache size, SSD settings, etc.) applies to both main and indexer caches.

```bash
# Standard configuration works for sparse attention models
export FLEXKV_CPU_CACHE_GB=32
```
