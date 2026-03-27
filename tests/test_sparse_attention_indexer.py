"""
Unit tests for sparse attention indexer cache support in FlexKV.

Tests cover all modules modified to support DeepSeek V3.2 DSA (Dense Sparse Attention)
indexer k_cache storage alongside main MLA KV cache:

1. DeviceType enum: GPU_INDEXER and CPU_INDEXER values
2. RegisterTPClientRequest: indexer_handles, indexer_layout, indexer_dtype fields
3. KVCacheLayout: layout construction for indexer cache tensors
4. StorageEngine: register_indexer_blocks, allocate for GPU_INDEXER/CPU_INDEXER
5. vLLM v1 adapter: kv_caches dict splitting by layer name (.k_cache)
6. KVTPClient: register_to_server with indexer params
7. TransferManager: indexer registration in _handle_gpu_blocks_registration
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import fields

from flexkv.common.transfer import DeviceType
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType, StorageHandle, AccessHandleType
from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.server.request import RegisterTPClientRequest
from flexkv.storage.storage_engine import StorageEngine


# ============================================================
# Module 1: DeviceType enum - GPU_INDEXER and CPU_INDEXER
# ============================================================

class TestDeviceTypeIndexer:
    """Test that the DeviceType enum includes the new indexer device types."""

    def test_gpu_indexer_exists(self):
        assert hasattr(DeviceType, 'GPU_INDEXER')
        assert DeviceType.GPU_INDEXER == 6

    def test_cpu_indexer_exists(self):
        assert hasattr(DeviceType, 'CPU_INDEXER')
        assert DeviceType.CPU_INDEXER == 7

    def test_original_device_types_unchanged(self):
        assert DeviceType.CPU == 0
        assert DeviceType.GPU == 1
        assert DeviceType.SSD == 2
        assert DeviceType.REMOTE == 3
        assert DeviceType.PEERCPU == 4
        assert DeviceType.PEERSSD == 5

    def test_indexer_types_are_distinct(self):
        assert DeviceType.GPU_INDEXER != DeviceType.CPU_INDEXER
        assert DeviceType.GPU_INDEXER != DeviceType.GPU
        assert DeviceType.CPU_INDEXER != DeviceType.CPU


# ============================================================
# Module 2: RegisterTPClientRequest - indexer fields
# ============================================================

class TestRegisterTPClientRequest:
    """Test RegisterTPClientRequest with optional indexer fields."""

    def test_default_indexer_fields_are_none(self):
        mock_handles = [MagicMock(spec=TensorSharedHandle)]
        mock_layout = MagicMock(spec=KVCacheLayout)
        req = RegisterTPClientRequest(
            dp_client_id=0, device_id=0,
            handles=mock_handles, gpu_layout=mock_layout,
        )
        assert req.indexer_handles is None
        assert req.indexer_layout is None
        assert req.indexer_dtype is None

    def test_with_indexer_fields(self):
        mock_handles = [MagicMock(spec=TensorSharedHandle)]
        mock_layout = MagicMock(spec=KVCacheLayout)
        mock_indexer_handles = [MagicMock(spec=TensorSharedHandle)]
        mock_indexer_layout = MagicMock(spec=KVCacheLayout)

        req = RegisterTPClientRequest(
            dp_client_id=0, device_id=0,
            handles=mock_handles, gpu_layout=mock_layout,
            indexer_handles=mock_indexer_handles,
            indexer_layout=mock_indexer_layout,
            indexer_dtype=torch.uint8,
        )
        assert req.indexer_handles == mock_indexer_handles
        assert req.indexer_layout == mock_indexer_layout
        assert req.indexer_dtype == torch.uint8

    def test_dataclass_has_indexer_fields(self):
        field_names = [f.name for f in fields(RegisterTPClientRequest)]
        assert 'indexer_handles' in field_names
        assert 'indexer_layout' in field_names
        assert 'indexer_dtype' in field_names

    def test_backward_compatible_construction(self):
        mock_handles = [MagicMock(spec=TensorSharedHandle)]
        mock_layout = MagicMock(spec=KVCacheLayout)
        req = RegisterTPClientRequest(0, 0, mock_handles, mock_layout)
        assert req.dp_client_id == 0
        assert req.device_id == 0
        assert req.handles == mock_handles
        assert req.gpu_layout == mock_layout


# ============================================================
# Module 3: KVCacheLayout for indexer cache
# ============================================================

class TestKVCacheLayoutIndexer:
    """Test KVCacheLayout construction for sparse attention indexer cache."""

    def test_indexer_layout_construction(self):
        layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=61, num_block=2048, tokens_per_block=16,
            num_head=1, head_size=132, is_mla=True,
        )
        assert layout.num_layer == 61
        assert layout.num_head == 1
        assert layout.head_size == 132
        assert layout.is_mla is True

    def test_indexer_layout_vs_main_layout(self):
        main_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=61, num_block=2048, tokens_per_block=16,
            num_head=1, head_size=576, is_mla=True,
        )
        indexer_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=61, num_block=2048, tokens_per_block=16,
            num_head=1, head_size=132, is_mla=True,
        )
        assert main_layout != indexer_layout
        assert main_layout.head_size != indexer_layout.head_size

    def test_indexer_layout_kv_shape(self):
        layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=4, num_block=64, tokens_per_block=16,
            num_head=1, head_size=132, is_mla=True,
        )
        expected_shape = torch.Size([4, 1, 64, 16, 1, 132])
        assert layout.kv_shape == expected_shape

    def test_indexer_layout_elements_per_block(self):
        layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=4, num_block=64, tokens_per_block=16,
            num_head=1, head_size=132, is_mla=True,
        )
        assert layout.get_elements_per_block() == 4 * 1 * 16 * 1 * 132


# ============================================================
# Module 4: StorageEngine - register_indexer_blocks and allocate
# ============================================================

class TestStorageEngineIndexer:
    """Test StorageEngine methods for indexer cache registration."""

    @pytest.fixture
    def basic_model_config(self):
        return ModelConfig(
            num_layers=4, num_kv_heads=1, head_size=576,
            dtype=torch.bfloat16, use_mla=True, tp_size=1, dp_size=1,
        )

    @pytest.fixture
    def basic_cache_config(self):
        return CacheConfig(
            tokens_per_block=16, enable_cpu=True,
            enable_ssd=False, enable_remote=False, num_cpu_blocks=64,
        )

    @pytest.fixture
    def storage_engine(self, basic_model_config, basic_cache_config):
        return StorageEngine(basic_model_config, basic_cache_config)

    def test_no_indexer_handle_before_register(self, storage_engine):
        assert not storage_engine.has_storage_handle(DeviceType.GPU_INDEXER)
        assert not storage_engine.has_storage_handle(DeviceType.CPU_INDEXER)

    def test_allocate_cpu_indexer(self, storage_engine):
        indexer_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=4, num_block=64, tokens_per_block=16,
            num_head=1, head_size=132, is_mla=True,
        )
        result = storage_engine.allocate(
            device_type=DeviceType.CPU_INDEXER,
            layout=indexer_layout, dtype=torch.uint8,
        )
        assert result is True
        assert storage_engine.has_storage_handle(DeviceType.CPU_INDEXER)
        handle = storage_engine.get_storage_handle(DeviceType.CPU_INDEXER)
        assert handle.handle_type == AccessHandleType.TENSOR
        assert handle.dtype == torch.uint8
        assert isinstance(handle.data, torch.Tensor)

    def test_allocate_cpu_indexer_duplicate(self, storage_engine):
        indexer_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=4, num_block=64, tokens_per_block=16,
            num_head=1, head_size=132, is_mla=True,
        )
        r1 = storage_engine.allocate(device_type=DeviceType.CPU_INDEXER,
                                      layout=indexer_layout, dtype=torch.uint8)
        r2 = storage_engine.allocate(device_type=DeviceType.CPU_INDEXER,
                                      layout=indexer_layout, dtype=torch.uint8)
        assert r1 is True
        assert r2 is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_allocate_gpu_indexer_with_raw_data(self, storage_engine):
        indexer_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=2, num_block=32, tokens_per_block=16,
            num_head=1, head_size=132, is_mla=True,
        )
        gpu_tensors = [
            torch.zeros(32, 16, 132, dtype=torch.uint8, device='cuda:0')
            for _ in range(2)
        ]
        result = storage_engine.allocate(
            device_type=DeviceType.GPU_INDEXER,
            layout=indexer_layout, dtype=torch.uint8,
            device_id=0, raw_data=gpu_tensors,
        )
        assert result is True
        assert storage_engine.has_storage_handle(DeviceType.GPU_INDEXER, device_id=0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_register_indexer_blocks_creates_gpu_and_cpu(self, storage_engine):
        indexer_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=2, num_block=32, tokens_per_block=16,
            num_head=1, head_size=132, is_mla=True,
        )
        gpu_tensors = [
            torch.zeros(32, 16, 132, dtype=torch.uint8, device='cuda:0')
            for _ in range(2)
        ]
        handles = [TensorSharedHandle(t, 0) for t in gpu_tensors]
        storage_engine.register_indexer_blocks(
            indexer_blocks=handles, indexer_layout=indexer_layout,
            device_id=0, dtype=torch.uint8,
        )
        assert storage_engine.has_storage_handle(DeviceType.GPU_INDEXER, device_id=0)
        assert storage_engine.has_storage_handle(DeviceType.CPU_INDEXER)


# ============================================================
# Module 5: vLLM v1 adapter - kv_caches splitting
# ============================================================

class TestVLLMAdapterKVCachesSplitting:
    """Test kv_caches dict splitting logic used in FlexKVWorkerConnector."""

    def _build_mock_kv_caches(self, num_main_layers=4, num_indexer_layers=4,
                               num_blocks=64, block_size=16,
                               main_head_size=576, indexer_head_size=132):
        kv_caches = {}
        for i in range(num_main_layers):
            name = f"model.layers.{i}.self_attn.kv_b_proj"
            kv_caches[name] = torch.zeros(
                num_blocks, block_size, main_head_size,
                dtype=torch.bfloat16, device='cpu')
        for i in range(num_indexer_layers):
            name = f"model.layers.{i}.self_attn.indexer.k_cache"
            kv_caches[name] = torch.zeros(
                num_blocks, block_size, indexer_head_size,
                dtype=torch.uint8, device='cpu')
        return kv_caches

    def test_split_with_indexer(self):
        kv_caches = self._build_mock_kv_caches(4, 4)
        main = {n: t for n, t in kv_caches.items() if ".k_cache" not in n}
        indexer = {n: t for n, t in kv_caches.items() if ".k_cache" in n}
        assert len(main) == 4
        assert len(indexer) == 4

    def test_split_without_indexer(self):
        kv_caches = self._build_mock_kv_caches(4, 0)
        main = {n: t for n, t in kv_caches.items() if ".k_cache" not in n}
        indexer = {n: t for n, t in kv_caches.items() if ".k_cache" in n}
        assert len(main) == 4
        assert len(indexer) == 0

    def test_main_layout_construction_mla(self):
        kv_caches = self._build_mock_kv_caches(4, 0)
        gpu_blocks = list(kv_caches.values())
        assert gpu_blocks[0].ndim == 3
        layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=4, num_block=gpu_blocks[0].shape[0],
            tokens_per_block=gpu_blocks[0].shape[1],
            num_head=1, head_size=gpu_blocks[0].shape[2],
            is_mla=True,
        )
        assert layout.num_block == 64
        assert layout.head_size == 576

    def test_indexer_layout_from_tensor_shapes(self):
        kv_caches = self._build_mock_kv_caches(4, 4)
        indexer = {n: t for n, t in kv_caches.items() if ".k_cache" in n}
        blocks = list(indexer.values())
        t = blocks[0]
        assert t.ndim == 3
        assert t.shape == (64, 16, 132)
        assert t.dtype == torch.uint8
        layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=len(indexer), num_block=t.shape[0],
            tokens_per_block=t.shape[1],
            num_head=1, head_size=t.shape[2], is_mla=True,
        )
        assert layout.num_layer == 4
        assert layout.head_size == 132

    def test_non_mla_no_indexer(self):
        kv_caches = {}
        for i in range(4):
            kv_caches[f"model.layers.{i}.self_attn.attn"] = torch.zeros(
                2, 64, 16, 32, 128, dtype=torch.float16, device='cpu')
        indexer = {n: t for n, t in kv_caches.items() if ".k_cache" in n}
        assert len(indexer) == 0


# ============================================================
# Module 6: KVTPClient - register_to_server with indexer
# ============================================================

class TestKVTPClientIndexerRegistration:
    """Test KVTPClient.register_to_server with indexer cache parameters."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_register_without_indexer(self):
        from flexkv.server.client import KVTPClient
        mock_send = MagicMock()
        client = KVTPClient.__new__(KVTPClient)
        client.send_to_server = mock_send
        client.dp_client_id = 0
        client.device_id = 0

        gpu_blocks = [torch.zeros(64, 16, 576, dtype=torch.bfloat16, device='cuda:0')]
        layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=1, num_block=64, tokens_per_block=16,
            num_head=1, head_size=576, is_mla=True,
        )
        client.register_to_server(gpu_blocks, layout)
        mock_send.send_pyobj.assert_called_once()
        req = mock_send.send_pyobj.call_args[0][0]
        assert isinstance(req, RegisterTPClientRequest)
        assert req.indexer_handles is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_register_with_indexer(self):
        from flexkv.server.client import KVTPClient
        mock_send = MagicMock()
        client = KVTPClient.__new__(KVTPClient)
        client.send_to_server = mock_send
        client.dp_client_id = 0
        client.device_id = 0

        gpu_blocks = [torch.zeros(64, 16, 576, dtype=torch.bfloat16, device='cuda:0')]
        layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=1, num_block=64, tokens_per_block=16,
            num_head=1, head_size=576, is_mla=True,
        )
        indexer_blocks = [torch.zeros(64, 16, 132, dtype=torch.uint8, device='cuda:0')]
        indexer_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=1, num_block=64, tokens_per_block=16,
            num_head=1, head_size=132, is_mla=True,
        )
        client.register_to_server(
            gpu_blocks, layout,
            indexer_caches=indexer_blocks,
            indexer_layout=indexer_layout,
            indexer_dtype=torch.uint8,
        )
        mock_send.send_pyobj.assert_called_once()
        req = mock_send.send_pyobj.call_args[0][0]
        assert req.indexer_handles is not None
        assert len(req.indexer_handles) == 1
        assert req.indexer_layout == indexer_layout
        assert req.indexer_dtype == torch.uint8


# ============================================================
# Module 7: TransferManager - indexer registration
# ============================================================

class TestTransferManagerIndexerRegistration:
    """Test TransferManager handles indexer cache registration."""

    def test_without_indexer(self):
        from flexkv.transfer_manager import TransferManager
        tm = TransferManager.__new__(TransferManager)
        tm.all_gpu_blocks = {}
        tm.all_gpu_layouts = {}
        tm.gpu_client_mapping = {}
        tm.all_indexer_blocks = {}
        tm.all_indexer_layouts = {}
        tm.all_indexer_dtypes = {}

        req = RegisterTPClientRequest(
            dp_client_id=0, device_id=0,
            handles=[MagicMock(spec=TensorSharedHandle)],
            gpu_layout=MagicMock(spec=KVCacheLayout),
        )
        tm._handle_gpu_blocks_registration(req)
        assert 0 in tm.all_gpu_blocks
        assert 0 not in tm.all_indexer_blocks

    def test_with_indexer(self):
        from flexkv.transfer_manager import TransferManager
        tm = TransferManager.__new__(TransferManager)
        tm.all_gpu_blocks = {}
        tm.all_gpu_layouts = {}
        tm.gpu_client_mapping = {}
        tm.all_indexer_blocks = {}
        tm.all_indexer_layouts = {}
        tm.all_indexer_dtypes = {}

        mock_indexer_layout = MagicMock(spec=KVCacheLayout)
        mock_indexer_layout.num_layer = 4
        mock_indexer_layout.head_size = 132

        req = RegisterTPClientRequest(
            dp_client_id=0, device_id=0,
            handles=[MagicMock(spec=TensorSharedHandle)],
            gpu_layout=MagicMock(spec=KVCacheLayout),
            indexer_handles=[MagicMock(spec=TensorSharedHandle)],
            indexer_layout=mock_indexer_layout,
            indexer_dtype=torch.uint8,
        )
        tm._handle_gpu_blocks_registration(req)
        assert 0 in tm.all_gpu_blocks
        assert 0 in tm.all_indexer_blocks
        assert tm.all_indexer_dtypes[0] == torch.uint8

    def test_duplicate_registration(self):
        from flexkv.transfer_manager import TransferManager
        tm = TransferManager.__new__(TransferManager)
        tm.all_gpu_blocks = {}
        tm.all_gpu_layouts = {}
        tm.gpu_client_mapping = {}
        tm.all_indexer_blocks = {}
        tm.all_indexer_layouts = {}
        tm.all_indexer_dtypes = {}

        mock_handles = [MagicMock(spec=TensorSharedHandle)]
        req = RegisterTPClientRequest(
            dp_client_id=0, device_id=0,
            handles=mock_handles,
            gpu_layout=MagicMock(spec=KVCacheLayout),
        )
        tm._handle_gpu_blocks_registration(req)
        assert 0 in tm.all_gpu_blocks
        # Second registration should not crash
        tm._handle_gpu_blocks_registration(req)
        assert tm.all_gpu_blocks[0] == mock_handles


# ============================================================
# Integration: End-to-end kv_caches splitting test
# ============================================================

class TestEndToEndKVCacheSplitting:
    """End-to-end test: from vLLM kv_caches dict to layout construction."""

    def test_deepseek_v3_full_pipeline(self):
        num_layers = 61
        kv_caches = {}
        for i in range(num_layers):
            kv_caches[f"model.layers.{i}.self_attn.kv_b_proj"] = torch.zeros(
                128, 16, 576, dtype=torch.bfloat16, device='cpu')
        for i in range(num_layers):
            kv_caches[f"model.layers.{i}.self_attn.indexer.k_cache"] = torch.zeros(
                128, 16, 132, dtype=torch.uint8, device='cpu')

        main = {n: t for n, t in kv_caches.items() if ".k_cache" not in n}
        indexer = {n: t for n, t in kv_caches.items() if ".k_cache" in n}
        assert len(main) == 61
        assert len(indexer) == 61

        main_blocks = list(main.values())
        main_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=61, num_block=128, tokens_per_block=16,
            num_head=1, head_size=576, is_mla=True,
        )
        indexer_blocks = list(indexer.values())
        indexer_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=61, num_block=128, tokens_per_block=16,
            num_head=1, head_size=132, is_mla=True,
        )
        assert main_layout.num_block == indexer_layout.num_block
        assert main_layout.tokens_per_block == indexer_layout.tokens_per_block
        assert main_layout.head_size != indexer_layout.head_size

    def test_standard_model_no_indexer(self):
        kv_caches = {}
        for i in range(32):
            kv_caches[f"model.layers.{i}.self_attn.kv_b_proj"] = torch.zeros(
                128, 16, 576, dtype=torch.bfloat16, device='cpu')
        indexer = {n: t for n, t in kv_caches.items() if ".k_cache" in n}
        assert len(indexer) == 0
        main = {n: t for n, t in kv_caches.items() if ".k_cache" not in n}
        assert len(main) == 32


# ============================================================
# Module 8: TransferEngine - indexer worker initialization
# ============================================================

class TestTransferEngineIndexer:
    """Test TransferEngine initialization with sparse attention indexer handles."""

    def test_init_without_indexer(self):
        """TransferEngine without indexer should have _has_indexer=False."""
        from flexkv.transfer.transfer_engine import TransferEngine
        mock_gpu_handle = MagicMock()
        mock_gpu_handle.dtype = torch.bfloat16
        mock_gpu_handle.gpu_device_id = 0
        mock_gpu_handle.kv_layout = MagicMock()

        mock_cpu_handle = MagicMock()
        mock_cpu_handle.dtype = torch.bfloat16
        mock_cpu_handle.kv_layout = MagicMock()

        model_config = ModelConfig(
            num_layers=4, num_kv_heads=1, head_size=576,
            dtype=torch.bfloat16, use_mla=True, tp_size=1, dp_size=1,
        )
        cache_config = CacheConfig(
            tokens_per_block=16, enable_cpu=True,
            enable_ssd=False, enable_remote=False, num_cpu_blocks=64,
        )

        engine = TransferEngine(
            gpu_handles={0: [mock_gpu_handle]},
            model_config=model_config,
            cache_config=cache_config,
            cpu_handle=mock_cpu_handle,
        )
        assert engine._has_indexer is False
        assert engine._indexer_gpu_handles is None
        assert engine._indexer_cpu_handle is None
        assert engine.indexer_finished_ops_queue is None

    def test_init_with_indexer(self):
        """TransferEngine with indexer handles should have _has_indexer=True."""
        from flexkv.transfer.transfer_engine import TransferEngine
        mock_gpu_handle = MagicMock()
        mock_gpu_handle.dtype = torch.bfloat16
        mock_gpu_handle.gpu_device_id = 0
        mock_gpu_handle.kv_layout = MagicMock()

        mock_cpu_handle = MagicMock()
        mock_cpu_handle.dtype = torch.bfloat16
        mock_cpu_handle.kv_layout = MagicMock()

        mock_indexer_gpu_handle = MagicMock()
        mock_indexer_gpu_handle.dtype = torch.uint8
        mock_indexer_gpu_handle.gpu_device_id = 0
        mock_indexer_gpu_handle.kv_layout = MagicMock()

        mock_indexer_cpu_handle = MagicMock()
        mock_indexer_cpu_handle.dtype = torch.uint8
        mock_indexer_cpu_handle.kv_layout = MagicMock()

        model_config = ModelConfig(
            num_layers=4, num_kv_heads=1, head_size=576,
            dtype=torch.bfloat16, use_mla=True, tp_size=1, dp_size=1,
        )
        cache_config = CacheConfig(
            tokens_per_block=16, enable_cpu=True,
            enable_ssd=False, enable_remote=False, num_cpu_blocks=64,
        )

        engine = TransferEngine(
            gpu_handles={0: [mock_gpu_handle]},
            model_config=model_config,
            cache_config=cache_config,
            cpu_handle=mock_cpu_handle,
            indexer_gpu_handles={0: [mock_indexer_gpu_handle]},
            indexer_cpu_handle=mock_indexer_cpu_handle,
        )
        assert engine._has_indexer is True
        assert engine._indexer_gpu_handles is not None
        assert engine._indexer_cpu_handle is not None
        assert engine.indexer_finished_ops_queue is not None

    def test_init_with_empty_indexer_handles(self):
        """TransferEngine with empty indexer dict should have _has_indexer=False."""
        from flexkv.transfer.transfer_engine import TransferEngine
        mock_gpu_handle = MagicMock()
        mock_gpu_handle.dtype = torch.bfloat16
        mock_gpu_handle.gpu_device_id = 0
        mock_gpu_handle.kv_layout = MagicMock()

        mock_cpu_handle = MagicMock()
        mock_cpu_handle.dtype = torch.bfloat16
        mock_cpu_handle.kv_layout = MagicMock()

        model_config = ModelConfig(
            num_layers=4, num_kv_heads=1, head_size=576,
            dtype=torch.bfloat16, use_mla=True, tp_size=1, dp_size=1,
        )
        cache_config = CacheConfig(
            tokens_per_block=16, enable_cpu=True,
            enable_ssd=False, enable_remote=False, num_cpu_blocks=64,
        )

        engine = TransferEngine(
            gpu_handles={0: [mock_gpu_handle]},
            model_config=model_config,
            cache_config=cache_config,
            cpu_handle=mock_cpu_handle,
            indexer_gpu_handles={},
            indexer_cpu_handle=None,
        )
        assert engine._has_indexer is False

    def test_indexer_separate_finished_queue(self):
        """Indexer workers should use a separate finished_ops_queue."""
        from flexkv.transfer.transfer_engine import TransferEngine
        mock_gpu_handle = MagicMock()
        mock_gpu_handle.dtype = torch.bfloat16
        mock_gpu_handle.gpu_device_id = 0
        mock_gpu_handle.kv_layout = MagicMock()

        mock_cpu_handle = MagicMock()
        mock_indexer_gpu_handle = MagicMock()
        mock_indexer_cpu_handle = MagicMock()

        model_config = ModelConfig(
            num_layers=4, num_kv_heads=1, head_size=576,
            dtype=torch.bfloat16, use_mla=True, tp_size=1, dp_size=1,
        )
        cache_config = CacheConfig(
            tokens_per_block=16, enable_cpu=True,
            enable_ssd=False, enable_remote=False, num_cpu_blocks=64,
        )

        engine = TransferEngine(
            gpu_handles={0: [mock_gpu_handle]},
            model_config=model_config,
            cache_config=cache_config,
            cpu_handle=mock_cpu_handle,
            indexer_gpu_handles={0: [mock_indexer_gpu_handle]},
            indexer_cpu_handle=mock_indexer_cpu_handle,
        )
        # indexer finished queue should be separate from main finished queue
        assert engine.indexer_finished_ops_queue is not None
        assert engine.indexer_finished_ops_queue is not engine.finished_ops_queue
