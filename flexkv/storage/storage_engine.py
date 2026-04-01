from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, List, Tuple, Union

import torch
import hashlib

from flexkv.common.config import ModelConfig, CacheConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import StorageHandle, KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import DeviceType
from flexkv.storage.allocator import CPUAllocator, GPUAllocator, SSDAllocator, RemoteAllocator


class StorageEngine:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig):
        """Initialize storage engine"""
        self._storage_handles: Dict[Tuple[DeviceType, int], StorageHandle] = {}
        self._indexer_storage_handles: Dict[Tuple[DeviceType, int], StorageHandle] = {}
        self._model_config = model_config
        self._cache_config = cache_config
        self._indexer_config = cache_config.indexer

        if self._cache_config.enable_cpu:
            self._cpu_layout: Optional[KVCacheLayout] = KVCacheLayout(
                type=GLOBAL_CONFIG_FROM_ENV.cpu_layout_type,
                num_layer=self._model_config.num_layers,
                num_block=self._cache_config.num_cpu_blocks,
                tokens_per_block=self._cache_config.tokens_per_block,
                num_head=self._model_config.num_kv_heads,
                head_size=self._model_config.head_size,
                is_mla=self._model_config.use_mla
            )
            self.allocate(
                device_type=DeviceType.CPU,
                layout=self._cpu_layout,
                dtype=self._model_config.dtype,
            )
            if self._indexer_config is not None:
                indexer_cpu_layout = KVCacheLayout(
                    type=GLOBAL_CONFIG_FROM_ENV.cpu_layout_type,
                    num_layer=self._model_config.num_layers,
                    num_block=self._cache_config.num_cpu_blocks,
                    tokens_per_block=self._cache_config.tokens_per_block,
                    num_head=self._indexer_config.num_kv_heads,
                    head_size=self._indexer_config.head_size,
                    is_mla=True
                )
                self.allocate_indexer(
                    device_type=DeviceType.CPU,
                    layout=indexer_cpu_layout,
                    dtype=self._indexer_config.dtype,
                )
                indexer_cpu_handle = self._indexer_storage_handles[(DeviceType.CPU, 0)]

        if self._cache_config.enable_ssd:
            if not GLOBAL_CONFIG_FROM_ENV.ssd_layout_type == self._cpu_layout.type:
                raise ValueError(f"SSD layout type must be the same as CPU layout type: {self._cpu_layout.type}")
            self._ssd_layout: Optional[KVCacheLayout] = KVCacheLayout(
                type=GLOBAL_CONFIG_FROM_ENV.ssd_layout_type,
                num_layer=self._model_config.num_layers,
                num_block=self._cache_config.num_ssd_blocks,
                tokens_per_block=self._cache_config.tokens_per_block,
                num_head=self._model_config.num_kv_heads,
                head_size=self._model_config.head_size,
                is_mla=self._model_config.use_mla
            )
            self.allocate(
                device_type=DeviceType.SSD,
                layout=self._ssd_layout,
                dtype=self._model_config.dtype,
                cache_dir=self._cache_config.ssd_cache_dir,
                max_file_size_gb=GLOBAL_CONFIG_FROM_ENV.max_file_size_gb
            )
            if self._indexer_config is not None:
                indexer_ssd_layout = KVCacheLayout(
                    type=GLOBAL_CONFIG_FROM_ENV.ssd_layout_type,
                    num_layer=self._model_config.num_layers,
                    num_block=self._cache_config.num_ssd_blocks,
                    tokens_per_block=self._cache_config.tokens_per_block,
                    num_head=self._indexer_config.num_kv_heads,
                    head_size=self._indexer_config.head_size,
                    is_mla=True
                )
                self.allocate_indexer(
                    device_type=DeviceType.SSD,
                    layout=indexer_ssd_layout,
                    dtype=self._indexer_config.dtype,
                    cache_dir=self._cache_config.ssd_cache_dir,
                    max_file_size_gb=GLOBAL_CONFIG_FROM_ENV.max_file_size_gb,
                )

        if self._cache_config.enable_remote:
            if not GLOBAL_CONFIG_FROM_ENV.remote_layout_type == self._cpu_layout.type:
                raise ValueError(f"Remote layout type must be the same as CPU layout type: {self._cpu_layout.type}")
            self._remote_layout: Optional[KVCacheLayout] = KVCacheLayout(
                type=GLOBAL_CONFIG_FROM_ENV.remote_layout_type,
                num_layer=self._model_config.num_layers,
                num_block=self._cache_config.num_remote_blocks,
                tokens_per_block=self._cache_config.tokens_per_block,
                num_head=self._model_config.num_kv_heads,
                head_size=self._model_config.head_size,
                is_mla=self._model_config.use_mla
            )
            self.allocate(
                device_type=DeviceType.REMOTE,
                layout=self._remote_layout,
                dtype=self._model_config.dtype,
                file_path=self._cache_config.remote_cache_path,
                remote_config_custom = self._cache_config.remote_config_custom
            )
            if self._indexer_config is not None:
                indexer_remote_layout = KVCacheLayout(
                    type=GLOBAL_CONFIG_FROM_ENV.remote_layout_type,
                    num_layer=self._model_config.num_layers,
                    num_block=self._cache_config.num_remote_blocks,
                    tokens_per_block=self._cache_config.tokens_per_block,
                    num_head=self._indexer_config.num_kv_heads,
                    head_size=self._indexer_config.head_size,
                    is_mla=True
                )
                indexer_remote_path = self._cache_config.remote_cache_path
                if isinstance(indexer_remote_path, str):
                    indexer_remote_path = indexer_remote_path + "_indexer"
                elif isinstance(indexer_remote_path, list):
                    indexer_remote_path = [p + "_indexer" for p in indexer_remote_path]
                self.allocate_indexer(
                    device_type=DeviceType.REMOTE,
                    layout=indexer_remote_layout,
                    dtype=self._indexer_config.dtype,
                    file_path=indexer_remote_path,
                    remote_config_custom=self._cache_config.remote_config_custom,
                )

    @property
    def _has_indexer(self) -> bool:
        """True when indexer is configured and CPU buffer is allocated."""
        return (DeviceType.CPU, 0) in self._indexer_storage_handles

    def register_gpu_blocks(self,
                            gpu_blocks: List[TensorSharedHandle],
                            gpu_layout: KVCacheLayout,
                            device_id: int = 0,
                            dtype: torch.dtype = torch.float16,
                            indexer_gpu_blocks: Optional[List[TensorSharedHandle]] = None,
                            indexer_gpu_layout: Optional[KVCacheLayout] = None,
                            indexer_dtype: Optional[torch.dtype] = None) -> None:
        self.allocate(
            device_type=DeviceType.GPU,
            layout=gpu_layout,
            dtype=dtype,
            device_id=device_id,
            raw_data=gpu_blocks
        )
        if indexer_gpu_blocks is not None:
            self.allocate_indexer(
                device_type=DeviceType.GPU,
                layout=indexer_gpu_layout,
                dtype=indexer_dtype if indexer_dtype is not None else dtype,
                device_id=device_id,
                raw_data=indexer_gpu_blocks,
            )

    def _allocate_impl(self,
                       storage_handles: Dict[Tuple[DeviceType, int], StorageHandle],
                       ssd_prefix_tag: str,
                       device_type: DeviceType,
                       layout: KVCacheLayout,
                       dtype: torch.dtype,
                       device_id: int = 0,
                       raw_data: Optional[Union[List[TensorSharedHandle], List[str], str]] = None,
                       **kwargs: Any) -> bool:
        """
        Internal implementation shared by allocate() and allocate_indexer().

        Args:
            storage_handles: The dict to check and write the resulting handle into.
            ssd_prefix_tag: Infix used in the SSD file_prefix to distinguish main KV
                            (``""`` → ``flexkv_ssdcache_<hash>``) from indexer
                            (``"indexer_"`` → ``flexkv_indexer_ssdcache_<hash>``).
            device_type: Type of the device (CPU, GPU, SSD, REMOTE).
            layout: Layout of kv cache.
            dtype: Data type of tensors.
            device_id: Device ID (default 0).
            raw_data: Optional raw data to be used for initialization.
                      The expected type depends on ``device_type``:

                      * ``DeviceType.CPU``    – ``torch.Tensor``
                      * ``DeviceType.GPU``    – ``List[TensorSharedHandle]`` or
                                               ``List[torch.Tensor]``
                      * ``DeviceType.SSD``    – ``str`` or ``List[str]``
                        (file path(s) to existing SSD cache files)
                      * ``DeviceType.REMOTE`` – ``str`` or ``List[str]``
                        (remote file path(s))
            **kwargs: Additional arguments for specific allocator types
                     (e.g., pin_memory for CPU, file_path for Disk).

        Returns:
            bool: True if allocator created successfully, False if already exists.
        """
        key = (device_type, device_id)
        if key in storage_handles:
            return False

        storage_handle: StorageHandle
        if device_type == DeviceType.CPU:
            pin_memory = kwargs.get('pin_memory', False)
            if raw_data is not None:
                assert isinstance(raw_data, torch.Tensor), \
                    "raw_data for CPUAllocator must be Tensor"
                storage_handle = CPUAllocator.from_raw_data(
                    data=raw_data,  # type: ignore
                    layout=layout,
                    dtype=dtype,
                    pin_memory=pin_memory
                )
            else:
                storage_handle = CPUAllocator.allocate(
                    layout=layout,
                    dtype=dtype,
                    pin_memory=pin_memory
                )
        elif device_type == DeviceType.GPU:
            num_chunks = kwargs.get('num_chunks', 1)
            if raw_data is not None:
                assert isinstance(raw_data, list) and \
                    (all(isinstance(x, TensorSharedHandle) for x in raw_data) or \
                     all(isinstance(x, torch.Tensor) for x in raw_data)), \
                    "raw_data for GPUAllocator must be List[TensorSharedHandle] or List[Tensor]"
                storage_handle = GPUAllocator.from_raw_data(
                    data=raw_data,  # type: ignore
                    layout=layout,
                    dtype=dtype,
                    device_id=device_id
                )
            else:
                storage_handle = GPUAllocator.allocate(
                    layout=layout,
                    dtype=dtype,
                    num_chunks=num_chunks,
                    device_id=device_id
                )
        elif device_type == DeviceType.SSD:
            cache_dir = kwargs.get('cache_dir')
            max_file_size_gb = kwargs.get('max_file_size_gb', -1)
            if raw_data is not None:
                assert isinstance(raw_data, str) or \
                    (isinstance(raw_data, list) and all(isinstance(x, str) for x in raw_data)), \
                    "raw_data for SSDAllocator must be str or List[str]"
                storage_handle = SSDAllocator.from_raw_data(
                    data=raw_data,  # type: ignore
                    layout=layout,
                    dtype=dtype,
                )
            else:
                if not cache_dir:
                    raise ValueError("cache_dir is required for SSD allocator")
                server_recv_port = GLOBAL_CONFIG_FROM_ENV.server_recv_port
                hash_value = hashlib.md5(server_recv_port.encode()).hexdigest()
                rand_suffix = f"{hash_value[:6]}"
                file_prefix = f"flexkv_{ssd_prefix_tag}ssdcache_{rand_suffix}"
                storage_handle = SSDAllocator.allocate(
                    layout=layout,
                    dtype=dtype,
                    cache_dir=cache_dir,
                    file_prefix=file_prefix,
                    max_file_size_gb=max_file_size_gb
                )
        elif device_type == DeviceType.REMOTE:
            file_path = kwargs.get('file_path')
            remote_config_custom = kwargs.get('remote_config_custom')
            if raw_data is not None:
                if (isinstance(raw_data, str) or \
                    (isinstance(raw_data, list) and all(isinstance(x, str) for x in raw_data))):
                    if not isinstance(remote_config_custom, dict):
                        raise TypeError("remote_config_custom for RemoteAllocator.from_raw_data must be dict[str, Any]")
                    storage_handle = RemoteAllocator.from_raw_data(
                        data=raw_data,  # type: ignore
                        layout=layout,
                        dtype=dtype,
                        remote_config_custom=remote_config_custom
                    )
                else:
                    raise TypeError("raw_data for RemoteAllocator must be str or List[str]")
            else:
                if not file_path:
                    raise ValueError("file_path is required for remote allocator")
                if not isinstance(remote_config_custom, dict):
                    raise TypeError("remote_config_custom for RemoteAllocator must be dict[str, Any]")
                storage_handle = RemoteAllocator.allocate(
                    layout=layout,
                    dtype=dtype,
                    file_path=file_path,
                    remote_config_custom=remote_config_custom
                )
        else:
            raise ValueError(f"Unsupported device type: {device_type}")
        storage_handles[key] = storage_handle
        return True

    def allocate(self,
                 device_type: DeviceType,
                 layout: KVCacheLayout,
                 dtype: torch.dtype,
                 device_id: int = 0,
                 raw_data: Optional[Union[List[TensorSharedHandle], List[str], str]] = None,
                 **kwargs: Any) -> bool:
        """
        Create and add an allocator for specified device.

        Args:
            device_type: Type of the device (CPU, GPU, etc.)
            layout: Layout of kv cache
            dtype: Data type of tensors
            device_id: Device ID (default 0)
            raw_data: Optional raw data to be used for initialization
            **kwargs: Additional arguments for specific allocator types
                     (e.g., pin_memory for CPU, file_path for Disk)

        Returns:
            bool: True if allocator created successfully, False otherwise
        """
        return self._allocate_impl(
            self._storage_handles, "",
            device_type, layout, dtype, device_id, raw_data, **kwargs
        )

    def allocate_indexer(self,
                         device_type: DeviceType,
                         layout: KVCacheLayout,
                         dtype: torch.dtype,
                         device_id: int = 0,
                         raw_data: Optional[Union[List[TensorSharedHandle], List[str], str]] = None,
                         **kwargs: Any) -> bool:
        """
        Allocate indexer storage handle for specified device.
        Mirrors allocate() but writes into _indexer_storage_handles.
        For SSD, the file_prefix uses 'indexer_' tag to avoid collision with
        main KV SSD files (e.g. ``flexkv_indexer_ssdcache_<hash>``).

        Returns:
            bool: True if allocator created successfully, False otherwise
        """
        return self._allocate_impl(
            self._indexer_storage_handles, "indexer_",
            device_type, layout, dtype, device_id, raw_data, **kwargs
        )

    def _get_handle_impl(self,
                         storage_handles: Dict[Tuple[DeviceType, int], StorageHandle],
                         label: str,
                         device_type: DeviceType,
                         device_id: int = 0) -> StorageHandle:
        """Internal implementation shared by get_storage_handle() and get_indexer_storage_handle()."""
        key = (device_type, device_id)
        if key not in storage_handles:
            raise ValueError(f"{label} storage handle not found for device type: {device_type}, device id: {device_id}")
        return storage_handles[key]

    def _has_handle_impl(self,
                         storage_handles: Dict[Tuple[DeviceType, int], StorageHandle],
                         device_type: DeviceType,
                         device_id: int = 0) -> bool:
        """Internal implementation shared by has_storage_handle() and has_indexer_storage_handle()."""
        return (device_type, device_id) in storage_handles

    def get_storage_handle(self,
                           device_type: DeviceType,
                           device_id: int = 0) -> StorageHandle:
        """
        Get accessible handle for specified blocks

        Args:
            device_type: Type of the device to get handle from
            device_id: Device ID
        """
        return self._get_handle_impl(self._storage_handles, "Storage", device_type, device_id)

    def has_storage_handle(self,
                           device_type: DeviceType,
                           device_id: int = 0) -> bool:
        """Check if storage handle exists for given device type and id"""
        return self._has_handle_impl(self._storage_handles, device_type, device_id)

    def get_indexer_storage_handle(self,
                                   device_type: DeviceType,
                                   device_id: int = 0) -> StorageHandle:
        """
        Get indexer storage handle for specified device

        Args:
            device_type: Type of the device to get handle from
            device_id: Device ID
        """
        return self._get_handle_impl(self._indexer_storage_handles, "Indexer storage", device_type, device_id)

    def has_indexer_storage_handle(self,
                                   device_type: DeviceType,
                                   device_id: int = 0) -> bool:
        """Check if indexer storage handle exists for given device type and id"""
        return self._has_handle_impl(self._indexer_storage_handles, device_type, device_id)
