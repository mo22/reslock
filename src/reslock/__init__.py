"""reslock — Resource lock manager for coordinating shared system resources."""

from __future__ import annotations

from reslock.audit import OrphanReport, gpu_orphans
from reslock.detect import (
    CPU_CORES_KEY,
    RAM_MB_KEY,
    disk_mb_key,
    get_disk_free_mb,
    gpu_resource_key,
    gpu_uuid_for_torch_index,
    gpu_vram_key,
    parse_disk_mb_key,
    parse_gpu_vram_key,
)
from reslock.models import SCHEMA_VERSION, Lease, PoolStatus, QueueEntry, State
from reslock.pool import EntryHandle, LeaseHandle, ResourcePool
from reslock.resources import (
    detect_cpu_cores,
    detect_disk_mb,
    detect_gpu_vram_mb,
    detect_gpu_vram_mb_cuda_driver,
    detect_gpu_vram_mb_nvidia_smi,
    detect_gpu_vram_mb_torch,
    detect_network_bandwidth,
    detect_ram_mb,
)
from reslock.state import SchemaVersionMismatch, force_reset_state, peek_state_version

__all__ = [
    "CPU_CORES_KEY",
    "RAM_MB_KEY",
    "SCHEMA_VERSION",
    "EntryHandle",
    "Lease",
    "LeaseHandle",
    "OrphanReport",
    "PoolStatus",
    "QueueEntry",
    "ResourcePool",
    "SchemaVersionMismatch",
    "State",
    "detect_cpu_cores",
    "detect_disk_mb",
    "detect_gpu_vram_mb",
    "detect_gpu_vram_mb_cuda_driver",
    "detect_gpu_vram_mb_nvidia_smi",
    "detect_gpu_vram_mb_torch",
    "detect_network_bandwidth",
    "detect_ram_mb",
    "disk_mb_key",
    "force_reset_state",
    "get_disk_free_mb",
    "gpu_orphans",
    "parse_disk_mb_key",
    "gpu_resource_key",
    "gpu_uuid_for_torch_index",
    "gpu_vram_key",
    "parse_gpu_vram_key",
    "peek_state_version",
]
