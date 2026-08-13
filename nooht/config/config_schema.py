from dataclasses import dataclass, field
from typing import Dict, List, Any
from enum import Enum

class DeviceType(Enum): GPU="gpu"; TPU="tpu"; CPU="cpu"
class ComputePrecision(Enum): FP32="fp32"; FP16="fp16"; BF16="bf16"

@dataclass
class ModuleConfig: name: str; enabled: bool = True

@dataclass
class KTUConfig(ModuleConfig): char_dim: int = 128; token_dim: int = 512; structure_dim: int = 256; ktu_dim: int = 768; name: str = "ktu"

@dataclass
class MemoryConfig(ModuleConfig): l0_max_bytes: int = 16*(1024**3); l1_max_size: int = 1000; l2_max_size: int = 100; name: str = "memory"

@dataclass
class SchedulerConfig(ModuleConfig): max_concurrent_tasks: int = 100; task_history_limit: int = 10000; name: str = "scheduler"

@dataclass
class RuntimeConfig(ModuleConfig): devices: List[DeviceType] = field(default_factory=lambda: [DeviceType.CPU]); precision: ComputePrecision = ComputePrecision.FP32; num_workers: int = 4; name: str = "runtime"

@dataclass
class ConfigSchema:
    project_name: str = "Nooht"; version: str = "2.0.0"
    ktu: KTUConfig = field(default_factory=KTUConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
