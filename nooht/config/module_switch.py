"""
Nooht Module Switch — 全局模块开关控制
每个模块可独立启用/禁用，低 B 模型不加载高 B 功能。
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ModuleSwitch:
    attention: bool = True
    ffn: bool = True
    rmsnorm: bool = True
    rope: bool = True

    memory_bank: bool = False
    bypass_gate: bool = False
    self_verifier: bool = False
    budget_predictor: bool = False

    vision_encoder: bool = False
    audio_encoder: bool = False
    multimodal_fusion: bool = False

    thinking_module: bool = False
    tool_caller: bool = False

    fsdp: bool = False
    pipeline_parallel: bool = False
    sequence_parallel: bool = False
    gradient_checkpointing: bool = False

    def enable(self, module_name: str):
        if hasattr(self, module_name): setattr(self, module_name, True)

    def disable(self, module_name: str):
        if hasattr(self, module_name): setattr(self, module_name, False)

    def is_enabled(self, module_name: str) -> bool:
        return getattr(self, module_name, False)

    def to_dict(self) -> Dict[str, bool]:
        return {k: v for k, v in self.__dict__.items() if isinstance(v, bool)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModuleSwitch":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "1M": {"dim": 128, "num_layers": 2, "num_heads": 2, "ffn_multiplier": 2, "num_memory_slots": 0, "vocab_size": 8000, "max_seq_len": 512, "modules": {"memory_bank": False, "bypass_gate": False, "self_verifier": False, "vision_encoder": False, "audio_encoder": False, "thinking_module": False, "tool_caller": False, "fsdp": False}},
    "10M": {"dim": 256, "num_layers": 4, "num_heads": 4, "ffn_multiplier": 4, "num_memory_slots": 8, "vocab_size": 16000, "max_seq_len": 1024, "modules": {"memory_bank": True, "bypass_gate": False, "self_verifier": False, "vision_encoder": False, "audio_encoder": False, "thinking_module": False, "tool_caller": False, "fsdp": False}},
    "50M": {"dim": 512, "num_layers": 6, "num_heads": 8, "ffn_multiplier": 4, "num_memory_slots": 16, "vocab_size": 32000, "max_seq_len": 2048, "modules": {"memory_bank": True, "bypass_gate": True, "self_verifier": False, "vision_encoder": False, "audio_encoder": False, "thinking_module": False, "tool_caller": False, "fsdp": False}},
    "150M": {"dim": 768, "num_layers": 12, "num_heads": 12, "ffn_multiplier": 4, "num_memory_slots": 32, "vocab_size": 32000, "max_seq_len": 2048, "modules": {"memory_bank": True, "bypass_gate": True, "self_verifier": True, "vision_encoder": False, "audio_encoder": False, "thinking_module": False, "tool_caller": False, "fsdp": False}},
    "350M": {"dim": 1024, "num_layers": 24, "num_heads": 16, "ffn_multiplier": 4, "num_memory_slots": 64, "vocab_size": 32000, "max_seq_len": 4096, "modules": {"memory_bank": True, "bypass_gate": True, "self_verifier": True, "vision_encoder": False, "audio_encoder": False, "thinking_module": True, "tool_caller": False, "fsdp": False}},
    "1B": {"dim": 2048, "num_layers": 24, "num_heads": 16, "ffn_multiplier": 4, "num_memory_slots": 64, "vocab_size": 32000, "max_seq_len": 4096, "modules": {"memory_bank": True, "bypass_gate": True, "self_verifier": True, "vision_encoder": True, "audio_encoder": False, "thinking_module": True, "tool_caller": True, "fsdp": False}},
    "3B": {"dim": 2560, "num_layers": 32, "num_heads": 32, "ffn_multiplier": 4, "num_memory_slots": 128, "vocab_size": 32000, "max_seq_len": 8192, "modules": {"memory_bank": True, "bypass_gate": True, "self_verifier": True, "vision_encoder": True, "audio_encoder": True, "thinking_module": True, "tool_caller": True, "fsdp": False}},
    "7B": {"dim": 4096, "num_layers": 32, "num_heads": 32, "ffn_multiplier": 4, "num_memory_slots": 128, "vocab_size": 32000, "max_seq_len": 8192, "modules": {"memory_bank": True, "bypass_gate": True, "self_verifier": True, "vision_encoder": True, "audio_encoder": True, "thinking_module": True, "tool_caller": True, "fsdp": True, "gradient_checkpointing": True}},
    "13B": {"dim": 5120, "num_layers": 40, "num_heads": 40, "ffn_multiplier": 4, "num_memory_slots": 256, "vocab_size": 32000, "max_seq_len": 16384, "modules": {"memory_bank": True, "bypass_gate": True, "self_verifier": True, "vision_encoder": True, "audio_encoder": True, "thinking_module": True, "tool_caller": True, "fsdp": True, "gradient_checkpointing": True, "sequence_parallel": True}},
    "30B": {"dim": 6656, "num_layers": 60, "num_heads": 52, "ffn_multiplier": 4, "num_memory_slots": 256, "vocab_size": 32000, "max_seq_len": 32768, "modules": {"memory_bank": True, "bypass_gate": True, "self_verifier": True, "vision_encoder": True, "audio_encoder": True, "thinking_module": True, "tool_caller": True, "fsdp": True, "pipeline_parallel": True, "gradient_checkpointing": True, "sequence_parallel": True}},
    "70B": {"dim": 8192, "num_layers": 80, "num_heads": 64, "ffn_multiplier": 4, "num_memory_slots": 512, "vocab_size": 32000, "max_seq_len": 65536, "modules": {"memory_bank": True, "bypass_gate": True, "self_verifier": True, "vision_encoder": True, "audio_encoder": True, "thinking_module": True, "tool_caller": True, "fsdp": True, "pipeline_parallel": True, "gradient_checkpointing": True, "sequence_parallel": True}},
    "100B": {"dim": 10240, "num_layers": 96, "num_heads": 80, "ffn_multiplier": 4, "num_memory_slots": 512, "vocab_size": 64000, "max_seq_len": 131072, "modules": {"memory_bank": True, "bypass_gate": True, "self_verifier": True, "vision_encoder": True, "audio_encoder": True, "thinking_module": True, "tool_caller": True, "fsdp": True, "pipeline_parallel": True, "gradient_checkpointing": True, "sequence_parallel": True}},
}

def get_model_config(preset: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(MODEL_PRESETS.keys())}")
    config = MODEL_PRESETS[preset].copy()
    module_flags = config.pop("modules", {})
    config["module_switch"] = ModuleSwitch.from_dict(module_flags)
    if overrides:
        config.update(overrides)
    return config
