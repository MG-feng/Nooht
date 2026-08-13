from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModuleSwitch:
    attention: bool = True
    ffn: bool = True
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

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModuleSwitch":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "1M": {"dim": 128, "num_layers": 2, "num_heads": 2, "ffn_multiplier": 2, "num_memory_slots": 0, "vocab_size": 8000, "max_seq_len": 512, "modules": {"memory_bank": False}},
    "150M": {"dim": 768, "num_layers": 12, "num_heads": 12, "ffn_multiplier": 4, "num_memory_slots": 32, "vocab_size": 32000, "max_seq_len": 2048, "modules": {"memory_bank": True, "bypass_gate": True, "self_verifier": True}},
    "1B": {"dim": 2048, "num_layers": 24, "num_heads": 16, "ffn_multiplier": 4, "num_memory_slots": 64, "vocab_size": 32000, "max_seq_len": 4096, "modules": {"memory_bank": True, "bypass_gate": True, "self_verifier": True, "vision_encoder": True, "thinking_module": True, "tool_caller": True}},
    "7B": {"dim": 4096, "num_layers": 32, "num_heads": 32, "ffn_multiplier": 4, "num_memory_slots": 128, "vocab_size": 32000, "max_seq_len": 8192, "modules": {"memory_bank": True, "bypass_gate": True, "self_verifier": True, "vision_encoder": True, "audio_encoder": True, "thinking_module": True, "tool_caller": True, "fsdp": True, "gradient_checkpointing": True}},
}

def get_model_config(preset: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    if preset not in MODEL_PRESETS: raise ValueError(f"Unknown preset: {preset}")
    config = MODEL_PRESETS[preset].copy()
    config["module_switch"] = ModuleSwitch.from_dict(config.pop("modules", {}))
    if overrides: config.update(overrides)
    return config
