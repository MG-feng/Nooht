from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModuleSwitch:
    # 基础
    attention: bool = True
    ffn: bool = True
    # Nooht 核心
    memory_bank: bool = False
    bypass_gate: bool = False
    self_verifier: bool = False
    # 模态
    vision_encoder: bool = False
    audio_encoder: bool = False
    file_io_handler: bool = False  # 新增：文件读写能力
    # 硬件与底层
    tpu_gpu_mixed: bool = False    # 新增：异构混合训练开关
    internal_translation: bool = False # 新增：底层英文计算翻译器
    # 高级认知
    thinking_module: bool = False
    tool_caller: bool = False
    # 扩展
    fsdp: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModuleSwitch":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})
