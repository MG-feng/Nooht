import torch
from typing import Any

class CompatibilityLayer:
    @staticmethod
    def compile(model: torch.nn.Module, backend: str = "inductor") -> torch.nn.Module:
        if backend == "inductor": return torch.compile(model, mode="reduce-overhead")
        return model
    @staticmethod
    def apply_fsdp(model: torch.nn.Module) -> torch.nn.Module: return model
