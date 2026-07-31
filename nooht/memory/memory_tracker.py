import torch
from typing import Any

class MemoryFootprintTracker:
    @staticmethod
    def estimate_bytes(data: Any) -> int:
        if isinstance(data, torch.Tensor): return data.numel() * data.element_size()
        elif isinstance(data, str): return len(data.encode("utf-8"))
        elif isinstance(data, dict): return sum(MemoryFootprintTracker.estimate_bytes(v) for v in data.values())
        elif isinstance(data, list): return sum(MemoryFootprintTracker.estimate_bytes(v) for v in data)
        return 0
