import torch
from typing import Dict, Any
from .device_manager import DeviceManager
class TensorManager:
    def __init__(self, dm: DeviceManager): self.dm = dm; self._reg = {}; self._bytes = 0
    def create(self, name: str, shape: tuple, dtype: torch.dtype):
        t = self.dm.allocate(shape, dtype); self._reg[name] = t; self._bytes += t.numel() * t.element_size(); return t
    def release(self, name: str):
        if name in self._reg:
            t = self._reg[name]; self._bytes -= t.numel() * t.element_size(); self.dm.free(t); del self._reg[name]
    def clear_cache(self):
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        import gc; gc.collect()
