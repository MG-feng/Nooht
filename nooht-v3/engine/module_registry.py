import asyncio
from typing import Dict, Type, Optional
from .module_interface import NoohtModule
import logging
logger = logging.getLogger(__name__)

class ModuleRegistry:
    def __init__(self): self._modules = {}; self._module_types = {}; self._lock = None
    def _get_lock(self):
        if self._lock is None: self._lock = asyncio.Lock()
        return self._lock
    async def create_instance(self, module_type: str, instance_name: str, config: Dict = None) -> Optional[NoohtModule]:
        async with self._get_lock():
            if module_type not in self._module_types: return None
            if instance_name in self._modules: return self._modules[instance_name]
            cls = self._module_types[module_type]; inst = cls(name=instance_name, config=config or {})
            await inst.initialize(); self._modules[instance_name] = inst; return inst
    async def shutdown_all(self):
        async with self._get_lock():
            for m in list(self._modules.values()): await m.cleanup()
            self._modules.clear()
