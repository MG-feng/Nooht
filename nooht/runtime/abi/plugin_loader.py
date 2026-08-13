from typing import Dict, Type, Optional
import logging
from .plugin_base import NoohtPlugin
logger = logging.getLogger(__name__)

class PluginLoader:
    def __init__(self): self._registry = {}; self._instances = {}
    def register(self, plugin_class: Type[NoohtPlugin]):
        temp = plugin_class(); self._registry[temp.name] = plugin_class
    async def create_instance(self, name: str, config: Dict) -> Optional[NoohtPlugin]:
        if name not in self._registry: return None
        if name in self._instances: return self._instances[name]
        instance = self._registry[name]()
        if await instance.init(config): self._instances[name] = instance; return instance
        return None
    async def shutdown_all(self):
        for i in self._instances.values(): await i.shutdown()
        self._instances.clear()
