import asyncio
import torch
from typing import Dict, Any
from nooht.runtime.abi.plugin_base import NoohtPlugin, ResourceRequest

class DataLoaderPlugin(NoohtPlugin):
    @property
    def name(self) -> str: return "data_loader"
    @property
    def version(self) -> str: return "1.0.0"
    def get_resource_request(self) -> ResourceRequest: return ResourceRequest(vram_mb=0, cpu_cores=1, ram_mb=256, device_type="cpu")
    async def init(self, config: Dict[str, Any]) -> bool: return True
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]: return {"batch": "dummy"}
    async def shutdown(self) -> bool: return True
