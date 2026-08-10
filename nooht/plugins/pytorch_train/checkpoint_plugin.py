import asyncio
import time
import os
import torch
from typing import Dict, Any
from nooht.runtime.abi.plugin_base import NoohtPlugin, ResourceRequest

class AsyncCheckpointPlugin(NoohtPlugin):
    @property
    def name(self) -> str: return "async_checkpoint"
    @property
    def version(self) -> str: return "2.0.0" # V2 Atomic
    
    def __init__(self):
        self._ckpt_dir = "/tmp/nooht_ckpt"
        self._io_delay = 2.0
        
    def get_resource_request(self) -> ResourceRequest:
        return ResourceRequest(vram_mb=0, cpu_cores=1, ram_mb=512, device_type="cpu")
        
    async def init(self, config: Dict[str, Any]) -> bool:
        self._ckpt_dir = config.get("checkpoint_dir", "/tmp/nooht_ckpt")
        self._io_delay = config.get("io_delay", 2.0)
        os.makedirs(self._ckpt_dir, exist_ok=True)
        return True
        
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ckpt_data = inputs.get("data", {})
        step = ckpt_data.get("step", 0)
        start_time = time.time()
        
        await asyncio.sleep(self._io_delay) # 模拟 IO 延迟
        
        final_path = os.path.join(self._ckpt_dir, f"ckpt_step_{step}.pt")
        tmp_path = final_path + ".tmp"
        
        try:
            # V2 Rule 31: 原子写盘 + fsync
            with open(tmp_path, "wb") as f:
                torch.save(ckpt_data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, final_path)
        except Exception as e:
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except OSError: pass
            raise RuntimeError(f"Atomic write failed: {e}")
            
        return {"saved": True, "step": step, "io_time": time.time() - start_time}
        
    async def shutdown(self) -> bool: return True
