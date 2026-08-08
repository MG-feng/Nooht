"""
Mock IO Plugin - 模拟 Checkpoint 写盘
用于验证 NRT 对 IO 密集型任务的调度能力
"""
import asyncio
import os
import tempfile
from typing import Dict, Any
from nooht.runtime.abi.plugin_base import NoohtPlugin, ResourceRequest

class DummyIOPlugin(NoohtPlugin):
    name = "dummy_io"
    version = "1.0.0"
    
    def get_resource_request(self) -> ResourceRequest:
        return ResourceRequest(vram_mb=0, cpu_cores=1, ram_mb=256, device_type="cpu")
    
    async def init(self, config: Dict[str, Any]) -> bool:
        self._tmp_dir = config.get("tmp_dir", tempfile.mkdtemp())
        self._file_count = config.get("file_count", 5)
        return True
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """模拟写盘：创建 N 个临时文件"""
        files_written = []
        for i in range(self._file_count):
            path = os.path.join(self._tmp_dir, f"ckpt_{i}.tmp")
            with open(path, 'w') as f:
                f.write(f"mock checkpoint data {i}")
            files_written.append(path)
            await asyncio.sleep(0.05)  # 模拟 IO 延迟
        return {'files_written': files_written, 'count': len(files_written)}
    
    async def shutdown(self) -> bool:
        return True
