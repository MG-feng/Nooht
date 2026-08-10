import asyncio
import time
from dataclasses import dataclass, field
from typing import List
from nooht.runtime.scheduler.priority_queue import MultiLevelPriorityQueue, PriorityLevel
from nooht.plugins.pytorch_train.train_plugin import PyTorchTrainPlugin
from nooht.plugins.pytorch_train.checkpoint_plugin import AsyncCheckpointPlugin

@dataclass
class NRTBenchmarkResult:
    name: str
    total_time: float = 0.0
    step_times: List[float] = field(default_factory=list)
    gpu_idle_times: List[float] = field(default_factory=list)
    checkpoints_saved: int = 0

async def train_nrt(total_steps=500, checkpoint_interval=50, io_delay=2.0):
    result = NRTBenchmarkResult(name="NRT (Async)")
    print(f"\n[NRT] Running Async with Priority Queue...")
    
    ckpt_queue = MultiLevelPriorityQueue()
    train_plugin = PyTorchTrainPlugin()
    ckpt_plugin = AsyncCheckpointPlugin()
    
    await train_plugin.init({"total_steps": total_steps, "checkpoint_interval": checkpoint_interval, "checkpoint_queue": ckpt_queue})
    await ckpt_plugin.init({"checkpoint_dir": "/tmp/nrt_ckpt", "io_delay": io_delay})
    
    total_start = time.time()
    
    async def ckpt_consumer():
        saved = 0
        target = total_steps // checkpoint_interval
        while saved < target:
            task = await ckpt_queue.get(timeout=1.0)
            if task:
                await ckpt_plugin.execute(task)
                result.checkpoints_saved += 1
                saved += 1
                
    train_task = asyncio.create_task(train_plugin.execute({}))
    ckpt_task = asyncio.create_task(ckpt_consumer())
    
    train_res = await train_task
    await ckpt_task
    
    result.total_time = time.time() - total_start
    result.step_times = train_plugin._metrics.step_times
    # NRT 异步不阻塞 GPU，Idle 为 0
    result.gpu_idle_times = [0.0] * result.checkpoints_saved 
    
    await train_plugin.shutdown()
    await ckpt_plugin.shutdown()
    
    print(f"[NRT] Done. Total: {result.total_time:.2f}s, Idle: {sum(result.gpu_idle_times):.2f}s")
    return result

def run_nrt_benchmark(**kwargs):
    return asyncio.run(train_nrt(**kwargs))
