import asyncio, itertools
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import logging
logger = logging.getLogger(__name__)

class TaskPriority(Enum): CRITICAL = 1; HIGH = 2; MEDIUM = 3; LOW = 4
class TaskStatus(Enum): PENDING = "pending"; RUNNING = "running"; COMPLETED = "completed"; FAILED = "failed"
class DeviceType(Enum): GPU = "gpu"; TPU = "tpu"; CPU = "cpu"

@dataclass
class Task:
    task_id: str; priority: TaskPriority = TaskPriority.MEDIUM; status: TaskStatus = TaskStatus.PENDING
    payload: Dict[str, Any] = None; callback = None; timeout: int = 300; compute_units: float = 1.0

@dataclass
class DeviceInfo:
    device_id: str; device_type: DeviceType; compute_capacity: float; current_load: float = 0.0; is_available: bool = True

class HybridScheduler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config; self._tasks = {}; self._completed = {}
        self._max_completed = config.get("task_history_limit", 10000)
        self._devices = {"cpu_0": DeviceInfo("cpu_0", DeviceType.CPU, 1.0)}
        self._queue = None; self._counter = itertools.count(); self._running = False; self._workers = []

    async def start(self):
        if self._queue is None: self._queue = asyncio.PriorityQueue()
        self._running = True
        for i in range(self.config.get("num_workers", 4)): self._workers.append(asyncio.create_task(self._loop(f"w_{i}")))

    async def stop(self):
        self._running = False
        for w in self._workers: w.cancel()
        self._workers.clear()

    async def submit(self, task: Task) -> bool:
        if self._queue is None: self._queue = asyncio.PriorityQueue()
        c = next(self._counter)
        await self._queue.put((task.priority.value, c, task.task_id, task)); self._tasks[task.task_id] = task; return True

    async def _loop(self, name):
        while self._running:
            try:
                p, c, tid, t = await asyncio.wait_for(self._queue.get(), 1.0)
                await self.execute(tid); self._queue.task_done()
            except asyncio.TimeoutError: continue
            except asyncio.CancelledError: break

    async def execute(self, tid: str):
        t = self._tasks[tid]; t.status = TaskStatus.RUNNING
        avail = [d for d in self._devices.values() if d.is_available]
        if not avail: raise RuntimeError("No available devices")
        dev = min(avail, key=lambda d: d.current_load); dev.current_load += t.compute_units
        try:
            res = await asyncio.wait_for(t.callback(t.payload), timeout=t.timeout)
            t.status = TaskStatus.COMPLETED; return res
        except Exception as e:
            t.status = TaskStatus.FAILED; raise e
        finally:
            dev.current_load = max(0, dev.current_load - t.compute_units)
            if tid in self._tasks:
                t = self._tasks.pop(tid)
                if len(self._completed) >= self._max_completed: del self._completed[next(iter(self._completed))]
                self._completed[tid] = t
