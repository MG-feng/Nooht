import asyncio
from enum import IntEnum
from typing import Any, Optional, Dict
import logging
logger = logging.getLogger(__name__)

class PriorityLevel(IntEnum):
    KILL_SWITCH=0; CHECKPOINT=1; INFERENCE=2; EVAL=3; TRAIN=4; IO=5

class MultiLevelPriorityQueue:
    def __init__(self):
        self._queues = {l: asyncio.Queue() for l in PriorityLevel}
        self._total_size = 0
        self._cond = asyncio.Condition()
    async def put(self, item: Any, priority: PriorityLevel):
        async with self._cond:
            await self._queues[priority].put(item)
            self._total_size += 1
            self._cond.notify()
    async def get(self, timeout: float = 1.0) -> Optional[Any]:
        async with self._cond:
            deadline = asyncio.get_running_loop().time() + timeout
            while True:
                for level in PriorityLevel:
                    if not self._queues[level].empty():
                        self._total_size -= 1
                        return self._queues[level].get_nowait()
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0: return None
                try: await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                except asyncio.TimeoutError: return None
    def size(self) -> int: return self._total_size
