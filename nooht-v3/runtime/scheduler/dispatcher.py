import asyncio
from typing import Dict, Optional
from dataclasses import dataclass
import logging
from .priority_queue import MultiLevelPriorityQueue
logger = logging.getLogger(__name__)

@dataclass
class WorkerHandle:
    worker_id: str; hostname: str; status: str = "online"; current_task_id: Optional[str] = None

class TaskDispatcher:
    def __init__(self, queue: MultiLevelPriorityQueue):
        self._queue = queue; self._workers = {}
    def register_worker(self, worker_id: str, hostname: str) -> WorkerHandle:
        handle = WorkerHandle(worker_id=worker_id, hostname=hostname)
        self._workers[worker_id] = handle; return handle
    def get_available_worker(self) -> Optional[WorkerHandle]:
        for w in self._workers.values():
            if w.status == "online" and not w.current_task_id: return w
        return None
