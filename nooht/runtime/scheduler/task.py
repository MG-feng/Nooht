from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional
import uuid, time
class TaskStatus(Enum): CREATED="created"; QUEUED="queued"; ASSIGNED="assigned"; RUNNING="running"; COMPLETED="completed"; FAILED="failed"; REQUEUED="requeued"; KILLED="killed"
@dataclass
class Task:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type: str = "train"
    config: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.CREATED
    resume_step: int = 0
    assigned_worker: Optional[str] = None
    created_at: float = field(default_factory=time.time)
