import asyncio
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Callable, Any
import logging
logger = logging.getLogger(__name__)

class TaskState(Enum): CREATED="created"; QUEUED="queued"; ASSIGNED="assigned"; RUNNING="running"; COMPLETED="completed"; FAILED="failed"; REQUEUED="requeued"; KILLED="killed"
class TaskEvent(Enum): ENQUEUE="enqueue"; ASSIGN="assign"; START="start"; COMPLETE="complete"; FAIL="fail"; WORKER_LOST="worker_lost"; KILL="kill"

@dataclass
class TransitionRule:
    target: TaskState
    guard: Optional[Callable[[Any], bool]] = None

class IllegalTransitionError(ValueError): pass
class GuardConditionFailedError(ValueError): pass

TRANSITIONS: Dict[Tuple[TaskState, TaskEvent], TransitionRule] = {
    (TaskState.CREATED, TaskEvent.ENQUEUE): TransitionRule(TaskState.QUEUED),
    (TaskState.QUEUED, TaskEvent.ASSIGN): TransitionRule(TaskState.ASSIGNED),
    (TaskState.ASSIGNED, TaskEvent.START): TransitionRule(TaskState.RUNNING, guard=lambda ctx: ctx is not None),
    (TaskState.RUNNING, TaskEvent.COMPLETE): TransitionRule(TaskState.COMPLETED, guard=lambda ctx: ctx is not None and getattr(ctx, 'result', None) is not None),
    (TaskState.RUNNING, TaskEvent.FAIL): TransitionRule(TaskState.FAILED),
    (TaskState.RUNNING, TaskEvent.WORKER_LOST): TransitionRule(TaskState.REQUEUED),
    (TaskState.REQUEUED, TaskEvent.ENQUEUE): TransitionRule(TaskState.QUEUED),
    (TaskState.QUEUED, TaskEvent.KILL): TransitionRule(TaskState.KILLED),
    (TaskState.ASSIGNED, TaskEvent.KILL): TransitionRule(TaskState.KILLED),
    (TaskState.RUNNING, TaskEvent.KILL): TransitionRule(TaskState.KILLED),
}

class TaskFSM:
    def __init__(self, task_id: str, initial_state: TaskState = TaskState.CREATED):
        self.task_id = task_id; self.state = initial_state; self._lock = asyncio.Lock()
    async def transition(self, event: TaskEvent, context: Any = None) -> TaskState:
        async with self._lock:
            key = (self.state, event)
            if key not in TRANSITIONS: raise IllegalTransitionError(f"Illegal: {self.state}+{event}")
            rule = TRANSITIONS[key]
            if rule.guard and not rule.guard(context): raise GuardConditionFailedError(f"Guard failed: {self.state}+{event}")
            self.state = rule.target; return self.state
    def is_terminal(self) -> bool: return self.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.KILLED)
