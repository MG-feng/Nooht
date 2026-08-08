import pytest
import asyncio
from nooht.runtime.scheduler.priority_queue import MultiLevelPriorityQueue, PriorityLevel

@pytest.mark.asyncio
async def test_priority_ordering():
    """高优先级任务必须先出队"""
    queue = MultiLevelPriorityQueue()
    
    await queue.put({'id': 'io_task'}, PriorityLevel.IO)
    await queue.put({'id': 'train_task'}, PriorityLevel.TRAIN)
    await queue.put({'id': 'eval_task'}, PriorityLevel.EVAL)
    await queue.put({'id': 'infer_task'}, PriorityLevel.INFERENCE)
    await queue.put({'id': 'ckpt_task'}, PriorityLevel.CHECKPOINT)
    await queue.put({'id': 'kill_task'}, PriorityLevel.KILL_SWITCH)
    
    item1 = await queue.get(timeout=0.5)
    assert item1['id'] == 'kill_task'
    
    item2 = await queue.get(timeout=0.5)
    assert item2['id'] == 'ckpt_task'
    
    item3 = await queue.get(timeout=0.5)
    assert item3['id'] == 'infer_task'

@pytest.mark.asyncio
async def test_same_priority_fifo():
    """同优先级必须 FIFO"""
    queue = MultiLevelPriorityQueue()
    
    await queue.put({'id': 'task_a'}, PriorityLevel.TRAIN)
    await queue.put({'id': 'task_b'}, PriorityLevel.TRAIN)
    await queue.put({'id': 'task_c'}, PriorityLevel.TRAIN)
    
    assert (await queue.get(timeout=0.5))['id'] == 'task_a'
    assert (await queue.get(timeout=0.5))['id'] == 'task_b'
    assert (await queue.get(timeout=0.5))['id'] == 'task_c'

@pytest.mark.asyncio
async def test_empty_queue_timeout():
    """空队列必须在 timeout 后返回 None"""
    queue = MultiLevelPriorityQueue()
    result = await queue.get(timeout=0.1)
    assert result is None

@pytest.mark.asyncio
async def test_queue_size():
    queue = MultiLevelPriorityQueue()
    assert queue.size() == 0
    
    await queue.put({'id': 't1'}, PriorityLevel.TRAIN)
    await queue.put({'id': 't2'}, PriorityLevel.EVAL)
    assert queue.size() == 2
    
    await queue.get(timeout=0.5)
    assert queue.size() == 1
