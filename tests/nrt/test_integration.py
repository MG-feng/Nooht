"""
Gate 1 核心集成测试：假任务全生命周期调度
必须在无 GPU 环境下通过
"""
import asyncio
import pytest
from nooht.runtime.scheduler.priority_queue import MultiLevelPriorityQueue, PriorityLevel
from nooht.runtime.scheduler.task import Task, TaskStatus
from nooht.runtime.state_machine.fsm import TaskFSM, TaskState, TaskEvent
from nooht.runtime.config.migrator import ConfigMigrator
from nooht.runtime.device.device_manager import DeviceManager
from nooht.runtime.abi.plugin_loader import PluginLoader
from nooht.runtime.abi.plugin_base import NoohtPlugin, ResourceRequest
from typing import Dict, Any

class LifecycleTestPlugin(NoohtPlugin):
    """模拟计算插件：验证 NRT 全生命周期调度"""
    name = 'lifecycle_test'
    version = '1.0.0'
    
    def get_resource_request(self) -> ResourceRequest:
        return ResourceRequest(vram_mb=0, cpu_cores=1, ram_mb=128, device_type='cpu')
    
    async def init(self, config: Dict[str, Any]) -> bool:
        self._step = config.get('start_step', 0)
        self._max_steps = config.get('max_steps', 5)
        self._checkpoint_steps = []
        return True
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        while self._step < self._max_steps:
            await asyncio.sleep(0.01)
            self._step += 1
            if self._step % 2 == 0:
                self._checkpoint_steps.append(self._step)
        return {'final_step': self._step, 'checkpoints': self._checkpoint_steps}
    
    async def shutdown(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_gate1_full_lifecycle():
    """Gate 1 测试 1: 假任务全生命周期调度"""
    queue = MultiLevelPriorityQueue()
    task = Task(task_id='gate1_001', task_type='train', config={'max_steps': 5})
    fsm = TaskFSM(task.task_id)
    assert fsm.state == TaskState.CREATED
    
    await queue.put(task, PriorityLevel.TRAIN)
    await fsm.transition(TaskEvent.ENQUEUE)
    assert fsm.state == TaskState.QUEUED
    
    item = await queue.get(timeout=1.0)
    assert item is not None
    assert item.task_id == 'gate1_001'
    await fsm.transition(TaskEvent.ASSIGN)
    assert fsm.state == TaskState.ASSIGNED
    
    plugin = LifecycleTestPlugin()
    await plugin.init(task.config)
    await fsm.transition(TaskEvent.START, context=task)
    assert fsm.state == TaskState.RUNNING
    
    result = await plugin.execute({})
    assert result['final_step'] == 5
    
    task.status = TaskStatus.COMPLETED
    await fsm.transition(TaskEvent.COMPLETE, context=task)
    assert fsm.state == TaskState.COMPLETED
    assert fsm.is_terminal()
    
    await plugin.shutdown()


@pytest.mark.asyncio
async def test_gate1_worker_lost_and_requeue():
    """Gate 1 测试 2: Worker 掉线重排队"""
    queue = MultiLevelPriorityQueue()
    task = Task(task_id='gate1_002', task_type='train')
    fsm = TaskFSM(task.task_id)
    
    await fsm.transition(TaskEvent.ENQUEUE)
    await queue.put(task, PriorityLevel.TRAIN)
    await fsm.transition(TaskEvent.ASSIGN)
    await fsm.transition(TaskEvent.START, context=task)
    
    await fsm.transition(TaskEvent.WORKER_LOST)
    assert fsm.state == TaskState.REQUEUED
    
    task.status = TaskStatus.REQUEUED
    await queue.put(task, PriorityLevel.TRAIN)
    await fsm.transition(TaskEvent.ENQUEUE)
    assert fsm.state == TaskState.QUEUED
    
    item = await queue.get(timeout=1.0)
    assert item is not None
    assert item.task_id == 'gate1_002'


@pytest.mark.asyncio
async def test_gate1_kill_switch_preemption():
    """Gate 1 测试 3: Kill Switch 优先级抢占"""
    queue = MultiLevelPriorityQueue()
    
    train_task = Task(task_id='train_001', task_type='train')
    await queue.put(train_task, PriorityLevel.TRAIN)
    
    kill_task = Task(task_id='kill_001', task_type='kill')
    await queue.put(kill_task, PriorityLevel.KILL_SWITCH)
    
    item = await queue.get(timeout=1.0)
    assert item.task_id == 'kill_001'


def test_gate1_config_migration():
    """Gate 1 测试 4: v0.1 配置无损迁移至 v1.0"""
    migrator = ConfigMigrator()
    old_config = {
        'version': '0.1.0',
        'gpu_workers': 4,
        'batch_size': 32,
        'learning_rate': 1e-4
    }
    new_config = migrator.migrate(old_config)
    
    assert new_config['version'] == '1.0.0'
    assert new_config['workers'] == 4
    assert new_config['batch_size'] == 32
    assert new_config['learning_rate'] == 1e-4
    assert 'nrt' in new_config


def test_gate1_device_fallback():
    """Gate 1 测试 5: 无 GPU 环境优雅回退"""
    dm = DeviceManager()
    devices = dm.get_devices()
    
    assert len(devices) >= 1
    assert devices[0].device_type == 'cpu'
    
    best = dm.get_best_device('auto')
    assert best is not None
    
    if not any(d.device_type == 'cuda' for d in devices):
        assert best.device_type == 'cpu'


@pytest.mark.asyncio
async def test_gate1_plugin_lifecycle():
    """Gate 1 测试 6: Plugin 完整生命周期"""
    loader = PluginLoader()
    loader.register(LifecycleTestPlugin)
    
    instance = await loader.create_instance('lifecycle_test', {'max_steps': 3})
    assert instance is not None
    
    result = await instance.execute({})
    assert result['final_step'] == 3
    
    await loader.shutdown_all()
