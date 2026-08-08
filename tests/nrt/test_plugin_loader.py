import pytest
from nooht.runtime.abi.plugin_loader import PluginLoader
from nooht.runtime.abi.plugin_base import NoohtPlugin, ResourceRequest
from typing import Dict, Any

class TestPlugin(NoohtPlugin):
    name = 'test_plugin'
    version = '1.0.0'
    def get_resource_request(self): return ResourceRequest(vram_mb=128)
    async def init(self, config): self._initialized = True; return True
    async def execute(self, inputs): return {'result': 'ok'}
    async def shutdown(self): self._shutdown = True; return True

@pytest.mark.asyncio
async def test_register_and_create():
    loader = PluginLoader()
    loader.register(TestPlugin)
    
    instance = await loader.create_instance('test_plugin', {})
    assert instance is not None
    assert instance._initialized is True

@pytest.mark.asyncio
async def test_create_unknown_plugin():
    loader = PluginLoader()
    instance = await loader.create_instance('nonexistent', {})
    assert instance is None

@pytest.mark.asyncio
async def test_shutdown_all():
    loader = PluginLoader()
    loader.register(TestPlugin)
    instance = await loader.create_instance('test_plugin', {})
    
    await loader.shutdown_all()
    assert instance._shutdown is True

@pytest.mark.asyncio
async def test_duplicate_create_returns_same_instance():
    loader = PluginLoader()
    loader.register(TestPlugin)
    
    inst1 = await loader.create_instance('test_plugin', {})
    inst2 = await loader.create_instance('test_plugin', {})
    assert inst1 is inst2
