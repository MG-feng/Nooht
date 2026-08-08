import pytest
from nooht.runtime.config.migrator import ConfigMigrator

def test_migrate_v01_to_v10():
    """v0.1 -> v1.0 完整迁移链测试"""
    migrator = ConfigMigrator()
    old_config = {
        'version': '0.1.0',
        'gpu_workers': 4,
        'batch_size': 32
    }
    new_config = migrator.migrate(old_config)
    
    assert new_config['version'] == '1.0.0'
    assert 'workers' in new_config
    assert 'gpu_workers' not in new_config
    assert new_config['workers'] == 4
    assert 'nrt' in new_config

def test_migrate_v09_to_v10():
    """v0.9 -> v1.0 单步迁移测试"""
    migrator = ConfigMigrator()
    config = {
        'version': '0.9.0',
        'workers': 8,
        'nrt': {}
    }
    new_config = migrator.migrate(config)
    
    assert new_config['version'] == '1.0.0'
    assert 'scheduler' in new_config['nrt']

def test_no_migration_needed():
    """已是最新版本时不应修改"""
    migrator = ConfigMigrator()
    config = {'version': '1.0.0', 'workers': 2}
    result = migrator.migrate(config)
    assert result == config

def test_unknown_version_starts_from_head():
    """未知版本从头开始迁移"""
    migrator = ConfigMigrator()
    config = {'version': '0.0.1', 'custom_field': 'test'}
    new_config = migrator.migrate(config)
    assert new_config['version'] == '1.0.0'
