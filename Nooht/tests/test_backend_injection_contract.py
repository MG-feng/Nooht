"""vendored hook 依赖的注入/读取契约，必须被测试锁死。"""
import torch

import nooht.backends
import nooht.backends.detector as detector
from nooht.backends import backend_override, current_backend
from nooht.backends.base import NoohtBackend


class _Marker(NoohtBackend):
    name = "marker"
    priority = 0
    def is_available(self): return True
    def default_device(self): return torch.device("cpu")


def test_current_backend_reads_live_global_state():
    """current_backend() 每次调用实时读取，不缓存实例。"""
    marker = _Marker()
    with backend_override(marker):
        assert current_backend() is marker
    assert current_backend() is not marker


def test_hook_binding_is_live_function():
    """vendored hook 绑定的函数对象就是 detector.current_backend 本身。"""
    assert nooht.backends.current_backend is detector.current_backend


def test_backend_override_restores_on_exit():
    before = detector._current_backend
    with backend_override(_Marker()):
        pass
    assert detector._current_backend is before


def test_nested_override():
    outer, inner = _Marker(), _Marker()
    with backend_override(outer):
        assert current_backend() is outer
        with backend_override(inner):
            assert current_backend() is inner
        assert current_backend() is outer