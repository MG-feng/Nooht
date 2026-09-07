"""detector 优先级与覆盖的 mock-based 测试。"""
import pytest

import nooht.backends.detector as detector
from nooht.backends import detect_backend
from nooht.backends.registry import get_registered_backend

ALL_FOUR = ("cuda", "rocm", "xla", "oneapi")


@pytest.fixture(autouse=True)
def _isolate_global_state(monkeypatch):
    monkeypatch.setattr(detector, "_current_backend", None, raising=False)
    monkeypatch.delenv("NOOHT_BACKEND", raising=False)
    yield


def _set_availability(monkeypatch, available: dict):
    for name in ALL_FOUR:
        backend = get_registered_backend(name)
        avail = bool(available.get(name, False))
        monkeypatch.setattr(type(backend), "is_available",
                            lambda self, _a=avail: _a)


SCENARIOS = [
    ({"cuda": True}, "cuda"),
    ({"rocm": True}, "rocm"),
    ({"xla": True}, "xla"),
    ({"oneapi": True}, "oneapi"),
    ({"cuda": True, "rocm": True, "xla": True, "oneapi": True}, "cuda"),
    ({"rocm": True, "xla": True, "oneapi": True}, "rocm"),
    ({}, "cpu"),   # 全部不可用 → CPU 兜底
]


@pytest.mark.parametrize("available,expected", SCENARIOS)
def test_auto_selection(monkeypatch, available, expected):
    _set_availability(monkeypatch, available)
    assert detect_backend().name == expected


def test_env_override_to_cpu(monkeypatch):
    monkeypatch.setenv("NOOHT_BACKEND", "cpu")
    assert detect_backend().name == "cpu"


def test_env_override_unavailable_raises(monkeypatch):
    _set_availability(monkeypatch, {})
    monkeypatch.setenv("NOOHT_BACKEND", "cuda")
    with pytest.raises(RuntimeError):
        detect_backend()


def test_explicit_prefer_wins_over_env(monkeypatch):
    _set_availability(monkeypatch, {})
    monkeypatch.setenv("NOOHT_BACKEND", "cuda")   # 若生效会抛异常
    assert detect_backend(prefer="cpu").name == "cpu"


def test_explicit_prefer_cuda_available(monkeypatch):
    _set_availability(monkeypatch, {"cuda": True, "rocm": True})
    assert detect_backend(prefer="cuda").name == "cuda"


def test_explicit_prefer_cuda_unavailable_raises(monkeypatch):
    _set_availability(monkeypatch, {"rocm": True})
    with pytest.raises(RuntimeError):
        detect_backend(prefer="cuda")


def test_cpu_always_available():
    assert get_registered_backend("cpu").is_available()