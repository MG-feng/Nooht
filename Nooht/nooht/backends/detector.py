"""运行时自动检测 + 全局后端状态 + 正式注入接口。

优先级：CUDA > ROCm > XLA > oneAPI >（兜底）CPU 参考。
手动覆盖：detect_backend(prefer=...) 或环境变量 NOOHT_BACKEND。
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import List, Optional

from nooht.backends.base import NoohtBackend
from nooht.backends.registry import all_registered_backends, get_registered_backend

logger = logging.getLogger("nooht.backends")

_current_backend: Optional[NoohtBackend] = None


def list_backends(only_available: bool = True) -> List[NoohtBackend]:
    backends = sorted(all_registered_backends().values(), key=lambda b: b.priority)
    if only_available:
        backends = [b for b in backends if b.is_available()]
    return backends


def detect_backend(prefer: Optional[str] = None) -> NoohtBackend:
    """按优先级检测并返回后端实例（不修改全局状态）。"""
    prefer = prefer or os.environ.get("NOOHT_BACKEND")
    if prefer:
        backend = get_registered_backend(prefer)
        if not backend.is_available():
            raise RuntimeError(
                f"specified backend {prefer!r} is not available in this environment")
        return backend

    for backend in list_backends(only_available=True):
        if backend.name == "cpu":
            continue  # CPU 仅作最终兜底，不参与自动优选
        return backend

    logger.warning("no accelerator backend detected; falling back to CPU reference backend")
    return get_registered_backend("cpu")


def init_backend(prefer: Optional[str] = None, force: bool = False) -> NoohtBackend:
    """初始化全局后端（幂等）。"""
    global _current_backend
    if _current_backend is not None and not force:
        return _current_backend
    backend = detect_backend(prefer)
    backend.setup()
    _current_backend = backend
    logger.info("Nooht backend selected: %s (%s)", backend.name, backend.default_device())
    return backend


def current_backend() -> NoohtBackend:
    """当前后端。

    契约（vendored hook 依赖）：每次调用实时读取全局状态，不缓存后端实例；
    初始化后为 O(1) 读，热路径无探测/初始化逻辑。
    """
    if _current_backend is None:
        return init_backend()
    return _current_backend


def set_backend(backend: NoohtBackend) -> None:
    """显式设定当前后端（不校验可用性，调用方负责）。"""
    global _current_backend
    _current_backend = backend


@contextmanager
def backend_override(backend: NoohtBackend):
    """当前后端的临时注入接口（测试/验证专用）。

    ⚠ 限制：状态为进程级 global，不提供线程隔离。不得用于并发请求间的
    运行时后端选择；并行测试需避免多线程同时 override 互相踩状态。

    契约（vendored hook 依赖的唯一读取链）：
      vendored hook → nooht.backends.current_backend → detector.current_backend
      → 每次调用实时读取全局状态（无实例缓存）。
    注入立即生效，退出必然恢复（含异常路径，支持嵌套）。
    """
    global _current_backend
    prev = _current_backend
    _current_backend = backend
    try:
        yield backend
    finally:
        _current_backend = prev