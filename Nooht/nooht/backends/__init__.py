from nooht.backends.base import KERNEL_NAMES, BackendInfo, NoohtBackend
# 副作用导入：触发各后端注册
from nooht.backends import cpu, cuda, oneapi, rocm, xla  # noqa: F401
from nooht.backends.detector import (backend_override, current_backend,
                                     detect_backend, init_backend,
                                     list_backends, set_backend)
from nooht.backends.registry import all_registered_backends, get_registered_backend

__all__ = [
    "KERNEL_NAMES", "BackendInfo", "NoohtBackend",
    "backend_override", "current_backend", "detect_backend", "init_backend",
    "list_backends", "set_backend",
    "all_registered_backends", "get_registered_backend",
]