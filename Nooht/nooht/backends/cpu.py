from __future__ import annotations

import torch

from nooht.backends.base import NoohtBackend
from nooht.backends.registry import register_backend


@register_backend
class CpuBackend(NoohtBackend):
    """CPU 参考后端：仅用于单元测试与 CI 兜底，不属于交付的 4 个目标后端。"""

    name = "cpu"
    priority = 999

    def is_available(self) -> bool:
        return True

    def default_device(self) -> torch.device:
        return torch.device("cpu")