from __future__ import annotations

import torch

from nooht.backends.base import NoohtBackend
from nooht.backends.registry import register_backend


@register_backend
class OneapiBackend(NoohtBackend):
    """Intel oneAPI 后端（XPU），可选配 Intel Extension for PyTorch 提速。"""

    name = "oneapi"
    priority = 40

    def __init__(self) -> None:
        self._setup_done = False
        self._ipex = None

    def is_available(self) -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    def default_device(self) -> torch.device:
        return torch.device("xpu", 0)

    def device_count(self) -> int:
        return torch.xpu.device_count() if self.is_available() else 0

    def setup(self) -> None:
        """幂等初始化；显式区分「未 setup / 无 IPEX / 有 IPEX」。"""
        if self._setup_done:
            return
        try:
            import intel_extension_for_pytorch as ipex
            self._ipex = ipex
        except Exception:
            self._ipex = None
        self._setup_done = True

    def info(self):
        i = super().info()
        if self.is_available():
            i.extra = {"setup_done": self._setup_done,
                       "ipex_installed": self._ipex is not None}
        return i