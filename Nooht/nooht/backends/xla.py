from __future__ import annotations

import torch

from nooht.backends.base import NoohtBackend
from nooht.backends.registry import register_backend


@register_backend
class XlaBackend(NoohtBackend):
    """PyTorch/XLA 后端（TPU / XLA-GPU / XLA-CPU）。

    可用性语义：
    - installed(): torch_xla 可导入 = 已安装；
    - is_available(): 已安装 且 runtime 明确声明 XLA 设备类型 = 可用；
      detector 只选择"可用"的后端。无 runtime API 的旧版 torch_xla
      is_available()=False，请升级 torch_xla（不用裸 except 模糊 installed/usable）。
    """

    name = "xla"
    priority = 30

    def __init__(self) -> None:
        self._device = None

    def installed(self) -> bool:
        try:
            import torch_xla  # noqa: F401
            return True
        except Exception:
            return False

    def is_available(self) -> bool:
        """轻量无副作用探测：不初始化设备。"""
        if not self.installed():
            return False
        try:
            from torch_xla import runtime as xr
            return xr.device_type() is not None
        except Exception:
            return False

    def setup(self) -> None:
        if self._device is None:
            import torch_xla.core.xla_model as xm
            self._device = xm.xla_device()

    def default_device(self) -> torch.device:
        if self._device is None:
            self.setup()
        return self._device

    def synchronize(self) -> None:
        import torch_xla.core.xla_model as xm
        xm.wait_device_ops()

    def info(self):
        i = super().info()
        i.extra = {"installed": self.installed(), "available": self.is_available()}
        if self.is_available():
            import torch_xla
            i.extra["torch_xla_version"] = torch_xla.__version__
        return i