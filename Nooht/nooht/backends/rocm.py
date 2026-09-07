from __future__ import annotations

import torch

from nooht.backends.base import NoohtBackend
from nooht.backends.registry import register_backend


@register_backend
class RocmBackend(NoohtBackend):
    """AMD ROCm (HIP) 后端，优先级仅次于 CUDA。

    HIP 与 CUDA 对齐度高：本阶段 kernel 路径复用 PyTorch CUDA API
    （ROCm 构建下底层已是 HIP 编译产物）；后续阶段在此接入深度优化。
    """

    name = "rocm"
    priority = 20

    def is_available(self) -> bool:
        return torch.cuda.is_available() and torch.version.hip is not None

    def default_device(self) -> torch.device:
        return torch.device("cuda", 0)  # ROCm 同样经 torch.cuda API 暴露

    def device_count(self) -> int:
        return torch.cuda.device_count() if self.is_available() else 0

    def info(self):
        i = super().info()
        if self.is_available():
            i.extra = {"hip_version": torch.version.hip,
                       "device_name": torch.cuda.get_device_name(0)}
        return i