from __future__ import annotations

import torch

from nooht.backends.base import NoohtBackend
from nooht.backends.registry import register_backend


@register_backend
class CudaBackend(NoohtBackend):
    """NVIDIA CUDA 后端：性能对齐基准，优先级最高。"""

    name = "cuda"
    priority = 10

    def is_available(self) -> bool:
        # ROCm 构建下 torch.cuda.is_available() 也为 True，必须用 torch.version.hip 排除
        return torch.cuda.is_available() and torch.version.hip is None

    def default_device(self) -> torch.device:
        return torch.device("cuda", 0)

    def device_count(self) -> int:
        return torch.cuda.device_count() if self.is_available() else 0

    def info(self):
        i = super().info()
        if self.is_available():
            i.extra = {
                "torch_cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0),
                "capability": torch.cuda.get_device_capability(0),
            }
        return i