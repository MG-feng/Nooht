import torch
from typing import Tuple
import logging
logger = logging.getLogger(__name__)

class DeviceManager:
    def __init__(self, backend: str = "cpu"): self.backend = backend
    def get_device(self) -> torch.device:
        b_map = {"cuda":"cuda", "xla":"xla", "tpu":"xla", "cpu":"cpu", None:"cpu"}
        b_str = b_map.get(self.backend)
        if b_str is None: raise ValueError(f"Unsupported backend: {self.backend}")
        if b_str == "cuda" and not torch.cuda.is_available(): return torch.device("cpu")
        return torch.device(b_str)
    def allocate(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        dev = self.get_device()
        if dev.type == "cuda":
            free_b, _ = torch.cuda.mem_get_info(dev.index)
            req_b = torch.empty(shape, dtype=dtype).element_size() * torch.empty(shape).numel()
            if free_b < req_b: raise RuntimeError(f"OOM on {dev}: req {req_b}, avail {free_b}")
        return torch.empty(shape, dtype=dtype, device=dev)
    def free(self, tensor: torch.Tensor): del tensor
