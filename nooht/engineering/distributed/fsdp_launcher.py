"""FSDP 分布式启动器与异构硬件调度"""
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
import os

def setup_heterogeneous_cluster():
    """初始化支持 GPU/TPU 混合调度的分布式环境"""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        # 异构探测：如果是 TPU 节点，走 XLA 后端；如果是 GPU，走 CUDA
        if os.environ.get("TPU_WORKER", "0") == "1":
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        else:
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        return rank, device
    return 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")

def wrap_fsdp(model, device):
    bf16_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    return FSDP(model, mixed_precision=bf16_policy, device_id=device)
