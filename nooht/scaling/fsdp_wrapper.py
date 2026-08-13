import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)

class FSDPWrapper:
    @staticmethod
    def wrap(model, use_bf16=True, sharding_strategy="FULL_SHARD"):
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
            mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16) if use_bf16 else None
            strat = {"FULL_SHARD": ShardingStrategy.FULL_SHARD, "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP}.get(sharding_strategy, ShardingStrategy.FULL_SHARD)
            return FSDP(model, sharding_strategy=strat, mixed_precision=mp, device_id=torch.cuda.current_device() if torch.cuda.is_available() else None)
        except: return model
