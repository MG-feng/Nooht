import torch
import torch.distributed as dist
import logging
from .memory_importance import MemoryImportancePredictor
logger = logging.getLogger(__name__)

class MemorySyncManager:
    def __init__(self, memory_param: torch.Tensor, world_size: int, ema_decay: float = 0.999, importance_predictor: MemoryImportancePredictor = None):
        self.memory = memory_param; self.world_size = world_size; self.ema_decay = ema_decay
        self.memory.requires_grad = False # Disable Autograd
        self.warmup_steps = 1000; self.current_step = 0
        self.importance_predictor = importance_predictor # IMW Integration
        logger.info("MemorySyncManager initialized (Autograd disabled, EMA mode, IMW filtering).")

    @torch.no_grad()
    def sync_and_update(self, local_write_vectors: torch.Tensor):
        self.current_step += 1
        if self.importance_predictor is not None:
            filtered_vectors, scores, mask = self.importance_predictor.filter_by_importance(local_write_vectors)
        else: filtered_vectors = local_write_vectors
        if self.current_step < self.warmup_steps:
            gumbel_noise = torch.rand_like(filtered_vectors) * 0.1
            filtered_vectors = filtered_vectors + gumbel_noise
        if not dist.is_initialized(): global_write = filtered_vectors
        else:
            global_write = filtered_vectors.clone()
            dist.all_reduce(global_write, op=dist.ReduceOp.SUM)
            global_write /= self.world_size
        self.memory.data.mul_(self.ema_decay).add_(global_write, alpha=(1 - self.ema_decay))

    def get_memory_snapshot(self) -> torch.Tensor: return self.memory.data.clone()
