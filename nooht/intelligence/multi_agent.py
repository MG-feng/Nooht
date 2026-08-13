import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List

class PersonalAgentMemory(nn.Module):
    def __init__(self, dim: int, local_size: int = 128):
        super().__init__(); self.local_memory = nn.Parameter(torch.randn(local_size, dim) * 0.02); self.global_memory_ref = None
    def set_global_ref(self, global_mem: torch.Tensor): self.global_memory_ref = global_mem
    def forward(self, query: torch.Tensor) -> torch.Tensor:
        local_scores = torch.matmul(query, self.local_memory.T)
        local_weights = F.softmax(local_scores, dim=-1); local_out = torch.matmul(local_weights, self.local_memory)
        if self.global_memory_ref is not None:
            global_scores = torch.matmul(query, self.global_memory_ref.T)
            global_weights = F.softmax(global_scores, dim=-1); global_out = torch.matmul(global_weights, self.global_memory_ref)
            return local_out + global_out
        return local_out

class MultiAgentTopology:
    def __init__(self, agent_rank: int, world_size: int):
        self.rank = agent_rank; self.world_size = world_size; self.dist_initialized = dist.is_initialized()
    @torch.no_grad()
    def broadcast_ktu(self, ktu_tensor: torch.Tensor) -> List[torch.Tensor]:
        if not self.dist_initialized: return [ktu_tensor]
        gather_list = [torch.zeros_like(ktu_tensor) for _ in range(self.world_size)]
        dist.all_gather(gather_list, ktu_tensor); return gather_list
