import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .tool_library import ToolEmbeddingLibrary

class ActionEmbeddingSpace(nn.Module):
    def __init__(self, dim: int, num_base_actions: int = 6):
        super().__init__(); self.action_prototypes = nn.Parameter(torch.randn(num_base_actions, dim) * 0.02)
    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        scores = torch.matmul(hidden, self.action_prototypes.T)
        dist = F.softmax(scores, dim=-1)
        action_emb = torch.matmul(dist, self.action_prototypes)
        return {"dist": dist, "embedding": action_emb}

class CognitiveAgentCore(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.action_space = ActionEmbeddingSpace(dim)
        self.tool_library = ToolEmbeddingLibrary(dim); self.arg_predictor = nn.Linear(dim, dim)
    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        act_out = self.action_space(hidden); tool_dist = self.tool_library(hidden); args = self.arg_predictor(hidden)
        return {"action_dist": act_out["dist"], "action_embedding": act_out["embedding"], "tool_dist": tool_dist, "action_args": args}
