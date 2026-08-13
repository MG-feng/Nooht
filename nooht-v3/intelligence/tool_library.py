import torch
import torch.nn as nn
import torch.nn.functional as F

class ToolEmbeddingLibrary(nn.Module):
    def __init__(self, dim: int, max_tools: int = 1024):
        super().__init__(); self.dim = dim
        self.tool_embeddings = nn.Parameter(torch.randn(max_tools, dim) * 0.02)
        self.tool_mask = nn.Parameter(torch.zeros(max_tools), requires_grad=False)
    def add_tool(self, tool_id: int): self.tool_mask.data[tool_id] = 1.0
    def forward(self, query: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(query, self.tool_embeddings.T)
        scores = scores.masked_fill(self.tool_mask.unsqueeze(0) == 0, float("-inf"))
        return F.softmax(scores, dim=-1)
