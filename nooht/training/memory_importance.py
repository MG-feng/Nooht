import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryImportancePredictor(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__(); self.dim = dim
        self.predictor = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.threshold = nn.Parameter(torch.tensor(0.3))
    def forward(self, write_vectors: torch.Tensor) -> torch.Tensor: return self.predictor(write_vectors)
    def filter_by_importance(self, write_vectors: torch.Tensor, min_importance: float = None) -> tuple:
        scores = self.forward(write_vectors)
        threshold = min_importance if min_importance is not None else self.threshold
        mask_hard = (scores >= threshold).float()
        # V2 Fix: STE to save dead network from >= operator
        mask_ste = mask_hard + (scores - scores.detach())
        filtered = write_vectors * mask_ste
        return filtered, scores, mask_hard
