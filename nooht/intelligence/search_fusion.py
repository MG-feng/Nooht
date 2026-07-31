import torch
import torch.nn as nn
import torch.nn.functional as F

class TruthCalibrationModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.source_embed = nn.Embedding(10, dim)
        self.mlp = nn.Sequential(nn.Linear(dim * 2, 1), nn.Sigmoid())
    def forward(self, ktu_vec: torch.Tensor, source_id: torch.Tensor) -> torch.Tensor:
        src_vec = self.source_embed(source_id); combined = torch.cat([ktu_vec, src_vec], dim=-1)
        return self.mlp(combined)

class SearchFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.truth_calibrator = TruthCalibrationModule(dim)
        self.register_buffer("l0_isolated", torch.zeros(64, dim))
    def forward(self, query_ktu: torch.Tensor, search_results: torch.Tensor, source_ids: torch.Tensor) -> torch.Tensor:
        b, n, d = search_results.shape
        truth_scores = self.truth_calibrator(search_results.view(b*n, d), source_ids.view(b*n)).view(b, n, 1)
        attn_scores = torch.matmul(query_ktu.unsqueeze(1), search_results.transpose(1, 2))
        attn_weights = F.softmax(attn_scores, dim=-1)
        weighted_context = torch.matmul(attn_weights, search_results * truth_scores)
        return weighted_context.squeeze(1)
