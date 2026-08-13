import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

def compute_bypass_load_balancing_loss(gates: List[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    if not gates: return torch.tensor(0.0)
    loss = sum(((g.mean(dim=1, keepdim=True) - targets)**2).mean() for g in gates)
    return loss / len(gates)

class PerformanceDrivenScheduler:
    def __init__(self, init_w, max_w):
        self.weights = init_w.copy(); self.max_w = max_w
        self.min_w = {k: v*0.1 for k,v in init_w.items()}
    def update_weights(self, metrics): return self.weights

class NoohtLoss(nn.Module):
    def __init__(self, nce_temperature: float = 0.07):
        super().__init__()
        self.lm_crit = nn.CrossEntropyLoss(ignore_index=-100)
        self.tau = nce_temperature

    def forward(self, logits, labels, v_clean, v_noisy, gates, budgets, weights):
        lm_loss = self.lm_crit(logits, labels)
        
        # V2 FIX: 正确的 InfoNCE (clean vs noisy 交叉对比)
        B, S, D = v_clean.shape
        N = B * S
        c_flat = F.normalize(v_clean.view(N, D), dim=-1)
        n_flat = F.normalize(v_noisy.view(N, D), dim=-1)
        nce_logits = torch.matmul(c_flat, n_flat.T) / self.tau
        nce_labels = torch.arange(N, device=nce_logits.device)
        verify_loss = F.cross_entropy(nce_logits, nce_labels)
        
        bypass_loss = compute_bypass_load_balancing_loss(gates, budgets)
        total = weights.get("lm", 1.0)*lm_loss + weights.get("verify", 0.01)*verify_loss + weights.get("bypass", 0.1)*bypass_loss
        return {"total": total, "lm": lm_loss, "verify": verify_loss, "bypass": bypass_loss}
