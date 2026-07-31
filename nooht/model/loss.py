import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

def compute_bypass_load_balancing_loss(bypass_gates: List[torch.Tensor], target_budgets: torch.Tensor) -> torch.Tensor:
    total_loss = 0.0
    for gate in bypass_gates:
        batch_activation_rate = gate.mean(dim=1, keepdim=True)
        total_loss += ((batch_activation_rate - target_budgets) ** 2).mean()
    return total_loss / len(bypass_gates)

class PerformanceDrivenScheduler:
    def __init__(self, init_weights: Dict[str, float], max_weights: Dict[str, float]):
        self.weights = init_weights.copy(); self.max_weights = max_weights
        self.min_weights = {k: v * 0.1 for k, v in init_weights.items()}
        self.ema_metrics = {"ppl": 100.0, "mem_recall": 0.0, "reason_acc": 0.0}; self.ema_decay = 0.9
    def update_weights(self, metrics: Dict[str, float]) -> Dict[str, float]:
        for k in self.ema_metrics:
            self.ema_metrics[k] = self.ema_decay * self.ema_metrics[k] + (1 - self.ema_decay) * metrics.get(k, 0.0)
        if self.ema_metrics["mem_recall"] < self.ema_metrics["mem_recall"] * 0.98:
            self.weights["mem"] = min(self.weights["mem"] * 1.05, self.max_weights["mem"])
        else: self.weights["mem"] = max(self.weights["mem"] * 0.99, self.min_weights["mem"])
        if self.ema_metrics["reason_acc"] <= self.ema_metrics["reason_acc"] * 0.99:
            self.weights["verify"] = min(self.weights["verify"] * 1.05, self.max_weights["verify"])
        else: self.weights["verify"] = max(self.weights["verify"] * 0.99, self.min_weights["verify"])
        return self.weights

class NoohtLoss(nn.Module):
    def __init__(self, nce_temperature: float = 0.07):
        super().__init__(); self.lm_criterion = nn.CrossEntropyLoss(); self.nce_temperature = nce_temperature
    def forward(self, logits, labels, verify_clean, verify_noisy, bypass_gates, budget_targets, weights):
        lm_loss = self.lm_criterion(logits, labels)
        B, S, D = verify_clean.shape
        clean_flat = F.normalize(verify_clean.reshape(B*S, D), dim=-1)
        noisy_flat = F.normalize(verify_noisy.reshape(B*S, D), dim=-1)
        pos_sim = torch.matmul(clean_flat, clean_flat.T) / self.nce_temperature
        neg_sim = torch.matmul(clean_flat, noisy_flat.T) / self.nce_temperature
        logits_nce = torch.cat([pos_sim, neg_sim], dim=-1)
        labels_nce = torch.arange(B*S, device=clean_flat.device)
        verify_loss = F.cross_entropy(logits_nce, labels_nce)
        bypass_loss = compute_bypass_load_balancing_loss(bypass_gates, budget_targets)
        total_loss = weights["lm"] * lm_loss + weights["verify"] * verify_loss + weights["bypass"] * bypass_loss
        return {"total": total_loss, "lm": lm_loss, "verify": verify_loss, "bypass": bypass_loss}
