import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from .ngw_block import NGWBlock, RMSNorm

class NGWModel(nn.Module):
    def __init__(self, vocab_size: int = 50000, dim: int = 768, num_layers: int = 12, num_heads: int = 12, num_memory_slots: int = 64, ffn_multiplier: int = 4, max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__(); self.vocab_size = vocab_size; self.dim = dim; self.num_layers = num_layers
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([NGWBlock(dim, num_heads, num_memory_slots, ffn_multiplier, dropout) for _ in range(num_layers)])
        self.final_norm = RMSNorm(dim); self.output_proj = nn.Linear(dim, vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight; self._init_weights()
        self.budget_predictor = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1), nn.Sigmoid())
    def _init_weights(self):
        std = 0.02
        for m in self.modules():
            if isinstance(m, nn.Linear): torch.nn.init.normal_(m.weight, mean=0.0, std=std)
            elif isinstance(m, nn.Embedding): torch.nn.init.normal_(m.weight, mean=0.0, std=std)
    def get_memory_banks(self) -> List[Any]: return [blk.memory_bank for blk in self.blocks]
    def forward(self, input_ids: torch.Tensor, noise_std: float = 0.01, return_bypass_gates: bool = False, return_memory_writes: bool = False, return_budget_targets: bool = False) -> Dict[str, torch.Tensor]:
        batch, seq = input_ids.shape
        x = self.token_embedding(input_ids)
        bypass_gates, memory_writes = [], []
        for block in self.blocks:
            outputs = block(x, return_memory_weights=False)
            x = outputs[0]; verify_logits = outputs[1]; write_intent = outputs[2]; gate = outputs[3]
            bypass_gates.append(gate)
            if return_memory_writes: memory_writes.append(write_intent)
        x = self.final_norm(x)
        # V2 Fix: Dual-path Verifier for true NCE
        noise = torch.randn_like(x) * noise_std; x_noisy = x + noise
        last_verifier = self.blocks[-1].verifier
        verify_clean = last_verifier(x); verify_noisy = last_verifier(x_noisy)
        logits = self.output_proj(x)
        result = {"logits": logits, "hidden_states": x, "verify_clean": verify_clean, "verify_noisy": verify_noisy}
        if return_bypass_gates: result["bypass_gates"] = bypass_gates
        if return_memory_writes: result["memory_writes"] = memory_writes
        if return_budget_targets: result["budget_targets"] = self.budget_predictor(x.mean(dim=1))
        return result
    def get_num_params(self) -> int: return sum(p.numel() for p in self.parameters())
