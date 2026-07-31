import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"): return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x.float() / rms
        return (x_normed * self.weight).to(x.dtype)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__(); self.dim = dim; self.max_seq_len = max_seq_len
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs)
        t = torch.arange(max_seq_len).float()
        freqs_t = torch.outer(t, freqs)
        self.register_buffer("cos_cached", freqs_t.cos()); self.register_buffer("sin_cached", freqs_t.sin())
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        seq_len = x.size(2)
        cos = self.cos_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        x1 = x[..., 0::2]; x2 = x[..., 1::2]
        x_rot_even = x1 * cos - x2 * sin; x_rot_odd = x2 * cos + x1 * sin
        # V2 Fix: Out-of-place stack for torch.compile compatibility
        x_out = torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)
        return x_out.to(x.dtype)

class MemoryBank(nn.Module):
    def __init__(self, num_slots: int, dim: int):
        super().__init__(); self.num_slots = num_slots; self.dim = dim
        self.memory = nn.Parameter(torch.randn(num_slots, dim) / math.sqrt(dim))
        self.query_proj = nn.Linear(dim, dim, bias=False); self.output_proj = nn.Linear(dim, dim, bias=False)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq, dim = x.shape
        queries = self.query_proj(x)
        scores = torch.matmul(queries, self.memory.T) / math.sqrt(dim)
        attention_weights = F.softmax(scores, dim=-1)
        retrieved = torch.matmul(attention_weights, self.memory)
        output = self.output_proj(retrieved)
        write_intent = retrieved.mean(dim=1) # [batch, dim] for global sync
        return output, attention_weights, write_intent

class SelfVerifier(nn.Module):
    def __init__(self, dim: int, verify_dim: int = 128):
        super().__init__(); self.dim = dim; self.verify_dim = verify_dim
        self.verify_proj = nn.Sequential(nn.Linear(dim, dim // 2, bias=False), nn.GELU(), nn.Linear(dim // 2, verify_dim, bias=False))
        self.temperature = nn.Parameter(torch.tensor(0.07))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.verify_proj(x)

class NGWBlock(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 12, num_memory_slots: int = 64, ffn_multiplier: int = 4, dropout: float = 0.0):
        super().__init__(); self.dim = dim; self.num_heads = num_heads; self.head_dim = dim // num_heads
        self.attn_norm = RMSNorm(dim); self.q_proj = nn.Linear(dim, dim, bias=False); self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False); self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryPositionalEmbedding(self.head_dim)
        self.memory_norm = RMSNorm(dim); self.memory_bank = MemoryBank(num_memory_slots, dim)
        self.bypass_gate = nn.Linear(dim, 1, bias=True) # Soft Bypass Gate
        self.ffn_norm = RMSNorm(dim); self.ffn = nn.Sequential(nn.Linear(dim, dim * ffn_multiplier, bias=False), nn.GELU(), nn.Linear(dim * ffn_multiplier, dim, bias=False))
        self.verifier = SelfVerifier(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, return_memory_weights: bool = False) -> Tuple[torch.Tensor, ...]:
        batch, seq, dim = x.shape; extras = []
        residual = x; x_norm = self.attn_norm(x)
        q = self.q_proj(x_norm).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.rope(q); k = self.rope(k)
        dropout_p = self.dropout.p if self.training and hasattr(self.dropout, "p") else 0.0
        # V2 Fix: Embrace SDPA for FlashAttention
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, dim)
        attn_out = self.o_proj(attn_out); x = residual + self.dropout(attn_out)

        residual = x; x_norm = self.memory_norm(x)
        memory_out, memory_weights, write_intent = self.memory_bank(x_norm)
        gate = torch.sigmoid(self.bypass_gate(x_norm))
        scale_factor = 1.0 / torch.clamp(gate, min=0.1) # Forward Activation Scaling
        x = gate * (memory_out * scale_factor) + (1 - gate) * x
        x = residual + self.dropout(x)

        if return_memory_weights: extras.append(memory_weights)
        extras.append(write_intent); extras.append(gate)

        residual = x; x_norm = self.ffn_norm(x); ffn_out = self.ffn(x_norm); x = residual + self.dropout(ffn_out)
        verify_logits = self.verifier(x)
        result = (x, verify_logits)
        if extras: result += tuple(extras)
        return result
