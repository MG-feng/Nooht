import torch
import torch.nn as nn

VISION_START_TOKEN = 32001
VISION_END_TOKEN = 32002
AUDIO_START_TOKEN = 32003
AUDIO_END_TOKEN = 32004

class MultimodalFusion(nn.Module):
    def __init__(self, hidden_dim, fusion_mode="gated"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_mode = fusion_mode
        if fusion_mode == "gated":
            self.text_gate = nn.Parameter(torch.ones(1, 1, hidden_dim))
            self.vision_gate = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.audio_gate = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.gate_norm = nn.LayerNorm(hidden_dim)
        elif fusion_mode == "cross_attn":
            self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
            self.cross_norm = nn.LayerNorm(hidden_dim)
        self.modality_embed = nn.Embedding(4, hidden_dim)

    def forward(self, text_hidden, vision_hidden=None, audio_hidden=None, vision_positions=None, audio_positions=None):
        B, S_text, D = text_hidden.shape
        text_mod = self.modality_embed(torch.zeros(B, S_text, dtype=torch.long, device=text_hidden.device))
        combined = text_hidden + text_mod

        if vision_hidden is not None:
            vis_mod = self.modality_embed(torch.ones(B, vision_hidden.size(1), dtype=torch.long, device=vision_hidden.device))
            vision_hidden = vision_hidden + vis_mod
            if self.fusion_mode == "gated":
                combined = self._gated_fuse(combined, vision_hidden)
            elif self.fusion_mode == "cross_attn":
                combined = self._cross_attn_fuse(combined, vision_hidden)
            else:
                combined = torch.cat([combined, vision_hidden], dim=1)

        if audio_hidden is not None:
            aud_mod = self.modality_embed(torch.full((B, audio_hidden.size(1)), 2, dtype=torch.long, device=audio_hidden.device))
            audio_hidden = audio_hidden + aud_mod
            if self.fusion_mode == "gated":
                combined = self._gated_fuse(combined, audio_hidden)
            elif self.fusion_mode == "cross_attn":
                combined = self._cross_attn_fuse(combined, audio_hidden)
            else:
                combined = torch.cat([combined, audio_hidden], dim=1)
        return combined

    def _gated_fuse(self, x, y):
        gate_x = torch.sigmoid(self.text_gate)
        gate_y = torch.sigmoid(self.vision_gate)
        min_len = min(x.size(1), y.size(1))
        fused = gate_x * x[:, :min_len] + gate_y * y[:, :min_len]
        fused = self.gate_norm(fused)
        if x.size(1) > min_len:
            fused = torch.cat([fused, x[:, min_len:]], dim=1)
        return fused

    def _cross_attn_fuse(self, x, y):
        attn_out, _ = self.cross_attn(x, y, y)
        return self.cross_norm(x + attn_out)
