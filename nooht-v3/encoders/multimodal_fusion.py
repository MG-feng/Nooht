import torch
import torch.nn as nn

class MultimodalFusion(nn.Module):
    def __init__(self, hidden_dim, fusion_mode="gated"):
        super().__init__()
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

    def forward(self, text_hidden, vision_hidden=None, audio_hidden=None):
        B, S_text, D = text_hidden.shape
        combined = text_hidden + self.modality_embed(torch.zeros(B, S_text, dtype=torch.long, device=text_hidden.device))

        if vision_hidden is not None:
            vision_hidden = vision_hidden + self.modality_embed(torch.ones(B, vision_hidden.size(1), dtype=torch.long, device=vision_hidden.device))
            if self.fusion_mode == "gated":
                gate_x = torch.sigmoid(self.text_gate)
                gate_y = torch.sigmoid(self.vision_gate)
                min_len = min(combined.size(1), vision_hidden.size(1))
                fused = self.gate_norm(gate_x * combined[:, :min_len] + gate_y * vision_hidden[:, :min_len])
                if combined.size(1) > min_len: fused = torch.cat([fused, combined[:, min_len:]], dim=1)
                combined = fused
            else:
                combined = torch.cat([combined, vision_hidden], dim=1)

        if audio_hidden is not None:
            audio_hidden = audio_hidden + self.modality_embed(torch.full((B, audio_hidden.size(1)), 2, dtype=torch.long, device=audio_hidden.device))
            combined = torch.cat([combined, audio_hidden], dim=1)

        return combined
