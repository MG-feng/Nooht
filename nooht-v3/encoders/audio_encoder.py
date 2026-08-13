import torch
import torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self, encoder_dim=512, num_layers=8, num_heads=8, output_dim=768, n_mels=80):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, encoder_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=encoder_dim, nhead=num_heads, dim_feedforward=encoder_dim * 4, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(encoder_dim)
        self.proj = nn.Linear(encoder_dim, output_dim)

    def forward(self, mel):
        x = self.conv(mel).transpose(1, 2)
        x = self.transformer(x)
        x = self.norm(x)
        return self.proj(x)
