import torch
import torch.nn as nn

class MelSpectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=400, hop_length=160, n_mels=80):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    def forward(self, waveform):
        B, T = waveform.shape
        n_frames = T // self.hop_length + 1
        return torch.randn(B, self.n_mels, n_frames, device=waveform.device)

class AudioConvFrontend(nn.Module):
    def __init__(self, n_mels=80, conv_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, conv_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
    def forward(self, mel):
        x = self.conv(mel)
        return x.transpose(1, 2)

class AudioEncoder(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80, encoder_dim=512, num_layers=8, num_heads=8, output_dim=768, max_audio_len=30.0):
        super().__init__()
        self.mel_spec = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        self.conv_frontend = AudioConvFrontend(n_mels, encoder_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=encoder_dim, nhead=num_heads, dim_feedforward=encoder_dim * 4, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(encoder_dim)
        self.proj = nn.Linear(encoder_dim, output_dim)

    def forward(self, waveform):
        mel = self.mel_spec(waveform)
        x = self.conv_frontend(mel)
        x = self.transformer(x)
        x = self.norm(x)
        return self.proj(x)
