import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

class VisionTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))

class VisionEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, encoder_dim=768, num_layers=12, num_heads=12, output_dim=768):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, encoder_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, encoder_dim))
        self.blocks = nn.ModuleList([VisionTransformerBlock(encoder_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_dim)
        self.proj = nn.Linear(encoder_dim, output_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, images):
        B = images.shape[0]
        x = self.patch_embed(images)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self._interpolate_pos_embed(x.size(1))
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.proj(x)

    def _interpolate_pos_embed(self, target_len):
        pos = self.pos_embed
        cls_pos = pos[:, :1, :]
        patch_pos = pos[:, 1:, :]
        N = patch_pos.size(1)
        target_patches = target_len - 1
        if N == target_patches: return pos
        size = int(math.sqrt(N))
        target_size = int(math.sqrt(target_patches))
        patch_pos = patch_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(target_size, target_size), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, pos.size(-1))
        return torch.cat([cls_pos, patch_pos], dim=1)
