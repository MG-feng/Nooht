import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from .ngw_block import NGWBlock, RMSNorm
from ..config.module_switch import ModuleSwitch, get_model_config
from ..encoders.vision_encoder import VisionEncoder
from ..encoders.audio_encoder import AudioEncoder
from ..encoders.multimodal_fusion import MultimodalFusion
from ..thinking.thinking_module import ThinkingModule, ThinkingConfig
from ..agent.tool_api import ToolAPI

class NGWModelV2(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.switch = config.get("module_switch", ModuleSwitch())
        self.dim = config.get("dim", 768)
        self.vocab_size = config.get("vocab_size", 32000)
        self.num_layers = config.get("num_layers", 12)
        self.num_heads = config.get("num_heads", 12)
        self.num_memory_slots = config.get("num_memory_slots", 64)
        self.ffn_multiplier = config.get("ffn_multiplier", 4)

        self.token_embedding = nn.Embedding(self.vocab_size + 10, self.dim)
        self.blocks = nn.ModuleList([
            NGWBlock(self.dim, self.num_heads, self.num_memory_slots if self.switch.memory_bank else 0, self.ffn_multiplier)
            for _ in range(self.num_layers)
        ])
        self.final_norm = RMSNorm(self.dim)
        self.output_proj = nn.Linear(self.dim, self.vocab_size + 10, bias=False)
        self.output_proj.weight = self.token_embedding.weight

        self.vision_encoder = VisionEncoder(output_dim=self.dim) if self.switch.vision_encoder else None
        self.audio_encoder = AudioEncoder(output_dim=self.dim) if self.switch.audio_encoder else None
        self.multimodal_fusion = MultimodalFusion(self.dim) if self.switch.multimodal_fusion else None
        self.thinking_module = ThinkingModule(self.dim) if self.switch.thinking_module else None
        self.tool_api = ToolAPI() if self.switch.tool_caller else None
        self.budget_predictor = nn.Sequential(nn.Linear(self.dim, self.dim // 2), nn.GELU(), nn.Linear(self.dim // 2, 1), nn.Sigmoid()) if self.switch.budget_predictor else None

    def forward(self, input_ids, pixel_values=None, audio_waveform=None, thinking_enabled=None):
        x = self.token_embedding(input_ids)
        if self.multimodal_fusion and (pixel_values is not None or audio_waveform is not None):
            v = self.vision_encoder(pixel_values) if pixel_values is not None else None
            a = self.audio_encoder(audio_waveform) if audio_waveform is not None else None
            x = self.multimodal_fusion(x, v, a)
        if self.thinking_module:
            x = self.thinking_module(x, force_think=thinking_enabled if thinking_enabled is not None else False)["output"]
        for block in self.blocks:
            x = block(x)[0]
        x = self.final_norm(x)
        return {"logits": self.output_proj(x), "hidden_states": x}

    @classmethod
    def from_preset(cls, preset, **overrides):
        return cls(get_model_config(preset, overrides))
