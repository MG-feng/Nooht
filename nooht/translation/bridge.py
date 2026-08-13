"""
底层英文计算翻译器 (Internal Translation Bridge)
核心思想：无论用户输入何种语言，先投影到【英文语义基座空间】进行 Transformer 计算，
输出时再投影回【目标语言空间】。这能让模型共享英文的强逻辑推理能力。
"""
import torch
import torch.nn as nn

class TranslationBridge(nn.Module):
    def __init__(self, hidden_dim: int, multi_lang_vocab_size: int, en_base_vocab_size: int):
        super().__init__()
        # 输入对齐层：多语言 -> 英文基座语义
        self.input_projector = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # 输出解对齐层：英文基座语义 -> 多语言
        self.output_projector = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lang_embed = nn.Embedding(10, hidden_dim) # 支持10种语言标识

    def align_to_base(self, x: torch.Tensor, lang_id: int = 0) -> torch.Tensor:
        """将多语言隐状态对齐到英文计算基座"""
        lang_bias = self.lang_embed(torch.tensor(lang_id, device=x.device))
        return self.input_projector(x) + lang_bias

    def project_to_target(self, x: torch.Tensor) -> torch.Tensor:
        """将基座计算结果映射回多语言输出空间"""
        return self.output_projector(x)
