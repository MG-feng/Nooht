import torch
import torch.nn as nn
from typing import AsyncIterator, Dict, Any
import logging
from .agent_core import CognitiveAgentCore
from .search_fusion import SearchFusion
from .deep_thinking import DeepThinkingModule

logger = logging.getLogger(__name__)

class NoohtCognitiveAPI(nn.Module):
    def __init__(self, dim: int, model: nn.Module):
        super().__init__(); self.dim = dim; self.model = model
        self.agent_core = CognitiveAgentCore(dim); self.search_fusion = SearchFusion(dim); self.deep_thinking = DeepThinkingModule(dim)
    async def stream_inference(self, input_stream: AsyncIterator[torch.Tensor]) -> AsyncIterator[torch.Tensor]:
        async for ktu_input in input_stream:
            hidden = ktu_input
            act_out = self.agent_core(hidden); action_idx = act_out["action_dist"].argmax(dim=-1)
            if action_idx == 0: out_hidden = self.deep_thinking(hidden, reasoning_step_func=lambda x: x + 0.01)
            elif action_idx == 1: out_hidden = hidden
            else: out_hidden = hidden
            yield out_hidden
