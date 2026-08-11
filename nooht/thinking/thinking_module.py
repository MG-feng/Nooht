import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

THINK_START_ID = 32005
THINK_END_ID = 32006

@dataclass
class ThinkingConfig:
    enabled: bool = True
    max_thinking_tokens: int = 2048
    thinking_temperature: float = 0.7
    answer_temperature: float = 0.3
    thinking_depth: int = 2
    show_thinking: bool = False

class ThinkingModule(nn.Module):
    def __init__(self, hidden_dim, config: ThinkingConfig = None):
        super().__init__()
        self.config = config or ThinkingConfig()
        self.hidden_dim = hidden_dim
        self.complexity_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 4), nn.GELU(), nn.Linear(hidden_dim // 4, 1), nn.Sigmoid())
        self.depth_controller = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 4), nn.GELU(), nn.Linear(hidden_dim // 4, 3))
        self.thinking_accumulator = nn.GRUCell(hidden_dim, hidden_dim)
        self.think_to_answer = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.thinking_gate = nn.Parameter(torch.zeros(1))

    def should_think(self, hidden):
        return self.complexity_classifier(hidden)

    def get_thinking_depth(self, hidden):
        return F.softmax(self.depth_controller(hidden), dim=-1)

    def forward(self, hidden, thinking_tokens=None, force_think=False, force_no_think=False):
        B, S, D = hidden.shape
        last_hidden = hidden[:, -1, :]
        think_prob = self.should_think(last_hidden)

        if force_think:
            is_thinking = torch.ones(B, dtype=torch.bool, device=hidden.device)
        elif force_no_think:
            is_thinking = torch.zeros(B, dtype=torch.bool, device=hidden.device)
        else:
            is_thinking = (think_prob.squeeze(-1) > 0.5)

        thinking_state = torch.zeros(B, D, device=hidden.device)
        if thinking_tokens is not None:
            for t in range(thinking_tokens.size(1)):
                thinking_state = self.thinking_accumulator(thinking_tokens[:, t, :], thinking_state)
        elif is_thinking.any():
            depth_dist = self.get_thinking_depth(last_hidden)
            depth_steps = (depth_dist * torch.tensor([2, 4, 8], device=hidden.device)).sum(-1)
            max_steps = int(depth_steps.max().item())
            current = last_hidden
            for step in range(min(max_steps, 8)):
                thinking_state = self.thinking_accumulator(current, thinking_state)
                current = thinking_state

        gate = torch.sigmoid(self.thinking_gate)
        thinking_expanded = thinking_state.unsqueeze(1).expand(-1, S, -1)
        combined = torch.cat([hidden, thinking_expanded], dim=-1)
        thinking_enhanced = self.think_to_answer(combined)
        
        is_thinking_mask = is_thinking.float().unsqueeze(1).unsqueeze(2)
        output = is_thinking_mask * (gate * thinking_enhanced + (1 - gate) * hidden) + (1 - is_thinking_mask) * hidden
        return {"output": output, "think_prob": think_prob, "thinking_state": thinking_state, "is_thinking": is_thinking}
