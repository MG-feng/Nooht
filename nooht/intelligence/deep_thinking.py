import torch
import torch.nn as nn

class ExpectedGainPredictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.mlp = nn.Sequential(nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, 1), nn.Sigmoid())
    def forward(self, current_state: torch.Tensor) -> torch.Tensor: return self.mlp(current_state)

class DeepThinkingModule(nn.Module):
    def __init__(self, dim: int, max_steps: int = 100):
        super().__init__(); self.gain_predictor = ExpectedGainPredictor(dim)
        self.max_steps = max_steps; self.energy_threshold = 0.05
    def forward(self, query: torch.Tensor, reasoning_step_func, max_steps: int = None) -> torch.Tensor:
        if max_steps is None: max_steps = self.max_steps
        state = query
        for step in range(max_steps):
            expected_gain = self.gain_predictor(state)
            new_state = reasoning_step_func(state)
            keep_mask = (expected_gain >= self.energy_threshold).float()
            state = keep_mask * new_state + (1 - keep_mask) * state
        return state
