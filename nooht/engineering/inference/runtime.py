import torch
from typing import AsyncIterator, List

class NoohtInferenceRuntime:
    def __init__(self, model: torch.nn.Module): self.model = model
    async def generate(self, input_ids: List[int], max_tokens: int) -> AsyncIterator[int]:
        for _ in range(max_tokens): yield 0
    def quantize(self, mode: str = "int8"): pass
