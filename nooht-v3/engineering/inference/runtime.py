import torch
import torch.nn.functional as F
from typing import AsyncIterator, List, Optional

def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model

class NoohtInferenceRuntime:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self._device = next(model.parameters()).device
        unwrapped = _unwrap_model(model)
        self._max_seq_len = unwrapped.blocks[0].rope.max_seq_len if hasattr(unwrapped, "blocks") else 2048

    @torch.no_grad()
    def generate_sync(self, input_ids, max_tokens=128, temperature=1.0, top_k=50):
        generated = input_ids.to(self._device)
        for _ in range(max_tokens):
            ctx = generated[:, -self._max_seq_len:]
            logits = self.model(ctx)["logits"][:, -1, :]
            if temperature != 1.0: logits = logits / temperature
            if top_k > 0:
                top_k_vals = torch.topk(logits, top_k, dim=-1)[0]
                logits = logits.masked_fill(logits < top_k_vals[..., -1, None], float("-inf"))
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated
