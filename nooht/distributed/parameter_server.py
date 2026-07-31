import asyncio
from typing import Dict, Any
from dataclasses import dataclass
import torch
import logging
logger = logging.getLogger(__name__)

@dataclass
class GradientUpdate: worker_id: str; gradients: Dict[str, torch.Tensor]; step: int

class HybridParameterServer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config; self.lr = config.get("learning_rate", 1e-4); self.parameters = {}; self._grad_q = asyncio.Queue(); self._velocity = {}
    async def push(self, update: GradientUpdate): await self._grad_q.put(update)
    async def update_loop(self):
        while True:
            updates = []
            while not self._grad_q.empty(): updates.append(await self._grad_q.get())
            if not updates: await asyncio.sleep(0.01); continue
            for p_name in self.parameters.keys():
                sum_g = None; cnt = 0
                for u in updates:
                    if p_name in u.gradients:
                        g = u.gradients[p_name]
                        sum_g = g.clone() if sum_g is None else torch.add(sum_g, g)
                        cnt += 1
                if cnt > 0 and p_name in self.parameters:
                    avg = torch.div(sum_g, cnt)
                    if p_name not in self._velocity: self._velocity[p_name] = torch.zeros_like(self.parameters[p_name])
                    self._velocity[p_name] = 0.9 * self._velocity[p_name] - self.lr * avg
                    self.parameters[p_name] = torch.add(self.parameters[p_name], self._velocity[p_name])
