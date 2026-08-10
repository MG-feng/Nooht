import asyncio
import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from nooht.runtime.abi.plugin_base import NoohtPlugin, ResourceRequest

@dataclass
class TrainingMetrics:
    step_times: list = field(default_factory=list)
    total_compute_time: float = 0.0
    steps_completed: int = 0

class PyTorchTrainPlugin(NoohtPlugin):
    @property
    def name(self) -> str: return "pytorch_train"
    @property
    def version(self) -> str: return "1.0.0"
    
    def __init__(self):
        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._device: Optional[torch.device] = None
        self._metrics = TrainingMetrics()
        self._current_step = 0
        self._checkpoint_queue: Optional[asyncio.Queue] = None
        self._total_steps = 500
        self._checkpoint_interval = 50
        self._vocab_size = 30522
        self._seq_len = 128
        self._batch_size = 16

    def get_resource_request(self) -> ResourceRequest:
        return ResourceRequest(vram_mb=2048, cpu_cores=2, ram_mb=1024, device_type="cuda")

    async def init(self, config: Dict[str, Any]) -> bool:
        self._total_steps = config.get("total_steps", 500)
        self._checkpoint_interval = config.get("checkpoint_interval", 50)
        self._checkpoint_queue = config.get("checkpoint_queue")
        self._vocab_size = config.get("vocab_size", 30522)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # TinyBERT
        class TinyBERT(nn.Module):
            def __init__(self, v_s, dim, n_l, n_h):
                super().__init__()
                self.embedding = nn.Embedding(v_s, dim)
                self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=dim, nhead=n_h, dim_feedforward=dim*4, batch_first=True) for _ in range(n_l)])
                self.lm_head = nn.Linear(dim, v_s)
            def forward(self, x):
                x = self.embedding(x)
                for l in self.layers: x = l(x)
                return self.lm_head(x)
                
        self._model = TinyBERT(self._vocab_size, 312, 4, 12).to(self._device)
        self._model.train()
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=3e-4)
        return True

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        criterion = nn.CrossEntropyLoss()
        for step in range(self._total_steps):
            step_start = time.time()
            input_ids = torch.randint(0, self._vocab_size, (self._batch_size, self._seq_len), device=self._device)
            labels = torch.randint(0, self._vocab_size, (self._batch_size, self._seq_len), device=self._device)
            
            self._optimizer.zero_grad()
            logits = self._model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            self._optimizer.step()
            
            step_time = time.time() - step_start
            self._metrics.step_times.append(step_time)
            self._metrics.total_compute_time += step_time
            self._metrics.steps_completed += 1
            self._current_step = step + 1
            
            if self._current_step % self._checkpoint_interval == 0 and self._checkpoint_queue:
                ckpt_data = {"step": self._current_step, "model_state": {k: v.cpu().clone() for k, v in self._model.state_dict().items()}}
                await self._checkpoint_queue.put({"priority": "checkpoint", "data": ckpt_data})
            
            if step % 10 == 0: await asyncio.sleep(0)
            
        return {"steps_completed": self._metrics.steps_completed, "avg_step_time": sum(self._metrics.step_times)/len(self._metrics.step_times)}

    async def shutdown(self) -> bool:
        del self._model, self._optimizer
        if self._device.type == "cuda": torch.cuda.empty_cache()
        return True
