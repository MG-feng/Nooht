import torch
import torch.nn as nn
import time
import os
from typing import List
from dataclasses import dataclass, field

@dataclass
class BenchmarkResult:
    name: str
    total_time: float = 0.0
    step_times: List[float] = field(default_factory=list)
    gpu_idle_times: List[float] = field(default_factory=list)
    checkpoints_saved: int = 0

def build_tiny_model():
    class TinyBERT(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(30522, 312)
            self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=312, nhead=12, dim_feedforward=1248, batch_first=True) for _ in range(4)])
            self.lm_head = nn.Linear(312, 30522)
        def forward(self, x):
            x = self.embedding(x)
            for l in self.layers: x = l(x)
            return self.lm_head(x)
    return TinyBERT()

def train_baseline(total_steps=500, checkpoint_interval=50, io_delay=2.0):
    result = BenchmarkResult(name="Baseline (Sync)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[BASELINE] Running on {device}...")
    
    model = build_tiny_model().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    ckpt_dir = "/tmp/baseline_ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    total_start = time.time()
    for step in range(total_steps):
        step_start = time.time()
        input_ids = torch.randint(0, 30522, (16, 128), device=device)
        labels = torch.randint(0, 30522, (16, 128), device=device)
        
        optimizer.zero_grad()
        loss = criterion(model(input_ids).view(-1, 30522), labels.view(-1))
        loss.backward()
        optimizer.step()
        
        if (step + 1) % checkpoint_interval == 0:
            ckpt_start = time.time()
            time.sleep(io_delay) # 模拟 IO
            # V2 Rule 32: 对齐 torch.save
            torch.save({"step": step+1}, os.path.join(ckpt_dir, f"ckpt_{step+1}.pt"))
            ckpt_time = time.time() - ckpt_start
            result.gpu_idle_times.append(ckpt_time)
            result.checkpoints_saved += 1
            
        result.step_times.append(time.time() - step_start)
        
    result.total_time = time.time() - total_start
    print(f"[BASELINE] Done. Total: {result.total_time:.2f}s, Idle: {sum(result.gpu_idle_times):.2f}s")
    return result
