import asyncio, time, math, torch
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from nooht.runtime.abi.plugin_base import NoohtPlugin, ResourceRequest
from nooht.model.ngw_model import NGWModel
from nooht.training.trainer import NoohtTrainer

@dataclass
class TrainingMetrics:
    step_times: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    steps_completed: int = 0

class CosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            scale = self.current_step / max(self.warmup_steps, 1)
        else:
            progress = (self.current_step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            scale = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale

class PyTorchTrainPlugin(NoohtPlugin):
    @property
    def name(self) -> str: return "pytorch_train"
    @property
    def version(self) -> str: return "3.1.0"
    
    def __init__(self):
        self._model = None
        self._trainer = None
        self._scheduler = None
        self._scaler = None
        self._device = None
        self._metrics = TrainingMetrics()

    def get_resource_request(self) -> ResourceRequest:
        return ResourceRequest(vram_mb=4096, cpu_cores=4, ram_mb=2048, device_type="cuda")

    async def init(self, config: Dict[str, Any]) -> bool:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_cfg = config.get("model", {})
        self._model = NGWModel(
            vocab_size=model_cfg.get("vocab_size", 32000),
            dim=model_cfg.get("dim", 768),
            num_layers=model_cfg.get("num_layers", 12),
            num_heads=model_cfg.get("num_heads", 12),
            num_memory_slots=model_cfg.get("num_memory_slots", 64),
        ).to(self._device)
        
        train_cfg = config.get("training", {"dim": model_cfg.get("dim", 768), "world_size": 1})
        self._trainer = NoohtTrainer(self._model, train_cfg)
        self._scheduler = CosineScheduler(self._trainer.optimizer.all_optimizers[0], 100, config.get("total_steps", 500))
        
        if self._device.type == "cuda" and not torch.cuda.is_bf16_supported():
            self._scaler = GradScaler()
        return True

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        all_opts = self._trainer.optimizer.all_optimizers
        for step in range(inputs.get("total_steps", 500)):
            start = time.time()
            B, S = 8, 128
            
            # ✅ 修复点 3：Dummy Batch 自回归移位与 ignore_index
            tokens = torch.randint(0, self._model.vocab_size, (B, S + 1), device=self._device)
            batch = {
                "input_ids": tokens[:, :-1],
                "labels": tokens[:, 1:].clone()
            }
            batch["labels"][:, -1] = -100
            
            for opt in all_opts: opt.zero_grad(set_to_none=True)
            use_amp = self._device.type == "cuda"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            
            if use_amp:
                with autocast(device_type="cuda", dtype=dtype):
                    out = self._model(batch["input_ids"], return_bypass_gates=True, return_memory_writes=True, return_budget_targets=True)
                    losses = self._trainer.loss_fn(out["logits"].view(-1, self._model.vocab_size), batch["labels"].view(-1), out["verify_clean"], out["verify_noisy"], out["bypass_gates"], out["budget_targets"], self._trainer.scheduler.weights)
                if self._scaler: self._scaler.scale(losses["total"]).backward()
                else: losses["total"].backward()
            else:
                out = self._model(batch["input_ids"], return_bypass_gates=True, return_memory_writes=True, return_budget_targets=True)
                losses = self._trainer.loss_fn(out["logits"].view(-1, self._model.vocab_size), batch["labels"].view(-1), out["verify_clean"], out["verify_noisy"], out["bypass_gates"], out["budget_targets"], self._trainer.scheduler.weights)
                losses["total"].backward()

            if self._scaler:
                for opt in all_opts: self._scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            if self._scaler:
                for opt in all_opts: self._scaler.step(opt)
                self._scaler.update()
            else:
                for opt in all_opts: opt.step()
                
            self._scheduler.step()
            self._metrics.step_times.append(time.time() - start)
            self._metrics.losses.append(losses["total"].item())
            self._metrics.steps_completed += 1
            
            if (step + 1) % 50 == 0:
                print(f"  Step {step+1} | Loss: {losses['total'].item():.4f}")
                
        return {"steps": self._metrics.steps_completed, "loss": sum(self._metrics.losses)/len(self._metrics.losses)}

    async def shutdown(self) -> bool: return True
