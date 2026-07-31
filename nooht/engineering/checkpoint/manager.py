import torch
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, config: Dict[str, Any]):
        self.save_dir = config.get("checkpoint_dir", "./checkpoints"); self.max_to_keep = config.get("max_to_keep", 3)
        os.makedirs(self.save_dir, exist_ok=True)
    def save(self, step: int, model: torch.nn.Module, optimizer: Any, scheduler: Any, memory_banks: list):
        ckpt_path = os.path.join(self.save_dir, f"ckpt_{step}.pt"); tmp_path = ckpt_path + ".tmp"
        state = {"step": step, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict() if optimizer else None, "scheduler_state": scheduler.state_dict() if scheduler else None, "memory_state": [m.memory.data.clone() for m in memory_banks]}
        torch.save(state, tmp_path); os.rename(tmp_path, ckpt_path); logger.info(f"Checkpoint saved to {ckpt_path}")
    def load(self, path: str) -> Dict[str, Any]:
        if not self.verify_integrity(path): raise RuntimeError(f"Checkpoint {path} corrupted!")
        return torch.load(path)
    def auto_resume(self) -> Dict[str, Any]:
        files = [f for f in os.listdir(self.save_dir) if f.startswith("ckpt_")]
        if not files: return None
        latest = sorted(files, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
        return self.load(os.path.join(self.save_dir, latest))
    def verify_integrity(self, path: str) -> bool: return os.path.exists(path) and os.path.getsize(path) > 0
