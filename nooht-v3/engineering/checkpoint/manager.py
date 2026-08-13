import torch
import os
import logging
from typing import Dict, Any
logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, config):
        self.save_dir = config.get("checkpoint_dir", "./checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)
    def save(self, step, model, optimizer, scheduler, memory_banks):
        path = os.path.join(self.save_dir, f"ckpt_{step}.pt")
        tmp = path + ".tmp"
        state = {"step": step, "model_state": model.state_dict(), "opt_state": optimizer.state_dict() if optimizer else None}
        torch.save(state, tmp)
        os.replace(tmp, path)
        logger.info(f"Checkpoint saved: {path}")
    def load(self, path):
        return torch.load(path)
