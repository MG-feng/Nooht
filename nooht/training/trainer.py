import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import logging
from .memory_sync import MemorySyncManager
from .memory_importance import MemoryImportancePredictor
from .loss import NoohtLoss, PerformanceDrivenScheduler
from .optimizer_router import OptimizerRouter

logger = logging.getLogger(__name__)

class NoohtTrainer:
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model; self.config = config; dim = config.get("dim", 768)
        self.importance_predictor = MemoryImportancePredictor(dim=dim, hidden_dim=config.get("imw_hidden_dim", 256))
        self.memory_banks = model.get_memory_banks() if hasattr(model, "get_memory_banks") else []
        self.memory_syncs = [
            MemorySyncManager(bank.memory, config.get("world_size", 1), config.get("ema_decay", 0.999), importance_predictor=self.importance_predictor)
            for bank in self.memory_banks
        ]
        self.optimizer = OptimizerRouter(model)
        self.scheduler = PerformanceDrivenScheduler(init_weights={"lm": 1.0, "mem": 0.1, "verify": 0.01, "bypass": 0.1}, max_weights={"lm": 1.0, "mem": 2.0, "verify": 1.5, "bypass": 1.0})
        self.loss_fn = NoohtLoss(); self.global_step = 0
        # ✅ FIX: 增加梯度裁剪阈值配置
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        
    def train_step(self, batch: Dict[str, torch.Tensor], metrics: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        self.model.train(); self.optimizer.zero_grad()
        if metrics: weights = self.scheduler.update_weights(metrics)
        else: weights = self.scheduler.weights
        outputs = self.model(batch["input_ids"], return_bypass_gates=True, return_memory_writes=True, return_budget_targets=True)
        losses = self.loss_fn(
            logits=outputs["logits"].reshape(-1, outputs["logits"].size(-1)),
            labels=batch["labels"].reshape(-1),
            verify_clean=outputs["verify_clean"], verify_noisy=outputs["verify_noisy"],
            bypass_gates=outputs["bypass_gates"], budget_targets=outputs["budget_targets"], weights=weights
        )
        losses["total"].backward()
        
        # ✅ FIX: 梯度裁剪 (防止 MemoryBank/BypassGate 梯度爆炸)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        if "memory_writes" in outputs:
            for sync_mgr, write_vec in zip(self.memory_syncs, outputs["memory_writes"]):
                sync_mgr.sync_and_update(write_vec)
        self.global_step += 1
        return {k: v.item() for k, v in losses.items()}
