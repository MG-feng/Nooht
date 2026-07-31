import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Any, Optional
import logging
import time

from nooht.model.ngw_model import NGWModel
from nooht.training.trainer import NoohtTrainer
from nooht.engineering.dataset.pipeline import DatasetPipeline

logger = logging.getLogger(__name__)

class StreamingTextDataset(IterableDataset):
    def __iter__(self):
        while True:
            yield {"input_ids": torch.randint(0, 32000, (2048,)), "labels": torch.randint(0, 32000, (2048,))}

def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    return dist.get_rank(), dist.get_world_size()

class PretrainLauncher:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rank = 0
        self.world_size = 1
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.rank, self.world_size = setup_ddp()
            
        device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        self.model = NGWModel(**config["model"]).to(device)
        
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank])
            
        if hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("torch.compile applied successfully.")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}. Falling back to eager mode.")
                
        self.trainer = NoohtTrainer(self.model, config.get("training", {}))
        
        self.dataset = DataLoader(
            StreamingTextDataset(), 
            batch_size=config.get("batch_size", 4), 
            pin_memory=True, 
            prefetch_factor=2,
            num_workers=2
        )

    def launch(self, total_steps: int = 1000):
        logger.info(f"[Rank {self.rank}] Starting Pretrain Launcher...")
        data_iter = iter(self.dataset)
        
        for step in range(total_steps):
            try: batch = next(data_iter)
            except StopIteration: data_iter = iter(self.dataset); batch = next(data_iter)
            
            device = next(self.model.parameters()).device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            metrics = self.trainer.train_step(batch)
            
            if step % 10 == 0 and self.rank == 0:
                mem_l2 = 0.0
                byp_proxy = 0.0
                layers = 0
                for blk in self.model.blocks if not isinstance(self.model, DDP) else self.model.module.blocks:
                    mem_l2 += blk.memory_bank.memory.data.norm().item()
                    byp_proxy += blk.bypass_gate.state_dict()["weight"].abs().mean().item()
                    layers += 1
                mem_l2 /= layers
                byp_proxy /= layers
                
                logger.info(
                    f"Step {step} | Loss: {metrics['total']:.4f} | LM: {metrics['lm']:.4f} | "
                    f"Verify: {metrics['verify']:.4f} | Bypass: {metrics['bypass']:.4f} | "
                    f"MemL2: {mem_l2:.4f} | BypProxy: {byp_proxy:.4f}"
                )
                
        if dist.is_initialized():
            dist.destroy_process_group()
