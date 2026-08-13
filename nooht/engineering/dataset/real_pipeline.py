"""真实数据管道 + W&B 监控集成"""
import torch
from torch.utils.data import IterableDataset
from typing import Iterator, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RealDataPipeline(IterableDataset):
    def __init__(self, hf_dataset_name: str, tokenizer, seq_len: int = 2048):
        from datasets import load_dataset
        self.stream = load_dataset(hf_dataset_name, split="train", streaming=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer = []
        for sample in self.stream:
            text = sample.get("text", "")
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            while len(buffer) > self.seq_len:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                yield {"input_ids": torch.tensor(chunk[:-1]), "labels": torch.tensor(chunk[1:])}

class TelemetryMonitor:
    """W&B / TensorBoard 监控面板"""
    def __init__(self, project_name="nooht-v4"):
        try:
            import wandb
            self.wandb = wandb
            wandb.init(project=project_name, config={"architecture": "NGW-v4"})
            self.enabled = True
        except ImportError:
            self.enabled = False

    def log(self, metrics: Dict[str, float], step: int):
        if self.enabled: self.wandb.log(metrics, step=step)
