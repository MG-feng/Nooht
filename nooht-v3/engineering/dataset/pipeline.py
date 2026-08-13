import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import Iterator, Dict, Any

class DatasetPipeline(IterableDataset):
    def __init__(self, config: Dict[str, Any]):
        self.vocab_size = config.get("vocab_size", 32000)
        self.seq_len = config.get("seq_len", 128)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        gen = torch.Generator()
        gen.manual_seed(42 + worker_id)
        step = 0
        
        while True:
            if step % num_workers != worker_id:
                step += 1
                continue
            step += 1
            
            tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,), generator=gen)
            input_ids = tokens[:-1]
            labels = tokens[1:]
            labels[-1] = -100 
            
            yield {"input_ids": input_ids, "labels": labels}
