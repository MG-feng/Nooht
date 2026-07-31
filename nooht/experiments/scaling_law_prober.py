import torch
import torch.nn as nn
from typing import Dict, Any, List
from dataclasses import dataclass
import logging
from nooht.model.ngw_model import NGWModel

logger = logging.getLogger(__name__)

@dataclass
class ModelProfile:
    name: str
    params: int
    flops: int
    peak_memory: int

PROBE_CONFIGS = [
    {"name": "50M", "dim": 512, "num_layers": 4, "num_heads": 8, "num_memory_slots": 32, "ffn_multiplier": 4},
    {"name": "150M", "dim": 768, "num_layers": 12, "num_heads": 12, "num_memory_slots": 64, "ffn_multiplier": 4},
    {"name": "300M", "dim": 1024, "num_layers": 24, "num_heads": 16, "num_memory_slots": 128, "ffn_multiplier": 4},
]

def run_scaling_probe():
    logger.info("=" * 60)
    logger.info("Nooht Scaling Law Probe")
    logger.info("=" * 60)
    
    profiles = []
    
    for cfg in PROBE_CONFIGS:
        logger.info(f"Probing {cfg['name']} model...")
        model = NGWModel(
            vocab_size=32000, dim=cfg["dim"], num_layers=cfg["num_layers"], 
            num_heads=cfg["num_heads"], num_memory_slots=cfg["num_memory_slots"], 
            ffn_multiplier=cfg["ffn_multiplier"], max_seq_len=2048, dropout=0.0
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        input_ids = torch.randint(0, 32000, (1, 2048)).to(device)
        
        model.eval()
        with torch.no_grad():
            for _ in range(3): _ = model(input_ids)
                
        if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_flops=True
        ) as prof:
            with torch.no_grad():
                for _ in range(10):
                    _ = model(input_ids)
                    
        if torch.cuda.is_available(): torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        
        total_flops = 0
        for event in prof.key_averages():
            if hasattr(event, "flops") and event.flops > 0:
                total_flops += event.flops
                
        total_flops = total_flops // 10
        num_params = model.get_num_params()
        
        profile = ModelProfile(
            name=cfg["name"], params=num_params, flops=total_flops, peak_memory=peak_mem
        )
        profiles.append(profile)
        
        logger.info(f"Profile [{profile.name}]: Params={profile.params/1e6:.2f}M, FLOPs={profile.flops/1e9:.2f}G, PeakMem={profile.peak_memory/1e9:.2f}GB")
        
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
    logger.info("=" * 60)
    logger.info("Scaling Law Probe Complete.")
    logger.info("=" * 60)
    return profiles

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_scaling_probe()
