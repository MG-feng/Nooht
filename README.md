# Nooht Architecture v1.0

Independent AI Architecture + Distributed Training/Inference System

## Core Components

- **NGWModel**: Transformer variant with MemoryBank, Bypass Gate, and Self-Verifier
- **HMC**: Hierarchical Memory Compression (L0/L1/L2)
- **Multi-Optimizer Router**: Lion/AdamW/Muon/Adafactor per parameter group
- **Dynamic Loss Scheduler**: Performance-driven weight adjustment
- **NRT Scheduler**: Non-blocking Real-time Task scheduling with priority queues

## Model Architecture

| Component | Details |
|-----------|---------|
| Attention | Multi-head + RoPE + FlashAttention (SDPA) |
| Memory | 64-slot differentiable MemoryBank per layer |
| Normalization | RMSNorm |
| FFN | GELU activation |
| Verifier | NCE-based Self-Verifier |

## Project Structure

```
nooht/
├── model/          # NGWModel, NGWBlock, Loss
├── training/       # Trainer, OptimizerRouter, MemorySync
├── distributed/    # Parameter Server
├── hmc/            # Hierarchical Memory Compression
├── engineering/    # Checkpoint, Dataset, Inference, Fault Recovery
├── engine/         # Pipeline Engine Core
├── runtime/        # Compute Graph, Device Manager, Scheduler
├── config/         # Configuration Schema & Migrator
├── plugins/        # PyTorch Training Plugins (Train, Checkpoint, Data)
└── experiments/    # Smoke Tests, Scaling Law Prober
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0

## Quick Start

```python
from nooht.model.ngw_model import NGWModel

model = NGWModel(
    vocab_size=32000,
    dim=768,
    num_layers=12,
    num_heads=12,
    num_memory_slots=64
)
```

## NRT Benchmark

Run the comparison between baseline sync training and NRT async training:

```bash
python -m benchmarks.compare
```

Expected results:
- **GPU Idle Time**: Eliminated (NRT overlaps checkpoint IO with computation)
- **Total Training Time**: Reduced by 5-15% in IO-intensive scenarios

## License

TBD
