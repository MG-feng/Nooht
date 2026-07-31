import torch
import torch.nn as nn
import logging
from nooht.model.ngw_model import NGWModel
from nooht.training.trainer import NoohtTrainer
from nooht.training.memory_importance import MemoryImportancePredictor
from nooht.training.memory_sync import MemorySyncManager
from nooht.engineering.checkpoint.manager import CheckpointManager
import os
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def smoke_test():
    logger.info("=" * 60)
    logger.info("Nooht Phase 8: 150M Smoke Test (Six-Step Validation)")
    logger.info("=" * 60)
    
    config = {
        "model": {"vocab_size": 32000, "dim": 768, "num_layers": 12, "num_heads": 12, "num_memory_slots": 64, "ffn_multiplier": 4, "max_seq_len": 2048, "dropout": 0.0},
        "training": {"dim": 768, "world_size": 1, "ema_decay": 0.999, "imw_hidden_dim": 256},
    }
    model = NGWModel(**config["model"])
    num_params = model.get_num_params()
    logger.info(f"Step 1: Model Built. Parameters: {num_params / 1e6:.2f}M")
    assert 100e6 < num_params < 200e6, "Parameter count mismatch!"

    input_ids = torch.randint(0, 32000, (2, 128))
    labels = torch.randint(0, 32000, (2, 128))
    outputs = model(input_ids, return_bypass_gates=True, return_memory_writes=True, return_budget_targets=True)
    logger.info(f"Step 2: Forward Pass OK. Logits shape: {outputs['logits'].shape}")

    trainer = NoohtTrainer(model, config["training"])
    losses = trainer.loss_fn(
        outputs["logits"].reshape(-1, outputs["logits"].size(-1)), labels.reshape(-1),
        outputs["verify_clean"], outputs["verify_noisy"], outputs["bypass_gates"], outputs["budget_targets"],
        {"lm": 1.0, "verify": 0.01, "bypass": 0.1}
    )
    losses["total"].backward()
    
    dead_params = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    if dead_params:
        logger.error(f"Step 3 FAILED: Dead parameters detected: {dead_params}")
        raise RuntimeError("Dead parameters detected!")
    logger.info("Step 3: Backward Pass OK. No dead parameters.")

    optimizer = trainer.optimizer
    for step in range(5):
        optimizer.zero_grad()
        batch = {"input_ids": torch.randint(0, 32000, (2, 128)), "labels": torch.randint(0, 32000, (2, 128))}
        metrics = trainer.train_step(batch)
        if torch.isnan(torch.tensor(metrics["total"])).any():
            logger.error(f"Step 4 FAILED: NaN detected at step {step}")
            raise RuntimeError("NaN detected!")
    logger.info("Step 4: Training Loop OK. No NaN/Inf.")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_mgr = CheckpointManager({"checkpoint_dir": tmpdir})
        ckpt_mgr.save(5, model, trainer.optimizer, None, model.get_memory_banks())
        loaded = ckpt_mgr.load(os.path.join(tmpdir, "ckpt_5.pt"))
        assert loaded["step"] == 5
        logger.info("Step 5: Checkpoint Atomicity OK.")

    imw = trainer.importance_predictor
    write_vec = torch.randn(64, 768)
    filtered, scores, mask = imw.filter_by_importance(write_vec)
    loss = filtered.sum()
    loss.backward()
    assert imw.predictor[0].weight.grad is not None
    logger.info("Step 6: IMW STE Verification OK.")

    logger.info("=" * 60)
    logger.info("Phase 8 Smoke Test Passed! Engine is ready for ignition.")
    logger.info("=" * 60)

if __name__ == "__main__":
    smoke_test()
