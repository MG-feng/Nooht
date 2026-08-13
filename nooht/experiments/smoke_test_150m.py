import torch
import logging
from nooht.model.ngw_model import NGWModel
from nooht.training.trainer import NoohtTrainer
from nooht.engineering.checkpoint.manager import CheckpointManager
import os, tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def smoke_test():
    logger.info("=" * 60)
    logger.info("Nooht Phase 8: 150M Smoke Test")
    logger.info("=" * 60)
    
    config = {
        "model": {"vocab_size": 32000, "dim": 768, "num_layers": 12, "num_heads": 12, "num_memory_slots": 64},
        "training": {"dim": 768, "world_size": 1, "ema_decay": 0.999, "imw_hidden_dim": 256},
    }
    model = NGWModel(**config["model"])
    logger.info(f"Step 1: Model Built. Parameters: {model.get_num_params() / 1e6:.2f}M")

    # ✅ 修复点 4：Smoke Test Dummy Data 自回归移位
    tokens = torch.randint(0, 32000, (2, 129))
    input_ids = tokens[:, :-1]
    labels = tokens[:, 1:].clone()
    labels[:, -1] = -100
    
    outputs = model(input_ids, return_bypass_gates=True, return_memory_writes=True, return_budget_targets=True)
    logger.info(f"Step 2: Forward Pass OK.")

    trainer = NoohtTrainer(model, config["training"])
    losses = trainer.loss_fn(
        outputs["logits"].reshape(-1, outputs["logits"].size(-1)), labels.reshape(-1),
        outputs["verify_clean"], outputs["verify_noisy"], outputs["bypass_gates"], outputs["budget_targets"],
        {"lm": 1.0, "verify": 0.01, "bypass": 0.1}
    )
    losses["total"].backward()
    
    dead_params = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    if dead_params: raise RuntimeError(f"Dead parameters: {dead_params}")
    logger.info("Step 3: Backward Pass OK.")

    for step in range(5):
        trainer.optimizer.zero_grad()
        
        # 循环内也保持移位
        t = torch.randint(0, 32000, (2, 129))
        batch = {"input_ids": t[:, :-1], "labels": t[:, 1:].clone()}
        batch["labels"][:, -1] = -100
        
        metrics = trainer.train_step(batch)
        if torch.isnan(torch.tensor(metrics["total"])).any(): raise RuntimeError("NaN detected!")
    logger.info("Step 4: Training Loop OK.")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_mgr = CheckpointManager({"checkpoint_dir": tmpdir})
        ckpt_mgr.save(5, model, trainer.optimizer, None, model.get_memory_banks())
        loaded = ckpt_mgr.load(os.path.join(tmpdir, "ckpt_5.pt"))
        assert loaded["step"] == 5
    logger.info("Step 5: Checkpoint OK.")

    logger.info("=" * 60)
    logger.info("Phase 8 Smoke Test Passed! Engine is ready for ignition.")
    logger.info("=" * 60)

if __name__ == "__main__":
    smoke_test()
