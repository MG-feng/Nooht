import torch.nn as nn
from torch.utils.checkpoint import checkpoint
def apply_gradient_checkpointing(model, ratio=1.0):
    if not hasattr(model, "blocks"): return model
    num = int(len(model.blocks) * ratio)
    for i in range(num):
        orig = model.blocks[i].forward
        def make_ckpt(fn):
            def fwd(*args, **kwargs): return checkpoint(fn, *args, use_reentrant=False, **kwargs)
            return fwd
        model.blocks[i].forward = make_ckpt(orig)
    return model
