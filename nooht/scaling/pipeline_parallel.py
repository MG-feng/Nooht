import torch
import torch.nn as nn
class PipelineParallel:
    @staticmethod
    def split_model(model, num_stages, stage_id):
        if not hasattr(model, "blocks"): return model
        total = len(model.blocks)
        lps = total // num_stages
        rem = total % num_stages
        start = stage_id * lps + min(stage_id, rem)
        end = start + lps + (1 if stage_id < rem else 0)
        return nn.ModuleList(model.blocks[start:end])
