import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import List

class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay); super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad; state = self.state[p]
                if len(state) == 0: state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]; beta1, beta2 = group["betas"]
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, momentum=0.95):
        defaults = dict(lr=lr, momentum=momentum); super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad; state = self.state[p]
                if len(state) == 0: state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]; buf.mul_(group["momentum"]).add_(grad)
                p.add_(buf, alpha=-group["lr"])

class Adafactor(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4): defaults = dict(lr=lr); super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.add_(p.grad, alpha=-group["lr"])

class OptimizerRouter:
    def __init__(self, model: nn.Module):
        mem_params, reason_params, know_params = [], [], []
        embed_params, ncc_dense_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            if "memory_bank" in name: mem_params.append(p)
            elif "reasoning" in name or "verifier" in name: reason_params.append(p)
            elif "knowledge" in name: know_params.append(p)
            elif "token_embedding" in name or "output_proj" in name: embed_params.append(p)
            else: ncc_dense_params.append(p)
        self.opt_mem = Lion(mem_params, lr=5e-5)
        self.opt_reason = AdamW(reason_params, lr=2e-4)
        self.opt_know = Muon(know_params, lr=8e-5)
        self.opt_embed = Adafactor(embed_params, lr=1e-4)
        self.opt_ncc = AdamW(ncc_dense_params, lr=1e-4, weight_decay=0.01)
        self.all_optimizers = [self.opt_mem, self.opt_reason, self.opt_know, self.opt_embed, self.opt_ncc]
    def step(self):
        for opt in self.all_optimizers: opt.step()
    def zero_grad(self, set_to_none=True):
        for opt in self.all_optimizers: opt.zero_grad(set_to_none=set_to_none)
