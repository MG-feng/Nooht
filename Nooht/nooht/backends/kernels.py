"""参考内核实现（纯 PyTorch，设备无关）。

签名严格对齐 transformers jamba 官方 kernel；数学与官方 torch 回退路径等价，
由 tests 中的独立 oracle 锁死。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN


def reference_selective_scan(
    hidden_states: torch.Tensor,        # (B, D, L)
    dt: torch.Tensor,                   # (B, D, L)
    A: torch.Tensor,                    # (D, N)
    B: torch.Tensor,                    # (B, N, L)
    C: torch.Tensor,                    # (B, N, L)
    D: Optional[torch.Tensor] = None,   # (D,)
    z: Optional[torch.Tensor] = None,   # (B, D, L)
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
    use_mambapy: bool = False,          # 接受并忽略（参考实现恒走顺序递推）
    use_associative_scan: bool = False,
    **kwargs,
):
    batch, dim, length = hidden_states.shape
    n = A.shape[1]

    if delta_bias is not None:
        dt = dt + delta_bias[None, :, None]
    if delta_softplus:
        dt = F.softplus(dt)

    delta_A = torch.exp(dt.unsqueeze(-1) * A[None, :, None, :])      # (B, D, L, N)
    delta_Bx = (dt.unsqueeze(-1)
                * B.permute(0, 2, 1)[:, None, :, :]
                * hidden_states.unsqueeze(-1))                       # (B, D, L, N)

    C_t = C.permute(0, 2, 1)                                         # (B, L, N)
    h = hidden_states.new_zeros(batch, dim, n)
    ys = []
    for t in range(length):
        h = delta_A[:, :, t] * h + delta_Bx[:, :, t]
        ys.append(torch.einsum("bdn,bn->bd", h, C_t[:, t]))
    y = torch.stack(ys, dim=-1)                                      # (B, D, L)

    if D is not None:
        y = y + hidden_states * D[None, :, None]
    if z is not None:
        y = y * F.silu(z)

    # return_last_state=True 返回 (y, h)；h 为最后一步递推状态 (B, D, N)
    return (y, h) if return_last_state else y


def reference_selective_state_update(
    state, hidden_states, dt, A, B, C,
    D=None, dt_bias=None, dt_softplus=False, z=None, **kwargs,
):
    """decode 单步递推；与官方一致：就地更新 state 并返回当前步输出。"""
    input_dtype = hidden_states.dtype
    if dt_bias is not None:
        dt = dt + dt_bias.to(dt.dtype)
    if dt_softplus:
        dt = F.softplus(dt)
    dA = torch.exp(dt.float()[..., None] * A.float()).to(device=state.device)
    dB = dt.float()[..., None] * B.float()[:, None, :]
    dBx = dB * hidden_states.float()[..., None]
    ssm_state = state.float() * dA + dBx
    state.copy_(ssm_state.to(state.dtype))
    out = torch.matmul(ssm_state.to(C.dtype), C.unsqueeze(-1)).squeeze(-1)
    if D is not None:
        out = out + hidden_states * D
    if z is not None:
        out = out * F.silu(z)
    return out.to(input_dtype)


def reference_causal_conv1d_fn(hidden_states, weight, bias=None,
                               activation=None, **kwargs):
    """全序列因果卷积参考实现，公式与官方 causal_conv1d_fn 一致。"""
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1
    out = F.conv1d(hidden_states.to(weight.dtype), weight=weight.unsqueeze(1),
                   bias=bias, padding=padding, groups=hidden_size)[:, :, :seq_len]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def reference_causal_conv1d_update(hidden_states, conv_state, weight,
                                   bias=None, activation=None):
    """decode 单步卷积参考实现；与官方一致，就地更新 conv_state。"""
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0,
                   groups=hidden_size)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)