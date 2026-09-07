"""reference kernel 正确性契约。

覆盖：shape/finite、forward+backward、D / z / delta_bias / delta_softplus 全组合、
return_last_state 语义、dtype、多种序列长度、CPU baseline、
独立数学递推 oracle、conv 公式级对照、state 更新内容 + activation。

last_state 语义：return_last_state=True 返回 (y, h)，h 为最后一步递推状态，
shape=(B, D, N)；当 D=None 且 z=None 时，契约保证
y[:, :, -1] == einsum(h, C[:, :, -1])。
"""
import itertools

import pytest
import torch

from nooht.backends.kernels import (reference_causal_conv1d_fn,
                                    reference_causal_conv1d_update,
                                    reference_selective_scan)

B, D, N = 2, 8, 4


def _inputs(seq_len, dtype, requires_grad=False):
    torch.manual_seed(0)
    x = torch.randn(B, D, seq_len, dtype=dtype).requires_grad_(requires_grad)
    dt = torch.randn(B, D, seq_len, dtype=dtype).requires_grad_(requires_grad)
    A = (torch.rand(D, N, dtype=dtype) + 0.5).detach().requires_grad_(requires_grad)
    Bm = torch.randn(B, N, seq_len, dtype=dtype).requires_grad_(requires_grad)
    Cm = torch.randn(B, N, seq_len, dtype=dtype).requires_grad_(requires_grad)
    Dm = torch.randn(D, dtype=dtype).requires_grad_(requires_grad)
    z = torch.randn(B, D, seq_len, dtype=dtype).requires_grad_(requires_grad)
    bias = torch.randn(D, dtype=dtype)
    return x, dt, A, Bm, Cm, Dm, z, bias


@pytest.mark.parametrize("seq_len", [1, 7, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("use_D,use_z,use_bias,softplus",
                         list(itertools.product([False, True], repeat=4)))
def test_forward_backward_contract(seq_len, dtype, use_D, use_z, use_bias, softplus):
    x, dt, A, Bm, Cm, Dm, z, bias = _inputs(seq_len, dtype, requires_grad=True)
    y = reference_selective_scan(
        x, dt, A, Bm, Cm,
        D=Dm if use_D else None,
        z=z if use_z else None,
        delta_bias=bias if use_bias else None,
        delta_softplus=softplus,
    )
    assert y.shape == (B, D, seq_len)
    assert torch.isfinite(y).all()

    y.sum().backward()
    for t in (x, dt, A, Bm, Cm):
        assert t.grad is not None and torch.isfinite(t.grad).all()


def test_last_state_semantics():
    seq_len = 12
    x, dt, A, Bm, Cm, *_ = _inputs(seq_len, torch.float32)
    y, h = reference_selective_scan(x, dt, A, Bm, Cm, return_last_state=True)
    assert h.shape == (B, D, N)
    y_last = torch.einsum("bdn,bn->bd", h, Cm[:, :, -1])
    assert torch.allclose(y[:, :, -1], y_last, atol=1e-6)


@pytest.mark.parametrize("seq_len", [1, 32])
def test_bfloat16_forward_only(seq_len):
    x, dt, A, Bm, Cm, Dm, *_ = _inputs(seq_len, torch.bfloat16)
    y = reference_selective_scan(x, dt, A, Bm, Cm, D=Dm)
    assert y.dtype == torch.bfloat16
    assert y.shape == (B, D, seq_len)
    assert torch.isfinite(y.float()).all()


# ---------------------------------------------------------------------------
# 独立数学递推 oracle：纯 Python 标量手写递推，实现路径与向量化参考实现完全独立
# ---------------------------------------------------------------------------

def _oracle_scalar_scan(x, dt, A, B, C, D=None, softplus=True):
    """B=D=N=1 的手写递推；x/dt/B/C 为长度 L 的 list，A 为标量。"""
    import math
    h = 0.0
    ys = []
    for t in range(len(x)):
        d = math.log1p(math.exp(dt[t])) if softplus else dt[t]
        h = math.exp(d * A) * h + d * B[t] * x[t]
        ys.append(h * C[t])
    if D is not None:
        ys = [ys[t] + D * x[t] for t in range(len(x))]
    return ys


@pytest.mark.parametrize("seq_len", [1, 2, 3])
@pytest.mark.parametrize("softplus", [True, False])
@pytest.mark.parametrize("use_D", [False, True])
def test_scalar_recurrence_oracle(seq_len, softplus, use_D):
    torch.manual_seed(42)
    x = torch.randn(1, 1, seq_len)
    dt = torch.randn(1, 1, seq_len) * 0.3   # 小值，避免 softplus/exp 溢出
    A = torch.rand(1, 1) + 0.5
    Bm = torch.randn(1, 1, seq_len)
    Cm = torch.randn(1, 1, seq_len)
    Dm = torch.randn(1) if use_D else None

    y = reference_selective_scan(x, dt, A, Bm, Cm, D=Dm, delta_softplus=softplus)
    expected = _oracle_scalar_scan(
        x[0, 0].tolist(), dt[0, 0].tolist(), A[0, 0].item(),
        Bm[0, 0].tolist(), Cm[0, 0].tolist(),
        D=Dm[0].item() if Dm is not None else None,
        softplus=softplus,
    )
    torch.testing.assert_close(y[0, 0], torch.tensor(expected),
                               atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# conv 参考内核：公式级对照（full causal conv 最后一步 ≡ incremental update）
# ---------------------------------------------------------------------------

def test_reference_conv_kernels_contract():
    torch.manual_seed(0)
    B_, H, K = 2, 4, 4
    w = torch.randn(H, K)
    state0 = torch.randn(B_, H, K - 1)
    x1 = torch.randn(B_, H, 1)

    out_update = reference_causal_conv1d_update(x1, state0.clone(), w)
    out_full = reference_causal_conv1d_fn(torch.cat([state0, x1], dim=-1), w)
    assert torch.allclose(out_full[:, :, -1:], out_update, atol=1e-6)
    assert out_update.shape == (B_, H, 1)


def test_reference_conv_update_mutates_state_in_place():
    torch.manual_seed(1)
    B_, H, K = 1, 2, 4
    w = torch.randn(H, K)
    state = torch.randn(B_, H, K - 1)
    before = state.clone()
    reference_causal_conv1d_update(torch.randn(B_, H, 1), state, w)
    assert not torch.equal(state, before)   # 与官方一致的 in-place 语义


def test_reference_conv_update_state_content_and_activation():
    """1) state 更新内容必须等于最新 K-1 帧；
    2) activation 分支 ≡ 手动对无激活输出施加 silu。"""
    torch.manual_seed(2)
    B_, H, K = 2, 4, 4
    w = torch.randn(H, K)
    state0 = torch.randn(B_, H, K - 1)
    x1 = torch.randn(B_, H, 1)

    state = state0.clone()
    out = reference_causal_conv1d_update(x1, state, w)

    assert torch.allclose(state, torch.cat([state0, x1], dim=-1)[:, :, -(K - 1):])

    state_a = state0.clone()
    out_act = reference_causal_conv1d_update(x1, state_a, w, activation="silu")
    assert torch.allclose(out_act, torch.nn.functional.silu(out), atol=1e-6)
    assert torch.allclose(state_a, state)   # activation 不影响 state 更新