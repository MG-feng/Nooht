"""里程碑1交付物：每个可用后端的基础前向/反向传播正确性测试。"""
import pytest
import torch

from nooht.backends import list_backends
from nooht.backends.kernels import reference_selective_scan

ATOL, RTOL = 1e-3, 1e-3


def _make_inputs(device, seed=0):
    g = torch.Generator().manual_seed(seed)
    B, D, L, N = 2, 16, 32, 8
    x = torch.randn(B, D, L, generator=g, device=device, requires_grad=True)
    dt = torch.randn(B, D, L, generator=g, device=device, requires_grad=True)
    A = (torch.rand(D, N, generator=g, device=device) + 0.5).requires_grad_(True)
    Bm = torch.randn(B, N, L, generator=g, device=device, requires_grad=True)
    Cm = torch.randn(B, N, L, generator=g, device=device, requires_grad=True)
    Dm = torch.randn(D, generator=g, device=device, requires_grad=True)
    return x, dt, A, Bm, Cm, Dm


@pytest.mark.parametrize("backend", list_backends(only_available=True),
                         ids=lambda b: b.name)
def test_forward_backward_correctness(backend):
    backend.setup()
    device = backend.default_device()

    x, dt, A, Bm, Cm, Dm = _make_inputs(device)
    y = backend.selective_scan(x, dt, A, Bm, Cm, Dm, delta_softplus=True)
    y.pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert dt.grad is not None and A.grad is not None

    # 与 CPU 参考实现数值对齐（本阶段各后端均走参考 kernel，容差内必须一致）
    x_c, dt_c, A_c, B_c, C_c, D_c = _make_inputs(torch.device("cpu"))
    y_ref = reference_selective_scan(x_c, dt_c, A_c, B_c, C_c, D_c,
                                     delta_softplus=True)
    assert torch.allclose(y.detach().cpu(), y_ref, atol=ATOL, rtol=RTOL)