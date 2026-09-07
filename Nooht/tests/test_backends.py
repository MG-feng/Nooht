import torch
import torch.nn.functional as F

from nooht.backends import detect_backend, list_backends
from nooht.backends.kernels import reference_selective_scan


def test_reference_scan_shape_and_determinism():
    torch.manual_seed(0)
    B, D, L, N = 2, 8, 16, 4
    x = torch.randn(B, D, L)
    dt = torch.randn(B, D, L)
    A = torch.rand(D, N) + 0.5
    Bm = torch.randn(B, N, L)
    Cm = torch.randn(B, N, L)
    Dm = torch.randn(D)
    y1 = reference_selective_scan(x, dt, A, Bm, Cm, Dm, delta_softplus=True)
    y2 = reference_selective_scan(x, dt, A, Bm, Cm, Dm, delta_softplus=True)
    assert y1.shape == (B, D, L)
    assert torch.allclose(y1, y2)


def test_reference_scan_matches_manual_recurrence():
    """手工展开递推，验证参考实现的递推正确性。"""
    torch.manual_seed(1)
    B, D, L, N = 1, 2, 5, 3
    x = torch.randn(B, D, L)
    dt = torch.randn(B, D, L)
    A = torch.rand(D, N) + 0.1
    Bm = torch.randn(B, N, L)
    Cm = torch.randn(B, N, L)

    y = reference_selective_scan(x, dt, A, Bm, Cm, D=None, delta_softplus=True)

    dts = F.softplus(dt)
    h = torch.zeros(B, D, N)
    expected = []
    for t in range(L):
        h = (torch.exp(dts[:, :, t].unsqueeze(-1) * A.unsqueeze(0)) * h
             + dts[:, :, t].unsqueeze(-1) * Bm[:, :, t].unsqueeze(1) * x[:, :, t].unsqueeze(-1))
        expected.append((h * Cm[:, :, t].unsqueeze(1)).sum(-1))
    expected = torch.stack(expected, dim=-1)
    assert torch.allclose(y, expected, atol=1e-5)


def test_detect_backend_returns_available():
    backend = detect_backend()
    assert backend.is_available()


def test_priority_order():
    names = [b.name for b in list_backends(only_available=False)]
    assert names.index("cuda") < names.index("rocm") < names.index("xla") \
        < names.index("oneapi") < names.index("cpu")


def test_five_backends_registered():
    names = {b.name for b in list_backends(only_available=False)}
    assert names == {"cuda", "rocm", "xla", "oneapi", "cpu"}