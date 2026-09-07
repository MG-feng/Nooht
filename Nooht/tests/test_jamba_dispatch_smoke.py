"""Jamba 派发端到端测试（vendored / integration / 钉版本环境）。

覆盖：
- vendored 改造点存在（防回退到未改造官方版）
- 默认后端严格调用官方原函数（独立于数值）
- 接管经由正式接口 backend_override
- 契约列表单一事实源
- prefill+decode 双路径、四 kernel 全覆盖的最终门禁
"""
import pytest

pytest.importorskip("transformers")

from nooht.modeling.verify import verify_dispatch  # noqa: E402

pytestmark = pytest.mark.integration


def test_vendored_module_exposes_dispatch_points():
    from transformers.models.jamba import modeling_jamba as mj
    assert hasattr(mj, "_nooht_hook")
    assert hasattr(mj, "NOOHT_KERNEL_DISPATCH_POINTS")
    for name in ("mamba_selective_scan", "mamba_selective_state_update",
                 "causal_conv1d_fn", "causal_conv1d_update"):
        assert hasattr(getattr(mj, name), "__wrapped__")


def test_hook_default_backend_calls_original_function_only():
    """overrides_kernel=False → 严格调用原函数；后端同名方法存在也绝不被调用。"""
    import torch
    from nooht.backends import backend_override
    from nooht.backends.base import NoohtBackend
    from transformers.models.jamba import modeling_jamba as mj

    calls = {"orig": 0, "backend_method": 0}

    def orig(x):
        calls["orig"] += 1
        return x

    class _DefaultLike(NoohtBackend):
        name = "default-like"
        priority = 0
        def is_available(self): return True
        def default_device(self): return torch.device("cpu")
        def selective_scan(self, *a, **k):
            calls["backend_method"] += 1
            return "MUST_NOT_HAPPEN"

    hooked = mj._nooht_hook("selective_scan", orig)
    with backend_override(_DefaultLike()):
        assert hooked(1) is 1
    assert calls == {"orig": 1, "backend_method": 0}


def test_hook_takeover_via_official_interface():
    """接管路径使用正式接口 backend_override，不触碰私有状态。"""
    import torch
    from nooht.backends import backend_override
    from nooht.backends.base import NoohtBackend
    from transformers.models.jamba import modeling_jamba as mj

    class _Fake(NoohtBackend):
        name = "fake"
        priority = 0
        def is_available(self): return True
        def default_device(self): return torch.device("cpu")
        def overrides_kernel(self, n): return n == "selective_scan"
        def selective_scan(self, x): return ("fake", x)

    hooked = mj._nooht_hook("selective_scan", lambda x: x)
    with backend_override(_Fake()):
        assert hooked(7) == ("fake", 7)


def test_dispatch_point_lists_in_sync():
    """契约列表单一事实源：防止只改一边造成检查盲区。"""
    from nooht.backends.base import KERNEL_NAMES
    from transformers.models.jamba import modeling_jamba as mj
    assert tuple(mj.NOOHT_KERNEL_DISPATCH_POINTS) == tuple(KERNEL_NAMES)


def test_four_kernel_takeover_prefill_and_decode():
    """最终门禁：三项独立断言，缺一即失败。"""
    result = verify_dispatch()
    assert result.dispatched, (
        f"hook 未被到达：{result.error}; queries={result.hook_queries}")
    assert result.kernel_takeover, (
        f"四 kernel 接管不完整：missing={result.missing_kernels}; "
        f"counts={result.takeover_counts}")
    assert result.numerics_match, result.error