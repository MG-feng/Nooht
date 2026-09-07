"""端到端验证（正式接口版）。三个判据互相独立：

  dispatched      —— 阶段1：Jamba forward 真实查询了 Nooht dispatch hook
                     （spy 后端只计数、不接管 → 证明 hook 可达，输出即官方基线）
  kernel_takeover —— 阶段2：四个 kernel 在 prefill+decode 全程被探针真实接管
                     （prefill → causal_conv1d_fn + selective_scan；
                       decode  → causal_conv1d_update + selective_state_update）
  numerics_match  —— 探针路径（数学等价的参考内核）与官方路径数值一致

语义边界：numerics_match 覆盖「贪心序列一致 + use_cache=False 的 prefill forward
logits 一致」；decode 路径的数值正确性由四个 takeover_counts > 0 与贪心序列一致
共同证明，不由 logits 对比推导（logits 对比不经过 decode kernel）。

后端注入只走正式接口 backend_override()，不触碰任何私有状态。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class VerifyResult:
    dispatched: bool
    kernel_takeover: bool
    numerics_match: bool
    hook_queries: Dict[str, int] = field(default_factory=dict)
    takeover_counts: Dict[str, int] = field(default_factory=dict)
    missing_kernels: List[str] = field(default_factory=list)
    error: Optional[str] = None


def make_tiny_jamba_config():
    """最小随机权重 Jamba 配置（字段名按官方 configuration_jamba.py 核对）。"""
    from transformers import JambaConfig
    return JambaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        num_experts=1, num_experts_per_tok=1,
        expert_layer_period=2, expert_layer_offset=1,
        attn_layer_period=4, attn_layer_offset=1,   # layer0=mamba, layer1=attention
        mamba_d_state=8, mamba_expand=1, mamba_d_conv=4,
        max_position_embeddings=64, tie_word_embeddings=False,
        use_mamba_kernels=False,     # 不依赖 mamba_ssm/causal_conv1d
        use_mambapy=False,
        use_associative_scan=False,  # 走顺序递推，跨环境数值稳定
    )


def _make_spy_backend():
    """阶段1：只计数、永不接管 → 其查询次数即 hook 可达证据。"""
    import torch
    from nooht.backends.base import KERNEL_NAMES, NoohtBackend

    class _HookReachabilitySpy(NoohtBackend):
        name = "hook-spy"
        priority = -1

        def __init__(self):
            self.queries = {name: 0 for name in KERNEL_NAMES}

        def is_available(self):
            return True

        def default_device(self):
            return torch.device("cpu")

        def overrides_kernel(self, name):
            if name in self.queries:
                self.queries[name] += 1
            return False

    return _HookReachabilitySpy()


def _make_probe_backend():
    """阶段2：接管全部四个 kernel，内部委托数学等价的参考实现。"""
    import torch
    from nooht.backends.base import KERNEL_NAMES, NoohtBackend

    class _KernelProbeBackend(NoohtBackend):
        name = "kernel-probe"
        priority = -1

        def __init__(self):
            self.calls = {name: 0 for name in KERNEL_NAMES}

        def is_available(self):
            return True

        def default_device(self):
            return torch.device("cpu")

        def overrides_kernel(self, name):
            return name in self.calls

        def selective_scan(self, *args, **kwargs):
            self.calls["selective_scan"] += 1
            from nooht.backends.kernels import reference_selective_scan
            return reference_selective_scan(*args, **kwargs)

        def selective_state_update(self, *args, **kwargs):
            self.calls["selective_state_update"] += 1
            from nooht.backends.kernels import reference_selective_state_update
            return reference_selective_state_update(*args, **kwargs)

        def causal_conv1d_fn(self, *args, **kwargs):
            self.calls["causal_conv1d_fn"] += 1
            from nooht.backends.kernels import reference_causal_conv1d_fn
            return reference_causal_conv1d_fn(*args, **kwargs)

        def causal_conv1d_update(self, *args, **kwargs):
            self.calls["causal_conv1d_update"] += 1
            from nooht.backends.kernels import reference_causal_conv1d_update
            return reference_causal_conv1d_update(*args, **kwargs)

    return _KernelProbeBackend()


def verify_dispatch(seed: int = 0, seq_len: int = 16,
                    max_new_tokens: int = 2) -> VerifyResult:
    """prefill+decode 双路径、四 kernel 全覆盖的端到端门禁。"""
    import torch
    from nooht.backends import backend_override
    from nooht.backends.base import KERNEL_NAMES

    try:
        from transformers import JambaForCausalLM
        torch.manual_seed(seed)
        model = JambaForCausalLM(make_tiny_jamba_config())
        model.eval()
        input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len))

        def _run():
            with torch.no_grad():
                # generate(max_new_tokens≥1) 内部必然经过：
                #   prefill（全序列）→ causal_conv1d_fn + selective_scan
                #   decode（单步）  → causal_conv1d_update + selective_state_update
                seq = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                     do_sample=False)
                fwd = model(input_ids, use_cache=False)   # 额外取 logits 做数值对照
            return seq, fwd.logits

        # ---- 阶段1：hook 可达（spy 只计数）→ 输出即官方基线 ----
        spy = _make_spy_backend()
        with backend_override(spy):
            seq_base, logits_base = _run()

        # ---- 阶段2：四 kernel 接管（probe 委托参考内核）----
        probe = _make_probe_backend()
        with backend_override(probe):
            seq_probe, logits_probe = _run()
    except Exception as exc:  # noqa: BLE001
        return VerifyResult(False, False, False, error=f"verify failed: {exc}")

    dispatched = sum(spy.queries.values()) > 0
    missing = [name for name in KERNEL_NAMES if probe.calls.get(name, 0) == 0]
    kernel_takeover = not missing
    numerics_match = bool(torch.equal(seq_base, seq_probe)) and bool(
        torch.allclose(logits_base, logits_probe, atol=1e-4))

    error = None
    if not dispatched:
        error = ("hook 未被到达：Jamba forward 从未查询 Nooht 后端。"
                 "请确认已运行 scripts/deploy_vendor.py 且 nooht 已安装")
    elif missing:
        error = f"以下 kernel 未被探针接管（prefill+decode 应覆盖全部四个）：{missing}"
    elif not numerics_match:
        error = "探针路径与官方路径数值不一致（贪心输出或 logits）"
    return VerifyResult(dispatched, kernel_takeover, numerics_match,
                        hook_queries=dict(spy.queries),
                        takeover_counts=dict(probe.calls),
                        missing_kernels=missing, error=error)