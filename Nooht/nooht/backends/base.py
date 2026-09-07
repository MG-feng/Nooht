"""Nooht 多硬件后端适配层 —— 基类与接口契约。"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict

import torch

#: 已在 vendored Jamba 中接线的所有 kernel 派发点。
#: 必须与 modeling_jamba.NOOHT_KERNEL_DISPATCH_POINTS 一致，
#: 漂移由 tests/test_jamba_dispatch_smoke.py::test_dispatch_point_lists_in_sync 拦截。
KERNEL_NAMES = ("selective_scan", "selective_state_update",
                "causal_conv1d_fn", "causal_conv1d_update")


@dataclass
class BackendInfo:
    name: str
    device: str
    available: bool
    extra: Dict[str, Any] = field(default_factory=dict)


class NoohtBackend(abc.ABC):
    """单个硬件后端的抽象基类。"""

    name: str = "abstract"
    #: 优先级，数值越小越优先：CUDA(10) > ROCm(20) > XLA(30) > oneAPI(40) > CPU(999)
    priority: int = 100

    # ---------- 能力探测 ----------
    @abc.abstractmethod
    def is_available(self) -> bool:
        """当前运行环境是否支持本后端。"""

    @abc.abstractmethod
    def default_device(self) -> torch.device:
        """本后端默认设备。"""

    def device_count(self) -> int:
        return 1 if self.is_available() else 0

    # ---------- 生命周期 ----------
    def setup(self) -> None:
        """一次性初始化（幂等）。默认空实现。"""

    def synchronize(self) -> None:
        """设备同步，用于性能基准计时。"""
        dev = self.default_device()
        if dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    # ---------- kernel 能力声明与默认实现 ----------
    def overrides_kernel(self, name: str) -> bool:
        """能力声明：该后端是否为指定 kernel 提供自己的实现（显式声明，非反射）。

        契约（由 tests/test_backend_api_contract.py 强制）：
        覆写某 kernel 的后端必须同时提供同名方法，且签名与官方 kernel 一致。
        """
        return False

    @property
    def overrides_selective_scan(self) -> bool:
        """兼容 PR#1 早期接口。"""
        return self.overrides_kernel("selective_scan")

    def selective_scan(self, hidden_states, dt, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False,
                       return_last_state=False, use_mambapy=False,
                       use_associative_scan=False, **kwargs):
        """默认 = 纯 PyTorch 参考实现，签名对齐官方 mamba_selective_scan。"""
        from nooht.backends.kernels import reference_selective_scan
        return reference_selective_scan(
            hidden_states, dt, A, B, C, D=D, z=z, delta_bias=delta_bias,
            delta_softplus=delta_softplus, return_last_state=return_last_state,
            use_mambapy=use_mambapy, use_associative_scan=use_associative_scan,
            **kwargs)

    def selective_state_update(self, state, hidden_states, dt, A, B, C,
                               D=None, dt_bias=None, dt_softplus=False,
                               z=None, **kwargs):
        from nooht.backends.kernels import reference_selective_state_update
        return reference_selective_state_update(
            state, hidden_states, dt, A, B, C, D=D, dt_bias=dt_bias,
            dt_softplus=dt_softplus, z=z, **kwargs)

    def causal_conv1d_fn(self, hidden_states, weight, bias=None,
                         activation=None, **kwargs):
        from nooht.backends.kernels import reference_causal_conv1d_fn
        return reference_causal_conv1d_fn(hidden_states, weight, bias=bias,
                                          activation=activation, **kwargs)

    def causal_conv1d_update(self, hidden_states, conv_state, weight,
                             bias=None, activation=None):
        from nooht.backends.kernels import reference_causal_conv1d_update
        return reference_causal_conv1d_update(hidden_states, conv_state, weight,
                                              bias=bias, activation=activation)

    # ---------- 信息 ----------
    def info(self) -> BackendInfo:
        return BackendInfo(name=self.name, device=str(self.default_device()),
                           available=self.is_available())

    def __repr__(self) -> str:
        return (f"<NoohtBackend {self.name} priority={self.priority} "
                f"available={self.is_available()}>")