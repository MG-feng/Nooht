"""漂移守卫：modular_jamba.py（唯一事实源）与 modeling_jamba.py（生成产物）
的 Nooht hook 必须一致。生成器重跑导致漂移时，本测试先行失败。"""
import ast
from pathlib import Path

import pytest

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "jamba"
KERNEL_TARGETS = ("mamba_selective_scan", "mamba_selective_state_update",
                  "causal_conv1d_fn", "causal_conv1d_update")


def _extract(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hook_src, rebinds, points = None, [], None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_nooht_hook":
            hook_src = ast.unparse(node)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id in KERNEL_TARGETS:
                rebinds.append(ast.unparse(node))
            if isinstance(t, ast.Name) and t.id == "NOOHT_KERNEL_DISPATCH_POINTS":
                points = ast.unparse(node.value)
    return hook_src, sorted(rebinds), points


@pytest.mark.skipif(not (VENDOR / "modular_jamba.py").exists(),
                    reason="vendor/jamba 未放置")
def test_modeling_and_modular_hook_in_sync():
    a = _extract(VENDOR / "modular_jamba.py")
    b = _extract(VENDOR / "modeling_jamba.py")
    assert a[0] is not None and b[0] is not None, "缺少 _nooht_hook 定义"
    assert a == b, "Nooht hook 漂移：重新执行同步流程保持生成产物一致"