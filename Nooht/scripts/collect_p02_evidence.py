#!/usr/bin/env python
"""P0-2 证据采集（钉版本环境中执行一次）。

流程：
0. 自动执行 scripts/deploy_vendor.py（部署 vendored Jamba，门禁前提）
1. 环境版本
2. scripts/inspect_jamba.py 实文输出（调用链 + 结论）
3. verify_dispatch() 完整 VerifyResult（含四项 takeover_counts）
4. pytest tests -v 结果摘要
产出：docs/notes/p02_evidence.md

用法：
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    pip install transformers==4.46.3 pytest==8.3.3
    pip install -e .
    python scripts/collect_p02_evidence.py
然后把 docs/notes/p02_evidence.md 贴回审核侧。
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list) -> tuple:
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    import torch
    import transformers

    repo = Path(__file__).resolve().parents[1]
    out = repo / "docs" / "notes" / "p02_evidence.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# P0-2 执行证据",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    # ---- 第0步：部署 vendored Jamba（门禁前提）----
    code, text = run([sys.executable, str(repo / "scripts" / "deploy_vendor.py")])
    lines += ["## 0. deploy_vendor", "```", text.strip(), "```",
              f"(exit code: {code})", ""]
    if code != 0:
        lines += ["**GATE: FAIL（部署失败，终止证据采集）**", ""]
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"evidence written: {out}")
        return 1

    # ---- 第1步：环境版本 ----
    lines += [
        "## 1. 环境版本",
        f"- python: {sys.version.split()[0]}",
        f"- torch: {torch.__version__}",
        f"- transformers: {transformers.__version__}",
        "",
    ]

    # ---- 第2步：inspect_jamba 实文 ----
    lines += ["## 2. inspect_jamba.py 实文输出", "```"]
    code, text = run([sys.executable, str(repo / "scripts" / "inspect_jamba.py")])
    lines += [text.strip(), "```", f"(exit code: {code})", ""]

    # ---- 第3步：verify_dispatch ----
    from nooht.modeling.verify import verify_dispatch  # 部署后再导入，避免提前加载
    lines += ["## 3. verify_dispatch() 实文输出", "```"]
    gate = False
    try:
        result = verify_dispatch()
        lines += [repr(result), "```", ""]
        gate = (result.dispatched and result.kernel_takeover
                and result.numerics_match and not result.missing_kernels)
        lines += [
            f"- dispatched={result.dispatched}",
            f"- kernel_takeover={result.kernel_takeover}",
            f"- numerics_match={result.numerics_match}",
            f"- missing_kernels={result.missing_kernels}",
            f"- takeover_counts={result.takeover_counts}",
            f"- hook_queries={result.hook_queries}",
            f"- **GATE: {'PASS' if gate else 'FAIL'}**",
            "",
        ]
    except Exception as exc:  # noqa: BLE001
        lines += [f"verify_dispatch raised: {exc!r}", "```", "",
                  "**GATE: FAIL**", ""]

    # ---- 第4步：pytest ----
    lines += ["## 4. pytest tests -v 结果（尾部摘要）", "```"]
    code, text = run([sys.executable, "-m", "pytest", "tests", "-v", "--tb=short"])
    lines += ["\n".join(text.strip().splitlines()[-40:]), "```",
              f"(exit code: {code})", ""]

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"evidence written: {out}")
    print(f"GATE: {'PASS' if gate and code == 0 else 'FAIL'}")
    return 0 if gate and code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())