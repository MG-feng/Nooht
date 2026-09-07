"""把 vendored Jamba 文件部署进已安装的 transformers 包。

这是 P0-2 门禁成立的前提：只有 vendored modeling_jamba.py 生效，
Jamba forward 才会经过 Nooht 派发 hook。

用法:
    python scripts/deploy_vendor.py            # 部署
    python scripts/deploy_vendor.py --check    # 只校验已部署版本与 vendor 一致
还原官方版：重装/覆盖 transformers 即可。
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


import argparse
import filecmp
import shutil
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "jamba"
FILES = ("__init__.py", "configuration_jamba.py",
         "modeling_jamba.py", "modular_jamba.py")


def _target_dir() -> Path:
    import transformers.models.jamba as jamba_pkg
    return Path(jamba_pkg.__path__[0])


def deploy() -> int:
    if not VENDOR.exists():
        print(f"ERROR: {VENDOR} 不存在，请先组装 vendor/jamba", file=sys.stderr)
        return 1
    target = _target_dir()
    for name in FILES:
        src = VENDOR / name
        if not src.exists():
            print(f"ERROR: {src} 缺失", file=sys.stderr)
            return 1
        dst = target / name
        shutil.copy2(src, dst)
        print(f"deployed: {name} -> {dst}")
    cache = target / "__pycache__"
    if cache.exists():
        shutil.rmtree(cache)
        print(f"cleared: {cache}")
    print("[OK] vendored Jamba 部署完成")
    return 0


def check() -> int:
    target = _target_dir()
    ok = True
    for name in FILES:
        src, dst = VENDOR / name, target / name
        if not dst.exists():
            print(f"[missing] {dst}")
            ok = False
        elif not filecmp.cmp(src, dst, shallow=False):
            print(f"[drift]   {name}: 已安装版本与 vendor/jamba 不一致")
            ok = False
        else:
            print(f"[ok]      {name}")
    print("[OK] 部署版本与 vendor 一致" if ok else "[FAIL] 需重新执行 deploy")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    return check() if args.check else deploy()


if __name__ == "__main__":
    raise SystemExit(main())