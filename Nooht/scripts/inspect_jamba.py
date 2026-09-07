"""确认当前 transformers 版本 modeling_jamba 的真实 scan 调用链（里程碑1 必跑）。

定位（审核收紧）：本工具是静态候选发现器，不是完整证明工具。
间接调用（如不含关键词的中转函数）可能漏检；最终证明必须是运行时
verify_dispatch()（P0-2 唯一权威判据）。

能力：
1. 每个 Call 恰好记录一次，归属最内层作用域；
2. 装饰器表达式归属 def 所在外层作用域；
3. Attribute 调用区分 receiver：self/cls → self_method；其他 → attribute_call；
4. 核心解析（parse_source）与结论（conclude）均为纯函数，见 tests/test_inspect_tool.py。

用法: python scripts/inspect_jamba.py [--json] [--module MODULE]
产物: 调用链表格，存档到 docs/notes/jamba_scan_callchain.md
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


import argparse
import ast
import importlib
import inspect
import json
import textwrap

KEYWORDS = ("scan", "ssm", "mamba", "conv1d", "discret", "state_update")


def is_kernel_name(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in KEYWORDS)


def _collect_definitions(tree: ast.Module):
    module_funcs, imported = {}, {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module_funcs[node.name] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported[alias.asname or alias.name] = node.module or "?"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported[alias.asname or alias.name.split(".")[0]] = alias.name
    return module_funcs, imported


def _iter_no_nested_defs(node):
    """ast.walk 变体：嵌套函数/类定义只 yield 节点本身，不展开子树。"""
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        if cur is not node and isinstance(
                cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for child in ast.iter_child_nodes(cur):
            stack.append(child)


def _receiver_name(value_node):
    """属性调用接收者；无法静态解析时返回 None。"""
    if isinstance(value_node, ast.Name):
        return value_node.id
    if isinstance(value_node, ast.Attribute):
        base = _receiver_name(value_node.value)
        return f"{base}.{value_node.attr}" if base else None
    return None


def parse_source(src: str):
    """解析源码字符串 → (module_funcs, imported, call_chain rows)。"""
    tree = ast.parse(textwrap.dedent(src))
    module_funcs, imported = _collect_definitions(tree)
    rows = []

    def origin_of(fname: str, kind: str, receiver) -> str:
        if kind == "self_method":
            return ("self/cls 方法调用 ← 模块级 patch 不可达，"
                    "需类方法级 patch")
        if kind == "attribute_call":
            return (f"receiver={receiver!r} 属性调用 ← 需核实 receiver 是否为模块对象；"
                    "若是，可在其所属模块做 patch")
        if fname in imported:
            return f"import from {imported[fname]}"
        if fname in module_funcs:
            return f"module-level def @L{module_funcs[fname]}"
        return "local/unknown"

    def record(call: ast.Call, scope):
        receiver = None
        if isinstance(call.func, ast.Name):
            fname, kind = call.func.id, "name"
        elif isinstance(call.func, ast.Attribute):
            fname = call.func.attr
            receiver = _receiver_name(call.func.value)
            kind = "self_method" if receiver in ("self", "cls") else "attribute_call"
        else:
            return
        if is_kernel_name(fname):
            rows.append({"caller": " > ".join(scope) or "<module>",
                         "callee": fname, "call_line": call.lineno,
                         "receiver": receiver, "kind": kind,
                         "origin": origin_of(fname, kind, receiver)})

    def scan(node: ast.AST, scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inner = scope + [f"{node.name}()"]
            for deco in node.decorator_list:
                for sub in _iter_no_nested_defs(deco):
                    if isinstance(sub, ast.Call):
                        record(sub, scope)
            for stmt in node.body:
                scan(stmt, inner)
        elif isinstance(node, ast.ClassDef):
            inner = scope + [node.name]
            for deco in node.decorator_list:
                for sub in _iter_no_nested_defs(deco):
                    if isinstance(sub, ast.Call):
                        record(sub, scope)
            for stmt in node.body:
                scan(stmt, inner)
        else:
            for sub in _iter_no_nested_defs(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef)):
                    scan(sub, scope)
                elif isinstance(sub, ast.Call):
                    record(sub, scope)

    scan(tree, [])
    return module_funcs, imported, rows


#: 真正构成 scan 派发点的 callee 名；其他名字的 self 方法调用（如
#: self.mamba()——那只是调用 mixer 子模块）不影响模块级 patch 结论。
_DISPATCH_CALLEES = frozenset((
    "selective_scan", "selective_state_update",
    "causal_conv1d_fn", "causal_conv1d_update",
    "selective_scan_fn", "mamba_selective_scan", "mamba_selective_state_update",
    "local_scan",  # test placeholder for test_inspect_tool.py
))


def conclude(rows) -> str:
    """静态候选发现结论（纯函数，可测试）。"""
    if not rows:
        return ("[WARN] 未发现 scan/mamba 相关调用：可能关键词漏检或存在间接调用，"
                "需人工扩充 KEYWORDS 或对候选做 callee 展开复查")
    if any(r["kind"] == "self_method" and r["callee"] in _DISPATCH_CALLEES for r in rows):
        return ("[WARN] 存在 self/cls 方法级 scan 调用：模块符号 patch 不可达，"
                "需增加类方法级候选")
    if any(r["kind"] == "attribute_call" for r in rows):
        return ("[WARN] 存在 receiver 属性调用：需确认 receiver 是否为模块对象，"
                "并在其所属模块上做 patch（静态分析无法自证）")
    return ("[OK] 候选均为模块级：模块符号 patch 可能可行；"
            "最终结论以 verify_dispatch() 实际通过为准")


def analyze(module_name: str = "transformers.models.jamba.modeling_jamba"):
    mj = importlib.import_module(module_name)
    module_funcs, imported, rows = parse_source(inspect.getsource(mj))
    kernel_defs = {n: f"@L{l}" for n, l in module_funcs.items() if is_kernel_name(n)}
    kernel_imports = {n: m for n, m in imported.items() if is_kernel_name(n)}
    return mj.__file__, kernel_defs, kernel_imports, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--module",
                        default="transformers.models.jamba.modeling_jamba",
                        help="被分析模块（可指向 vendored 版本）")
    args = parser.parse_args()

    path, defs, imports, rows = analyze(args.module)
    if args.json:
        print(json.dumps({"source": path, "kernel_defs": defs,
                          "kernel_imports": imports, "call_chain": rows,
                          "conclusion": conclude(rows)},
                         indent=2, ensure_ascii=False))
        return

    print(f"# modeling_jamba scan 调用链确认\nsource: {path}\n")
    print("## 模块级 kernel 定义")
    print("\n".join(f"- {k} {v}" for k, v in defs.items()) or "- 无")
    print("\n## kernel 相关 import")
    print("\n".join(f"- {k} ← {v}" for k, v in imports.items()) or "- 无")
    print("\n## 调用链表")
    if not rows:
        print("- 未发现 scan/mamba 相关调用")
    for r in rows:
        print(f"- {r['caller']}  →  {r['callee']}()  "
              f"[kind={r['kind']} receiver={r['receiver']}]  "
              f"[{r['origin']}]  @L{r['call_line']}")
    print(f"\n## 结论\n{conclude(rows)}")


if __name__ == "__main__":
    main()