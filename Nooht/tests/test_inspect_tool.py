"""inspect_jamba.py AST 分析逻辑的自测（工具的结论本身需要测试，
否则「AST 确认」会成为新的假阳性来源）。"""
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect_jamba.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("inspect_jamba_tool", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


SYNTHETIC = '''
from some_pkg import selective_scan_fn

def helper(x):
    return selective_scan_fn(x)

class JambaMamba:
    def forward(self, hidden):
        y = selective_scan_fn(hidden)
        z = self.local_scan(hidden)

        def nested():
            return selective_scan_fn(hidden)

        return y, z, nested

def local_scan(x):
    return x
'''


def test_nested_function_attributed_to_innermost_scope(tool):
    """核心回归项：嵌套函数的调用归属最内层作用域，不误归/重复计入外层。"""
    _, _, rows = tool.parse_source(SYNTHETIC)
    by_caller = defaultdict(list)
    for r in rows:
        by_caller[r["caller"]].append(r["callee"])
    assert "selective_scan_fn" in by_caller["JambaMamba > forward() > nested()"]
    assert by_caller["JambaMamba > forward()"].count("selective_scan_fn") == 1
    assert by_caller["helper()"] == ["selective_scan_fn"]


def test_each_call_recorded_exactly_once(tool):
    _, _, rows = tool.parse_source(SYNTHETIC)
    assert sum(r["callee"] == "selective_scan_fn" for r in rows) == 3


def test_self_method_recognized(tool):
    """self.foo() 识别为 self_method，receiver=self。"""
    _, _, rows = tool.parse_source(SYNTHETIC)
    r = next(r for r in rows if r["callee"] == "local_scan")
    assert r["kind"] == "self_method"
    assert r["receiver"] == "self"
    assert "self/cls 方法调用" in r["origin"]


MODULE_ATTR_SRC = '''
import kernels

def forward_like(x):
    return kernels.selective_scan_fn(x)
'''


def test_module_attribute_call_not_misclassified(tool):
    """module.foo() 不得被标成 self method。"""
    _, _, rows = tool.parse_source(MODULE_ATTR_SRC)
    r = rows[0]
    assert r["callee"] == "selective_scan_fn"
    assert r["kind"] == "attribute_call"
    assert r["receiver"] == "kernels"
    assert "self" not in r["origin"]


def test_import_origin(tool):
    _, _, rows = tool.parse_source(SYNTHETIC)
    r = next(r for r in rows if r["caller"] == "helper()")
    assert r["origin"] == "import from some_pkg"


DECORATOR_SRC = '''
def scan_kernel():
    def wrap(fn):
        return fn
    return wrap

class M:
    @scan_kernel()
    def forward(self):
        pass
'''


def test_decorator_belongs_to_def_outer_scope(tool):
    """装饰器调用归属 def 所在的外层作用域（类体），而非方法作用域。"""
    _, _, rows = tool.parse_source(DECORATOR_SRC)
    deco_rows = [r for r in rows if r["callee"] == "scan_kernel"]
    assert len(deco_rows) == 1
    assert deco_rows[0]["caller"] == "M"


def test_conclude_positions(tool):
    _, _, rows_self = tool.parse_source(SYNTHETIC)
    assert "类方法级" in tool.conclude(rows_self)
    _, _, rows_mod = tool.parse_source(MODULE_ATTR_SRC)
    assert "receiver 属性调用" in tool.conclude(rows_mod)
    assert tool.conclude([]).startswith("[WARN]")