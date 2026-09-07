"""后端 API 完整性契约：能力声明必须与实现一致，
不允许「声明 overrides=True 但方法缺失」在运行时才变成 AttributeError。"""
from nooht.backends import list_backends
from nooht.backends.base import KERNEL_NAMES
from nooht.modeling.verify import _make_probe_backend, _make_spy_backend


def _all_backends_under_test():
    return list(list_backends(only_available=False)) + [
        _make_probe_backend(), _make_spy_backend(),
    ]


def test_overrides_kernel_implies_callable_method():
    for backend in _all_backends_under_test():
        for name in KERNEL_NAMES:
            if backend.overrides_kernel(name):
                assert callable(getattr(backend, name, None)), (
                    f"{backend.name} 声明 overrides_kernel({name!r})=True "
                    f"但同名方法缺失或不可调用"
                )


def test_default_backends_declare_no_override():
    """冻结本阶段行为：全部目标后端 + CPU 参考后端默认不接管任何 kernel。"""
    for backend in list_backends(only_available=False):
        for name in KERNEL_NAMES:
            assert backend.overrides_kernel(name) is False