from __future__ import annotations

from typing import Dict, Type

from nooht.backends.base import NoohtBackend

_BACKEND_REGISTRY: Dict[str, NoohtBackend] = {}


def register_backend(cls: Type[NoohtBackend]) -> Type[NoohtBackend]:
    """类装饰器：实例化单例并注册。"""
    instance = cls()
    if instance.name in _BACKEND_REGISTRY:
        raise ValueError(f"duplicate backend: {instance.name}")
    _BACKEND_REGISTRY[instance.name] = instance
    return cls


def get_registered_backend(name: str) -> NoohtBackend:
    key = name.lower()
    if key not in _BACKEND_REGISTRY:
        raise KeyError(f"unknown backend: {name!r}; available: {sorted(_BACKEND_REGISTRY)}")
    return _BACKEND_REGISTRY[key]


def all_registered_backends() -> Dict[str, NoohtBackend]:
    return dict(_BACKEND_REGISTRY)