import threading
from typing import Any, Optional
from .config_schema import ConfigSchema

class FrozenConfigError(Exception): pass

class NoohtConfig:
    def __init__(self, schema: ConfigSchema):
        self._schema = schema; self._frozen = False; self._overrides = {}
    def freeze(self): self._frozen = True
    def get(self, path: str, default: Any = None) -> Any:
        if path in self._overrides: return self._overrides[path]
        parts = path.split("."); current = self._schema
        try:
            for part in parts:
                if hasattr(current, part): current = getattr(current, part)
                elif isinstance(current, dict): current = current[part]
                else: return default
            return current
        except: return default
    def set(self, path: str, value: Any):
        if self._frozen: raise FrozenConfigError(f"Cannot modify frozen config: '{path}'.")
        self._overrides[path] = value

class ConfigManager:
    _instance = None; _lock = threading.Lock()
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None: cls._instance = super().__new__(cls); cls._instance._initialized = False
        return cls._instance
    def __init__(self):
        if self._initialized: return
        self._initialized = True; self._config = None
