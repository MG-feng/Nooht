from typing import Dict, Any, Tuple, Callable
import logging
logger = logging.getLogger(__name__)
_MIGRATIONS: Dict[Tuple[str, str], Callable] = {}
def register_migration(f, t):
    def d(func): _MIGRATIONS[(f, t)] = func; return func
    return d

class ConfigMigrator:
    CURRENT_VERSION = "1.0.0"; CHAIN = ["0.1.0", "0.9.0", "1.0.0"]
    def migrate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        version = config.get("version", "0.1.0")
        if version == self.CURRENT_VERSION: return config
        try: idx = self.CHAIN.index(version)
        except: idx = 0
        for i in range(idx, len(self.CHAIN) - 1):
            frm, to = self.CHAIN[i], self.CHAIN[i+1]
            if (frm, to) in _MIGRATIONS: config = _MIGRATIONS[(frm, to)](config)
        config["version"] = self.CURRENT_VERSION
        return config

@register_migration("0.1.0", "0.9.0")
def _m1(config): 
    if "gpu_workers" in config: config["workers"] = config.pop("gpu_workers")
    return config

@register_migration("0.9.0", "1.0.0")
def _m2(config): 
    config.setdefault("nrt", {}).setdefault("scheduler", {"levels": 6})
    return config
