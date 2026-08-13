import math, time
from typing import Dict, Any
class MemoryRanker:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}; self._decay_half_life = 3600; self._access_counter = {}; self._last_access_time = {}
    def record_access(self, ktu_id: str):
        self._access_counter[ktu_id] = self._access_counter.get(ktu_id, 0) + 1; self._last_access_time[ktu_id] = time.time()
    def compute_score(self, ktu_id: str, conf: float, imp: float) -> float:
        acc_c = self._access_counter.get(ktu_id, 0); max_acc = max(self._access_counter.values()) if self._access_counter else 1
        acc_score = min(acc_c / max(max_acc, 1), 1.0)
        decay = 1.0 - math.exp(-(time.time() - self._last_access_time.get(ktu_id, time.time())) / self._decay_half_life)
        total = 0.25 * acc_score + 0.20 * conf + 0.20 * imp - 0.10 * decay
        return max(0.0, min(1.0, total))
