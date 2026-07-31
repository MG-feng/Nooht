import asyncio, random
from collections import OrderedDict
from typing import Dict, Any, List
from .compressed_ktu import MemoryLevel
from .memory_ranker import MemoryRanker
from .migration_manager import MigrationManager
from .compressor import SemanticCompressor, GraphCompressor, DeltaCompressor
import logging
logger = logging.getLogger(__name__)

class HMCController:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}; self._l0 = OrderedDict(); self._l0_max = self.config.get("l0_max_bytes", 16*(1024**3)); self._cur_l0 = 0
        self._l1 = {}; self._l1_max = 100; self._ranker = MemoryRanker(self.config); self._mig = MigrationManager(self.config)
        self._sem_comp = SemanticCompressor(); self._graph_comp = GraphCompressor(); self._delta_comp = DeltaCompressor()
        self._ev_lock = asyncio.Lock(); self._pending_ev = set()
    async def store_l0(self, ktu):
        ktu_b = 256 + len(ktu.fused_vector or []) * 8
        if ktu_b > self._l0_max: raise MemoryError(f"KTU size {ktu_b} exceeds L0 capacity {self._l0_max}")
        if self._cur_l0 + ktu_b > self._l0_max: await self._evict_l0()
        self._l0[ktu.ktu_id] = ktu; self._cur_l0 += ktu_b; self._ranker.record_access(ktu.ktu_id)
    async def _schedule_ev(self, coro):
        task = asyncio.create_task(coro); self._pending_ev.add(task); task.add_done_callback(self._pending_ev.discard)
    def _sample_and_rank(self) -> List:
        items = list(self._l0.items())
        if len(items) > 1000: items = random.sample(items, 1000)
        scores = [(self._ranker.compute_score(kid, k.confidence, k.importance), kid) for kid, k in items]
        scores.sort(key=lambda x: x[0]); return [s[1] for s in scores]
    async def _evict_l0(self):
        async with self._ev_lock:
            if self._cur_l0 <= self._l0_max * 0.9: return
            need = self._cur_l0 - self._l0_max * 0.7; freed = 0
            for kid in self._sample_and_rank():
                if freed >= need: break
                ktu = self._l0[kid]; k_b = 256 + len(ktu.fused_vector or []) * 8
                comp = await self._mig.migrate(kid, MemoryLevel.L0_ACTIVE, MemoryLevel.L1_COMPRESSED, ktu, self._sem_comp.compress)
                self._l1[comp.compressed_id] = comp
                del self._l0[kid]; self._cur_l0 -= k_b; freed += k_b
            if len(self._l1) > self._l1_max: await self._schedule_ev(self._evict_l1())
    async def _evict_l1(self):
        if len(self._l1) <= self._l1_max * 0.9: return
        target = len(self._l1) - int(self._l1_max * 0.7)
        for cid, comp in list(self._l1.items())[:target]:
            await self._mig.migrate(cid, MemoryLevel.L1_COMPRESSED, MemoryLevel.L2_ARCHIVAL, comp, self._graph_comp.compress)
            del self._l1[cid]
