from .compressed_ktu import MemoryLevel
from typing import Callable, Any, Dict
import asyncio
class IllegalMigrationError(Exception): pass
class MigrationManager:
    _VALID = {MemoryLevel.L0_ACTIVE: [MemoryLevel.L1_COMPRESSED], MemoryLevel.L1_COMPRESSED: [MemoryLevel.L2_ARCHIVAL], MemoryLevel.L2_ARCHIVAL: [MemoryLevel.L3_SYMBOLIC]}
    def __init__(self, config: Dict[str, Any]): self._sem = asyncio.Semaphore(config.get("max_concurrent_migrations", 2))
    def validate(self, curr: MemoryLevel, targ: MemoryLevel):
        if targ not in self._VALID.get(curr, []): raise IllegalMigrationError(f"Forbidden {curr.value}->{targ.value}")
    async def migrate(self, ktu_id: str, curr: MemoryLevel, targ: MemoryLevel, data: Any, comp: Callable) -> Any:
        self.validate(curr, targ)
        async with self._sem: return await comp(data)
