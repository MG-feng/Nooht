from dataclasses import dataclass, field
import uuid
@dataclass
class MemoryBlock:
    block_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8]); size_mb: int = 0; device_id: int = -1
class UnifiedAllocator:
    def __init__(self, device_manager): self.dm = device_manager
    def allocate(self, size_mb: int, device_id: int = -1) -> MemoryBlock:
        if self.dm.allocate(device_id, size_mb): return MemoryBlock(size_mb=size_mb, device_id=device_id)
        raise MemoryError("Allocation failed")
    def free(self, block: MemoryBlock): self.dm.free(block.device_id, block.size_mb)
