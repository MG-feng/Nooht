from dataclasses import dataclass, field
from enum import Enum
import time, uuid
class MemoryLevel(Enum): L0_ACTIVE = 0; L1_COMPRESSED = 1; L2_ARCHIVAL = 2; L3_SYMBOLIC = 3
@dataclass(frozen=True)
class CompressedKTU: compressed_id: str = field(default_factory=lambda: str(uuid.uuid4())); source_l0_id: str = ""; semantic_vector: list = field(default_factory=list); confidence: float = 0.0; importance: float = 0.5; compressed_at: float = field(default_factory=time.time)
@dataclass(frozen=True)
class ArchivedKTU: archived_id: str = field(default_factory=lambda: str(uuid.uuid4())); source_l1_id: str = ""; graph_edges: dict = field(default_factory=dict); archived_at: float = field(default_factory=time.time)
@dataclass(frozen=True)
class SymbolicKTU: symbol_id: str = field(default_factory=lambda: str(uuid.uuid4())); source_l2_id: str = ""; absolute_confidence: float = 0.0; existence_proof: str = ""
