class DeltaCompressor:
    async def compress(self, arch_ktu):
        from ..compressed_ktu import SymbolicKTU
        return SymbolicKTU(source_l2_id=arch_ktu.archived_id, absolute_confidence=0.8, existence_proof="hash")
