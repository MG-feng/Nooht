class SemanticCompressor:
    async def compress(self, ktu):
        from ..compressed_ktu import CompressedKTU
        return CompressedKTU(source_l0_id=ktu.ktu_id, semantic_vector=ktu.fused_vector or [], confidence=ktu.confidence)
