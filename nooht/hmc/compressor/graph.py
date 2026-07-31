class GraphCompressor:
    async def compress(self, comp_ktu):
        from ..compressed_ktu import ArchivedKTU
        return ArchivedKTU(source_l1_id=comp_ktu.compressed_id, graph_edges={"rel": "ref"})
