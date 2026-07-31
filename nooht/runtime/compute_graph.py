from typing import List, Dict, Any
import logging
logger = logging.getLogger(__name__)

class GraphNode:
    def __init__(self, node_id: str, inputs: List[str] = None): self.node_id = node_id; self.inputs = inputs or []
    async def forward(self, tensors: List[Any], ctx: Dict[str, Any]) -> Any: raise NotImplementedError

class ComputeGraph:
    def __init__(self): self.nodes = {}; self.edges = {}
    def add_node(self, node: GraphNode):
        if node.node_id in self.nodes: return
        if node.node_id in node.inputs: raise ValueError("Self-loop detected")
        self.nodes[node.node_id] = node; self.edges[node.node_id] = []
        for i in node.inputs:
            if i in self.nodes: self.edges[i].append(node.node_id)
            else: logger.warning(f"Dangling input {i}")

class GraphExecutor:
    def __init__(self, tm: Any, dm: Any): self.tm = tm; self.dm = dm
    async def execute(self, graph: ComputeGraph, inputs: Dict[str, Any]) -> Dict[str, Any]:
        in_deg = {u: 0 for u in graph.nodes}
        for u in graph.nodes:
            for v in graph.edges.get(u, []): in_deg[v] += 1
        q = [u for u in in_deg if in_deg[u] == 0]; order = []
        while q:
            u = q.pop(0); order.append(u)
            for v in graph.edges.get(u, []):
                in_deg[v] -= 1
                if in_deg[v] == 0: q.append(v)
        if len(order) != len(graph.nodes): raise ValueError("Cycle detected")
        ctx = inputs.copy()
        for nid in order:
            node = graph.nodes[nid]; t = [ctx[i] for i in node.inputs]
            ctx[nid] = await node.forward(t, {"tm": self.tm, "dm": self.dm})
        return ctx
