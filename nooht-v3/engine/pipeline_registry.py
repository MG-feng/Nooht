from typing import Dict, Optional
from .pipeline import Pipeline
class PipelineRegistry:
    def __init__(self): self._pipelines = {}
    def register(self, pipeline: Pipeline) -> bool: self._pipelines[pipeline.name] = pipeline; return True
    def get(self, name: str) -> Optional[Pipeline]: return self._pipelines.get(name)
    def clear(self): self._pipelines.clear()
