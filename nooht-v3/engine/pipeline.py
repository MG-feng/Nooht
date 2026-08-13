from typing import Dict, Any, List, Optional
from enum import Enum
class StageType(Enum): SEQUENTIAL="sequential"
class PipelineStage:
    def __init__(self, name: str, module_name: str, stage_type: StageType = StageType.SEQUENTIAL): self.name = name; self.module_name = module_name
class Pipeline:
    def __init__(self, name: str): self.name = name; self.stages = []
    def add_stage(self, stage: PipelineStage) -> "Pipeline": self.stages.append(stage); return self
class PipelineExecutor:
    def __init__(self, registry): self.registry = registry
    async def execute(self, pipeline: Pipeline, inputs: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(inputs)
        for stage in pipeline.stages:
            module = self.registry._modules.get(stage.module_name)
            if not module: raise RuntimeError(f"Module not found: {stage.module_name}")
            data = await module.execute(data)
        return data
