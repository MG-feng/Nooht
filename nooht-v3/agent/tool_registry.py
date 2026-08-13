from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

@dataclass
class ToolParameter:
    name: str
    type: str
    description: str = ""
    required: bool = True

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable] = None
    is_async: bool = False

class ToolRegistry:
    def __init__(self): self._tools = {}
    def register(self, tool: ToolDefinition): self._tools[tool.name] = tool
    def register_function(self, name, desc, handler, params=None, is_async=False):
        self.register(ToolDefinition(name, desc, params or [], handler, is_async))
    def get(self, name: str) -> Optional[ToolDefinition]: return self._tools.get(name)
