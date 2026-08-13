from typing import Dict, Any
from .tool_registry import ToolRegistry, ToolParameter
from .tool_caller import ToolCaller

class ToolAPI:
    def __init__(self):
        self.registry = ToolRegistry()
        self.caller = ToolCaller(self.registry)
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        self.registry.register_function("calculator", "Evaluate math", lambda expression: {"result": eval(expression)}, [ToolParameter("expression", "string")])
        self.registry.register_function("web_search", "Search web", lambda query: {"results": [f"Search: {query}"]}, [ToolParameter("query", "string")])

    async def process_model_output(self, text: str) -> Dict[str, Any]:
        if self.caller.TOOL_CALL_START not in text: return {"has_tool_calls": False}
        calls = self.caller.parse_tool_calls(text)
        results = [await self.caller.execute_call(c) for c in calls[:self.caller.max_calls]]
        return {"has_tool_calls": True, "results": results}
