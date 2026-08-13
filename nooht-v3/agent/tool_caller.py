import json
import re
from typing import Dict, Any, List
from .tool_registry import ToolRegistry

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_RESULT_START = "<tool_result>"
TOOL_RESULT_END = "</tool_result>"

class ToolCaller:
    def __init__(self, registry: ToolRegistry, max_calls: int = 5):
        self.registry = registry; self.max_calls = max_calls

    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        pattern = re.escape(TOOL_CALL_START) + r"(.*?)" + re.escape(TOOL_CALL_END)
        matches = re.findall(pattern, text, re.DOTALL)
        calls = []
        for match in matches:
            try:
                call_data = json.loads(match.strip())
                if "name" in call_data: calls.append(call_data)
            except: pass
        return calls

    async def execute_call(self, call: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.registry.get(call.get("name", ""))
        if not tool or not tool.handler: return {"error": "Tool not found"}
        try:
            if tool.is_async: result = await tool.handler(**call.get("arguments", {}))
            else: result = tool.handler(**call.get("arguments", {}))
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}
