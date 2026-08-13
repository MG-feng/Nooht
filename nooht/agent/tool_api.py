"""Nooht Tool API v4.0 — 全能工具链注册中心"""
from .tool_registry import ToolRegistry, ToolParameter
from .tool_caller import ToolCaller
from .tools.search_matrix import SearchMatrix
from .tools.terminal import TerminalExecutor
from .tools.meta_controllers import MetaController
from .tools.session_memory import SessionMemoryManager

class ToolAPI:
    def __init__(self):
        self.registry = ToolRegistry()
        self.caller = ToolCaller(self.registry)
        self.memory = SessionMemoryManager()
        self._register_all()

    def _register_all(self):
        # 1. 搜索矩阵
        self.registry.register_function("hybrid_search", "精准与模糊混合搜索", SearchMatrix.hybrid_search, [ToolParameter("query", "string")])
        self.registry.register_function("deep_search", "全网深度多重搜索", SearchMatrix.deep_search, [ToolParameter("query", "string")])
        self.registry.register_function("generative_search", "AI原生生成式网页搜索", SearchMatrix.generative_search, [ToolParameter("query", "string")])
        
        # 2. 终端
        self.registry.register_function("terminal_exec", "执行 CMD/Bash 命令", TerminalExecutor.execute, [ToolParameter("command", "string")])
        
        # 3. 元控制器 (模型自我调节)
        self.registry.register_function("set_thinking", "设置思考强度(off/low/mid/high/max)", MetaController.set_thinking_intensity, [ToolParameter("level", "string")])
        self.registry.register_function("set_length", "设置对话长度(short/standard/long)", MetaController.set_output_length, [ToolParameter("level", "string")])
        
        # 4. 记忆管理
        self.registry.register_function("remember_global", "将当前对话浓缩存入全局长期记忆", self.memory.compress_and_store_global, [ToolParameter("summary", "string")])
        self.registry.register_function("recall_memory", "读取历史记忆", self.memory.recall, [ToolParameter("scope", "string", required=False)])

    async def process(self, text: str):
        if self.caller.TOOL_CALL_START not in text: return {"has_tool_calls": False}
        calls = self.caller.parse_tool_calls(text)
        results = [await self.caller.execute_call(c) for c in calls]
        return {"has_tool_calls": True, "results": results}
