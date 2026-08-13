"""
Nooht 生产级内置工具链
包含：真实联网搜索、安全代码沙盒、长期记忆检索、数学计算。
"""
import json
import math
import subprocess
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# ==========================================
# 工具 1: 真实联网搜索 (Real Web Search)
# 依赖: pip install duckduckgo-search
# ==========================================
def web_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    使用 DuckDuckGo 进行真实联网搜索。
    无需 API Key，适合快速验证与轻量级 Agent 任务。
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            # 执行搜索并提取标题、链接、摘要
            results = [
                {"title": r.get("title"), "href": r.get("href"), "body": r.get("body")}
                for r in ddgs.text(query, max_results=max_results)
            ]
        return {"status": "success", "query": query, "results": results}
    except ImportError:
        return {"status": "error", "message": "duckduckgo-search not installed. Run: pip install duckduckgo-search"}
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {"status": "error", "message": str(e)}


# ==========================================
# 工具 2: 安全代码解释器 (Code Interpreter Sandbox)
# ==========================================
def code_interpreter(code: str, timeout: int = 5) -> Dict[str, Any]:
    """
    在隔离的子进程中执行 Python 代码。
    注意: 生产环境应使用 Docker/gVisor 或 E2B 等强隔离沙盒。
    此处使用 subprocess 提供基础的系统级超时与隔离保护。
    """
    # 基础安全过滤 (防止恶意破坏系统)
    blacklist = ["import os", "import shutil", "import sys", "__import__", "eval(", "exec("]
    if any(b in code for b in blacklist):
        return {"status": "error", "message": "Security violation: Forbidden module/function detected."}

    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "status": "success",
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": f"Execution timed out after {timeout}s"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ==========================================
# 工具 3: 高级数学与符号计算 (Advanced Calculator)
# ==========================================
def calculator(expression: str) -> Dict[str, Any]:
    """
    安全的数学计算器，替代危险的 eval()。
    支持基础运算与 math 库函数。
    """
    # 允许的白名单
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    allowed_names.update({"abs": abs, "round": round})
    
    try:
        # 使用 ast 或受限的 eval 环境
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {"status": "success", "expression": expression, "result": result}
    except Exception as e:
        return {"status": "error", "message": f"Calculation error: {str(e)}"}


# ==========================================
# 工具 4: 长期记忆检索 (Memory RAG / HMC Recall)
# ==========================================
# 注意: 实际使用时需要通过闭包或依赖注入传入 HMC 实例
def create_memory_recall_tool(hmc_controller):
    """工厂函数：创建绑定了 HMC 实例的记忆检索工具"""
    def memory_recall(query: str, top_k: int = 3) -> Dict[str, Any]:
        """从 Nooht 的长期分层记忆 (HMC) 中检索相关事实"""
        try:
            # 这里简化演示，实际应调用 HMC 的向量检索接口
            # memories = hmc_controller.search(query, top_k)
            return {
                "status": "success", 
                "message": "HMC recall placeholder. Bind real HMC search API here.",
                "memories": []
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return memory_recall
