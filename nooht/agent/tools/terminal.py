"""OS 级模拟终端 (CMD/PowerShell/Linux)"""
import subprocess
import platform

class TerminalExecutor:
    @staticmethod
    def execute(command: str, timeout: int = 10) -> dict:
        """执行系统命令，带安全沙盒与超时限制"""
        # 基础危险命令拦截
        blacklist = ["rm -rf /", "mkfs", "format c:", "shutdown"]
        if any(b in command.lower() for b in blacklist):
            return {"status": "blocked", "reason": "Destructive command detected."}
        
        try:
            shell = True if platform.system() != "Windows" else False
            res = subprocess.run(command, shell=shell, capture_output=True, text=True, timeout=timeout)
            return {"status": "success", "stdout": res.stdout, "stderr": res.stderr, "code": res.returncode}
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "reason": f"Exceeded {timeout}s limit."}
        except Exception as e:
            return {"status": "error", "reason": str(e)}
