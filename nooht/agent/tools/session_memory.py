"""多对话记忆管理器 (全局与项目级)"""
import time
import json
import os

class SessionMemoryManager:
    def __init__(self, storage_dir="./.nooht_memory"):
        os.makedirs(storage_dir, exist_ok=True)
        self.global_db = os.path.join(storage_dir, "global.json")
        self.project_db = os.path.join(storage_dir, "project.json")
        self._load()

    def _load(self):
        self.global_mem = self._read_json(self.global_db)
        self.project_mem = self._read_json(self.project_db)

    def _read_json(self, path):
        if os.path.exists(path):
            with open(path, "r") as f: return json.load(f)
        return []

    def _save_json(self, path, data):
        with open(path, "w") as f: json.dump(data, f)

    def compress_and_store_global(self, summary: str, ttl_days: int = 365):
        """全局记忆：浓缩并存盘，带 TTL 淘汰机制"""
        entry = {"summary": summary, "timestamp": time.time(), "ttl": ttl_days * 86400}
        self.global_mem.append(entry)
        # 淘汰过期记忆
        now = time.time()
        self.global_mem = [m for m in self.global_mem if (now - m["timestamp"]) < m["ttl"]]
        self._save_json(self.global_db, self.global_mem)
        return {"status": "stored_global", "active_memories": len(self.global_mem)}

    def store_project(self, project_name: str, context: str):
        """项目记忆：特定范围的局部记忆"""
        self.project_mem.append({"project": project_name, "context": context, "timestamp": time.time()})
        self._save_json(self.project_db, self.project_mem)
        return {"status": "stored_project"}

    def recall(self, scope: str = "global", limit: int = 5) -> dict:
        """读取记忆"""
        mem = self.global_mem if scope == "global" else self.project_mem
        return {"memories": mem[-limit:]}
