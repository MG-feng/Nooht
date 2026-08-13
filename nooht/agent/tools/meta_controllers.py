"""元控制器：动态调整模型自身的思考强度与对话长度"""
class MetaController:
    # 思考强度映射: (max_steps, energy_threshold)
    THINKING_PROFILES = {
        "off": (0, 1.0),    # 直接输出
        "low": (2, 0.8),    # 梳理方向
        "mid": (5, 0.5),    # 反复思考，总结
        "high": (15, 0.2),  # 深度推演
        "max": (50, 0.05)   # 终极方案推演
    }
    
    # 对话长度映射: (max_tokens, temperature)
    LENGTH_PROFILES = {
        "short": (128, 0.3),
        "standard": (512, 0.7),
        "long": (2048, 0.9)
    }

    @staticmethod
    def set_thinking_intensity(level: str) -> dict:
        """供模型调用，动态修改 ThinkingModule 的参数"""
        profile = MetaController.THINKING_PROFILES.get(level, MetaController.THINKING_PROFILES["mid"])
        return {"action": "thinking_updated", "max_steps": profile[0], "threshold": profile[1]}

    @staticmethod
    def set_output_length(level: str) -> dict:
        """供模型调用，动态修改生成策略"""
        profile = MetaController.LENGTH_PROFILES.get(level, MetaController.LENGTH_PROFILES["standard"])
        return {"action": "length_updated", "max_tokens": profile[0], "temperature": profile[1]}
