import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigMigrator:
    def __init__(self, current_version: str): self.current_version = current_version
    def migrate(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        version = old_config.get("version", "v1.0")
        logger.info(f"Migrating config from {version} to {self.current_version}")
        old_config["version"] = self.current_version
        return old_config
