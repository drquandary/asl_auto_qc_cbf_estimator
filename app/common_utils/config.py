"""Common configuration management for BSC AI Apps."""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field
from .logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class AppConfig:
    """Base configuration class for BSC AI Apps."""
    app_name: str = "BSC AI App"
    debug: bool = False
    host: str = "localhost"
    port: int = 8000
    log_level: str = "INFO"
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    temp_dir: Path = field(default_factory=lambda: Path("./temp"))

    def __post_init__(self):
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

def load_config_from_env(config_class: type = AppConfig) -> Any:
    """Load configuration from environment variables."""
    config_dict = {}

    for field_name, field_info in config_class.__dataclass_fields__.items():
        env_var = f"APP_{field_name.upper()}"
        env_value = os.getenv(env_var)

        if env_value is not None:
            # Convert string values to appropriate types
            if field_info.type == bool:
                config_dict[field_name] = env_value.lower() in ('true', '1', 'yes')
            elif field_info.type == int:
                try:
                    config_dict[field_name] = int(env_value)
                except ValueError:
                    logger.warning(f"Invalid integer value for {env_var}: {env_value}")
            elif field_info.type == Path:
                config_dict[field_name] = Path(env_value)
            else:
                config_dict[field_name] = env_value

    return config_class(**config_dict)

def load_config_from_file(config_file: Path, config_class: type = AppConfig) -> Any:
    """Load configuration from JSON file."""
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_file}")
        return config_class()

    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)

        # Convert string paths to Path objects
        for key, value in config_data.items():
            if key.endswith('_dir') and isinstance(value, str):
                config_data[key] = Path(value)

        return config_class(**config_data)
    except Exception as e:
        logger.error(f"Error loading config from {config_file}: {e}")
        return config_class()

def save_config_to_file(config: Any, config_file: Path):
    """Save configuration to JSON file."""
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings for JSON serialization
    config_dict = {}
    for field_name, field_info in config.__class__.__dataclass_fields__.items():
        value = getattr(config, field_name)
        if isinstance(value, Path):
            config_dict[field_name] = str(value)
        else:
            config_dict[field_name] = value

    try:
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {config_file}")
    except Exception as e:
        logger.error(f"Error saving config to {config_file}: {e}")

def get_config(config_class: type = AppConfig, config_file: Optional[Path] = None) -> Any:
    """Get configuration with priority: env vars > config file > defaults."""
    # Start with defaults
    config = config_class()

    # Override with config file if provided
    if config_file:
        file_config = load_config_from_file(config_file, config_class)
        for field_name in config_class.__dataclass_fields__:
            file_value = getattr(file_config, field_name)
            if file_value != getattr(config_class(), field_name):
                setattr(config, field_name, file_value)

    # Override with environment variables (highest priority)
    env_config = load_config_from_env(config_class)
    for field_name in config_class.__dataclass_fields__:
        env_value = getattr(env_config, field_name)
        default_value = getattr(config_class(), field_name)
        if env_value != default_value:
            setattr(config, field_name, env_value)

    return config