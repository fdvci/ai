# config.py
"""Configuration management for Nova AI Agent"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class NovaConfig:
    """Configuration class for Nova AI Agent with validation"""
    
    # Core settings
    brain_file: str = "data/memory.json"
    state_file: str = "data/agent_state.json"
    vector_db_path: str = "data/vector_db"
    
    # API keys
    openai_api_key: str = ""
    brave_api_key: str = ""
    
    # Search and retrieval settings
    max_search_results: int = 5
    confidence_threshold: float = 0.8
    vector_similarity_threshold: float = 0.7
    max_context_size: int = 10
    
    # Memory management
    max_memories: int = 10000
    memory_cleanup_days: int = 30
    consolidation_threshold: int = 500
    
    # Autonomy settings
    autonomy_enabled: bool = True
    learning_rate: float = 0.1
    curiosity_threshold: float = 0.6
    
    # Performance settings
    max_response_tokens: int = 800
    request_timeout: int = 30
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    
    # Safety settings
    enable_self_modification: bool = True
    max_modifications_per_session: int = 3
    safety_checks_enabled: bool = True
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True
    health_check_interval: int = 300  # 5 minutes
    
    # Error handling
    max_retry_attempts: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Load from environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        self.brave_api_key = os.getenv("BRAVE_API_KEY", self.brave_api_key)
        
        # Create necessary directories
        self._create_directories()
        
        # Validate configuration
        errors = self.validate()
        if errors:
            logger.warning(f"Configuration validation warnings: {errors}")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            os.path.dirname(self.brain_file),
            os.path.dirname(self.state_file),
            self.vector_db_path,
            "data/exports",
            "data/health_reports",
            "data/shutdown_reports",
            "data/code_backups",
            "data/web_cache"
        ]
        
        for directory in directories:
            if directory:  # Skip empty directory names
                try:
                    os.makedirs(directory, exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to create directory {directory}: {e}")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Check required API keys
        if not self.openai_api_key:
            errors.append("OpenAI API key is required")
        
        # Validate numeric ranges
        if not 0 < self.confidence_threshold <= 1:
            errors.append("confidence_threshold must be between 0 and 1")
        
        if not 0 < self.vector_similarity_threshold <= 1:
            errors.append("vector_similarity_threshold must be between 0 and 1")
        
        if self.max_context_size < 1:
            errors.append("max_context_size must be positive")
        
        if self.max_memories < 100:
            errors.append("max_memories should be at least 100")
        
        if self.memory_cleanup_days < 1:
            errors.append("memory_cleanup_days must be positive")
        
        if not 0 <= self.learning_rate <= 1:
            errors.append("learning_rate must be between 0 and 1")
        
        if self.request_timeout < 1:
            errors.append("request_timeout must be positive")
        
        if self.max_retry_attempts < 1:
            errors.append("max_retry_attempts must be positive")
        
        # Validate file paths
        try:
            Path(self.brain_file).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create brain file directory: {e}")
        
        try:
            Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create state file directory: {e}")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            errors.append(f"log_level must be one of {valid_log_levels}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {}
        for field_name, field_def in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            # Don't include sensitive data like API keys
            if 'api_key' not in field_name.lower():
                result[field_name] = value
            else:
                result[field_name] = "***" if value else ""
        return result
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file (excluding sensitive data)"""
        try:
            config_data = self.to_dict()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'NovaConfig':
        """Load configuration from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Remove sensitive data placeholders
            config_data = {k: v for k, v in config_data.items() if v != "***"}
            
            return cls(**config_data)
        except FileNotFoundError:
            logger.info(f"Configuration file {filepath} not found, using defaults")
            return cls()
        except Exception as e:
            logger.error(f"Failed to load configuration from {filepath}: {e}")
            logger.info("Using default configuration")
            return cls()
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """Update configuration from dictionary with validation"""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        # Re-validate after updates
        errors = self.validate()
        if errors:
            logger.warning(f"Configuration validation errors after update: {errors}")
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI-specific configuration"""
        return {
            "api_key": self.openai_api_key,
            "timeout": self.request_timeout,
            "max_retries": self.max_retry_attempts,
            "max_tokens": self.max_response_tokens
        }
    
    def get_search_config(self) -> Dict[str, Any]:
        """Get search-specific configuration"""
        return {
            "brave_api_key": self.brave_api_key,
            "max_results": self.max_search_results,
            "timeout": self.request_timeout
        }
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory-specific configuration"""
        return {
            "max_memories": self.max_memories,
            "cleanup_days": self.memory_cleanup_days,
            "consolidation_threshold": self.consolidation_threshold,
            "similarity_threshold": self.vector_similarity_threshold
        }
    
    def get_autonomy_config(self) -> Dict[str, Any]:
        """Get autonomy-specific configuration"""
        return {
            "enabled": self.autonomy_enabled,
            "learning_rate": self.learning_rate,
            "curiosity_threshold": self.curiosity_threshold
        }


# Global configuration instance
_config: Optional[NovaConfig] = None


def get_config() -> NovaConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        # Try to load from file first
        config_file = "data/nova_config.json"
        _config = NovaConfig.load_from_file(config_file)
        
        # Save default config if it doesn't exist
        if not os.path.exists(config_file):
            try:
                _config.save_to_file(config_file)
            except Exception as e:
                logger.error(f"Failed to save default configuration: {e}")
    
    return _config


def set_config(config: NovaConfig):
    """Set the global configuration instance"""
    global _config
    _config = config


def reload_config():
    """Reload configuration from file"""
    global _config
    config_file = "data/nova_config.json"
    _config = NovaConfig.load_from_file(config_file)
    return _config


def update_config(**kwargs):
    """Update configuration with new values"""
    config = get_config()
    config.update_from_dict(kwargs)
    
    # Save updated configuration
    try:
        config.save_to_file("data/nova_config.json")
    except Exception as e:
        logger.error(f"Failed to save updated configuration: {e}")


# Environment-specific configuration presets
DEVELOPMENT_CONFIG = {
    "log_level": "DEBUG",
    "max_memories": 1000,
    "safety_checks_enabled": True,
    "enable_self_modification": False,
    "health_check_interval": 60
}

PRODUCTION_CONFIG = {
    "log_level": "INFO",
    "max_memories": 10000,
    "safety_checks_enabled": True,
    "enable_self_modification": True,
    "health_check_interval": 300
}

TESTING_CONFIG = {
    "log_level": "WARNING",
    "max_memories": 100,
    "safety_checks_enabled": False,
    "enable_self_modification": False,
    "health_check_interval": 10,
    "brain_file": "test_data/memory.json",
    "state_file": "test_data/agent_state.json",
    "vector_db_path": "test_data/vector_db"
}


def apply_environment_config(environment: str = "development"):
    """Apply environment-specific configuration"""
    config_presets = {
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG,
        "testing": TESTING_CONFIG
    }
    
    if environment in config_presets:
        update_config(**config_presets[environment])
        logger.info(f"Applied {environment} configuration preset")
    else:
        logger.warning(f"Unknown environment: {environment}")


# Configuration validation utilities
def validate_api_keys() -> Dict[str, bool]:
    """Validate that required API keys are present"""
    config = get_config()
    return {
        "openai": bool(config.openai_api_key),
        "brave": bool(config.brave_api_key)
    }


def get_missing_requirements() -> List[str]:
    """Get list of missing required configuration items"""
    config = get_config()
    missing = []
    
    if not config.openai_api_key:
        missing.append("OpenAI API key (OPENAI_API_KEY)")
    
    # Brave API key is optional for basic functionality
    
    return missing


def is_configuration_complete() -> bool:
    """Check if configuration is complete for basic operation"""
    return len(get_missing_requirements()) == 0


# Configuration context manager for testing
class ConfigContext:
    """Context manager for temporary configuration changes"""
    
    def __init__(self, **overrides):
        self.overrides = overrides
        self.original_config = None
    
    def __enter__(self):
        global _config
        self.original_config = _config
        
        # Create new config with overrides
        if _config:
            new_config_dict = _config.to_dict()
            new_config_dict.update(self.overrides)
            _config = NovaConfig(**new_config_dict)
        else:
            _config = NovaConfig(**self.overrides)
        
        return _config
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _config
        _config = self.original_config