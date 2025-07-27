# nova_types.py
"""Type definitions for Nova AI Agent"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Protocol
from datetime import datetime
from enum import Enum
import json


class MemoryType(Enum):
    CONVERSATION = "conversation"
    RESPONSE = "response"
    FACT = "fact"
    EMOTION = "emotion"
    WEB_RESULT = "web_result"
    SELF_EVALUATION = "self_evaluation"
    AUTONOMOUS_LEARNING = "autonomous_learning"
    PATTERN_ANALYSIS = "pattern_analysis"
    INSIGHT = "insight"
    GOAL = "goal"
    EVOLUTION_PLANNED = "evolution_planned"
    GOAL_RESEARCH = "goal_research"
    CONSOLIDATED_SUMMARY = "consolidated_summary"


@dataclass
class Memory:
    id: str
    timestamp: str
    speaker: str
    type: str  # Keep as string for backward compatibility
    content: str
    embedding_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        return cls(**data)
    
    def is_recent(self, hours: int = 24) -> bool:
        try:
            memory_time = datetime.fromisoformat(self.timestamp)
            return (datetime.now() - memory_time).total_seconds() / 3600 < hours
        except (ValueError, TypeError):
            return False


@dataclass
class Goal:
    id: str
    description: str
    type: str
    importance: float
    created_at: str
    status: str = "active"
    progress: float = 0.0
    deadline: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Goal':
        return cls(**data)


@dataclass
class Insight:
    id: str
    timestamp: str
    type: str
    content: str
    source: str
    confidence: float = 0.5
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Insight':
        return cls(**data)


@dataclass
class PerformanceMetrics:
    successful_queries: int = 0
    failed_queries: int = 0
    learning_events: int = 0
    self_modifications: int = 0
    total_conversations: int = 0
    avg_response_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def get(self, key: str, default=None):
        """Dict-like access for compatibility"""
        return getattr(self, key, default)
    
    def __getitem__(self, key: str):
        """Dict-like access"""
        return getattr(self, key)
    
    def __setitem__(self, key: str, value):
        """Dict-like setting"""
        setattr(self, key, value)


class NovaError(Exception):
    """Base exception for Nova AI Agent"""
    pass


class APIError(NovaError):
    """API-related errors"""
    pass


class MemoryError(NovaError):
    """Memory management errors"""
    pass


class ConfigurationError(NovaError):
    """Configuration-related errors"""
    pass


class ValidationError(NovaError):
    """Data validation errors"""
    pass