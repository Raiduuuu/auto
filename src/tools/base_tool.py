"""
Base Tool - Foundation for MCP-style trading tools
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum


class ToolStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


@dataclass
class ToolResult:
    """Standardized result from any tool execution."""
    tool_name: str
    status: ToolStatus
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "status": self.status.value,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp,
            "execution_time_ms": self.execution_time_ms,
        }

    @property
    def is_success(self) -> bool:
        return self.status == ToolStatus.SUCCESS


class BaseTool(ABC):
    """
    Base class for all MCP-style trading tools.
    Provides a standardized interface for autonomous agent operations.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.call_count = 0
        self.last_call: Optional[datetime] = None

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Return the tool's parameter schema for LLM function calling."""
        pass

    def get_tool_spec(self) -> Dict[str, Any]:
        """Get tool specification for LLM tool use."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.get_schema(),
        }

    async def __call__(self, **kwargs) -> ToolResult:
        """Allow calling tool directly."""
        import time
        start = time.time()

        self.call_count += 1
        self.last_call = datetime.utcnow()

        result = await self.execute(**kwargs)
        result.execution_time_ms = (time.time() - start) * 1000

        return result
