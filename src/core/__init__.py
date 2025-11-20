"""Core module - Base classes and orchestration"""
from .base_agent import BaseAgent, AgentMessage, AgentRole
from .orchestrator import AgentOrchestrator

__all__ = ["BaseAgent", "AgentMessage", "AgentRole", "AgentOrchestrator"]
