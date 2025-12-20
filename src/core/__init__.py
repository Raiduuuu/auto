"""Core module - Base classes and orchestration"""
from .base_agent import BaseAgent, AgentMessage, AgentRole, TradingSignal
from .orchestrator import AgentOrchestrator, OrchestratorMode
from .competition import TradingCompetition, CompetitionConfig, CompetitorStats

__all__ = [
    "BaseAgent",
    "AgentMessage",
    "AgentRole",
    "TradingSignal",
    "AgentOrchestrator",
    "OrchestratorMode",
    "TradingCompetition",
    "CompetitionConfig",
    "CompetitorStats",
]
