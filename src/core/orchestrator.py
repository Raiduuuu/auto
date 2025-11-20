"""
Agent Orchestrator - Manages communication and coordination between agents
"""
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger

from .base_agent import BaseAgent, AgentMessage, TradingSignal


class AgentOrchestrator:
    """
    Orchestrates multiple trading agents, managing their communication
    and coordinating trading decisions.
    """

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.signals: List[TradingSignal] = []
        self.is_running = False
        self.decision_history: List[Dict[str, Any]] = []

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name} ({agent.role.value})")

    def unregister_agent(self, agent_name: str) -> None:
        """Remove an agent from the orchestrator."""
        if agent_name in self.agents:
            del self.agents[agent_name]
            logger.info(f"Unregistered agent: {agent_name}")

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self.agents.get(name)

    def get_agents_by_role(self, role: str) -> List[BaseAgent]:
        """Get all agents with a specific role."""
        return [
            agent for agent in self.agents.values()
            if agent.role.value == role
        ]

    async def broadcast_message(
        self,
        sender: str,
        content: Dict[str, Any],
        message_type: str = "broadcast",
        exclude: List[str] = None
    ) -> None:
        """Broadcast a message to all agents except those in exclude list."""
        exclude = exclude or []
        for agent_name, agent in self.agents.items():
            if agent_name not in exclude and agent_name != sender:
                message = AgentMessage(
                    sender=sender,
                    receiver=agent_name,
                    content=content,
                    message_type=message_type
                )
                await self.message_queue.put(message)

    async def send_message(self, message: AgentMessage) -> None:
        """Send a message to a specific agent."""
        await self.message_queue.put(message)

    async def process_messages(self) -> None:
        """Process all messages in the queue."""
        while not self.message_queue.empty():
            message = await self.message_queue.get()
            receiver = self.agents.get(message.receiver)

            if receiver:
                try:
                    response = await receiver.process_message(message)
                    if response:
                        await self.message_queue.put(response)
                except Exception as e:
                    logger.error(f"Error processing message for {message.receiver}: {e}")

    async def run_analysis_cycle(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a complete analysis cycle with all agents.
        Returns aggregated analysis and trading signals.
        """
        logger.info("Starting analysis cycle")
        results = {}

        # Phase 1: Market Analysis
        analyst_agents = [
            agent for agent in self.agents.values()
            if "analyst" in agent.role.value
        ]

        analysis_tasks = [
            agent.analyze(market_data) for agent in analyst_agents
        ]
        analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)

        for agent, analysis in zip(analyst_agents, analyses):
            if isinstance(analysis, Exception):
                logger.error(f"Analysis error from {agent.name}: {analysis}")
            else:
                results[agent.name] = analysis

        # Phase 2: Risk Assessment
        risk_manager = self.get_agents_by_role("risk_manager")
        if risk_manager:
            risk_data = {
                "analyses": results,
                "market_data": market_data
            }
            risk_assessment = await risk_manager[0].analyze(risk_data)
            results["risk_assessment"] = risk_assessment

        # Phase 3: Trading Decision
        trader = self.get_agents_by_role("trader")
        if trader:
            decision_data = {
                "analyses": results,
                "market_data": market_data,
                "risk_assessment": results.get("risk_assessment", {})
            }
            trading_decision = await trader[0].analyze(decision_data)
            results["trading_decision"] = trading_decision

        # Process any inter-agent messages
        await self.process_messages()

        # Store decision
        self.decision_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "results": results
        })

        logger.info("Analysis cycle completed")
        return results

    async def generate_consensus_signal(
        self,
        instrument: str,
        analyses: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """
        Generate a consensus trading signal from multiple agent analyses.
        """
        signals = []
        confidences = []

        for agent_name, analysis in analyses.items():
            if "signal" in analysis:
                signal = analysis["signal"]
                signals.append(signal.get("direction", "HOLD"))
                confidences.append(signal.get("confidence", 0.5))

        if not signals:
            return None

        # Simple majority voting
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")
        total = len(signals)

        if buy_count > total / 2:
            direction = "BUY"
        elif sell_count > total / 2:
            direction = "SELL"
        else:
            direction = "HOLD"

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return TradingSignal(
            instrument=instrument,
            direction=direction,
            confidence=avg_confidence,
            source_agent="orchestrator"
        )

    def get_active_agents(self) -> List[str]:
        """Get names of all active agents."""
        return [
            name for name, agent in self.agents.items()
            if agent.is_active
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "is_running": self.is_running,
            "agent_count": len(self.agents),
            "active_agents": self.get_active_agents(),
            "pending_messages": self.message_queue.qsize(),
            "signals_generated": len(self.signals),
            "decisions_made": len(self.decision_history)
        }
