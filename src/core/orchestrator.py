"""
Agent Orchestrator - Manages communication and coordination between agents
Supports both classic multi-agent mode and AI-Trader inspired autonomous mode
"""
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
from loguru import logger

from .base_agent import BaseAgent, AgentMessage, TradingSignal


class OrchestratorMode(Enum):
    """Operating mode for the orchestrator."""
    CLASSIC = "classic"          # Multi-agent collaboration (original)
    AUTONOMOUS = "autonomous"    # Pure tool-driven autonomous agent
    HYBRID = "hybrid"            # Both modes with consensus


class AgentOrchestrator:
    """
    Orchestrates multiple trading agents, managing their communication
    and coordinating trading decisions.
    """

    def __init__(self, mode: OrchestratorMode = OrchestratorMode.CLASSIC):
        self.agents: Dict[str, BaseAgent] = {}
        self.autonomous_agent: Optional[BaseAgent] = None
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.signals: List[TradingSignal] = []
        self.is_running = False
        self.decision_history: List[Dict[str, Any]] = []
        self.mode = mode
        self.toolchain = None  # Set externally for autonomous mode

    def set_mode(self, mode: OrchestratorMode) -> None:
        """Change the operating mode."""
        self.mode = mode
        logger.info(f"Orchestrator mode set to: {mode.value}")

    def set_autonomous_agent(self, agent: BaseAgent) -> None:
        """Set the autonomous agent for autonomous/hybrid mode."""
        self.autonomous_agent = agent
        logger.info(f"Autonomous agent set: {agent.name}")

    def set_toolchain(self, toolchain: Any) -> None:
        """Set the toolchain for autonomous mode."""
        self.toolchain = toolchain

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
        Run a complete analysis cycle based on current mode.
        Returns aggregated analysis and trading signals.
        """
        logger.info(f"Starting analysis cycle (mode: {self.mode.value})")

        if self.mode == OrchestratorMode.AUTONOMOUS:
            return await self._run_autonomous_cycle(market_data)
        elif self.mode == OrchestratorMode.HYBRID:
            return await self._run_hybrid_cycle(market_data)
        else:
            return await self._run_classic_cycle(market_data)

    async def _run_classic_cycle(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run classic multi-agent analysis cycle."""
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
            "mode": "classic",
            "results": results
        })

        logger.info("Classic analysis cycle completed")
        return results

    async def _run_autonomous_cycle(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run autonomous single-agent analysis cycle."""
        if not self.autonomous_agent:
            logger.error("No autonomous agent configured")
            return {"error": "No autonomous agent configured"}

        results = await self.autonomous_agent.analyze(market_data)

        # Store decision
        self.decision_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "autonomous",
            "results": results
        })

        logger.info("Autonomous analysis cycle completed")
        return results

    async def _run_hybrid_cycle(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run hybrid cycle with both classic and autonomous analysis.
        Uses weighted consensus for final decision.
        """
        # Run both cycles in parallel
        classic_task = self._run_classic_cycle(market_data)
        autonomous_task = self._run_autonomous_cycle(market_data)

        classic_result, autonomous_result = await asyncio.gather(
            classic_task, autonomous_task, return_exceptions=True
        )

        # Handle errors
        if isinstance(classic_result, Exception):
            classic_result = {"error": str(classic_result)}
        if isinstance(autonomous_result, Exception):
            autonomous_result = {"error": str(autonomous_result)}

        # Generate consensus
        consensus = self._generate_hybrid_consensus(classic_result, autonomous_result)

        results = {
            "classic_analysis": classic_result,
            "autonomous_analysis": autonomous_result,
            "hybrid_consensus": consensus,
            "mode": "hybrid"
        }

        # Store decision
        self.decision_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "hybrid",
            "results": results
        })

        logger.info("Hybrid analysis cycle completed")
        return results

    def _generate_hybrid_consensus(
        self,
        classic: Dict[str, Any],
        autonomous: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate consensus signal from both analysis modes."""
        signals = []
        weights = []

        # Extract classic signal
        classic_decision = classic.get("trading_decision", {})
        if classic_decision:
            classic_signal = classic_decision.get("signal", {})
            if classic_signal:
                signals.append(classic_signal)
                weights.append(0.4)  # 40% weight for classic

        # Extract autonomous signal
        autonomous_signal = autonomous.get("signal", {})
        if autonomous_signal:
            signals.append(autonomous_signal)
            weights.append(0.6)  # 60% weight for autonomous (more adaptive)

        if not signals:
            return {"direction": "HOLD", "confidence": 0.0, "source": "no_signals"}

        # Weighted voting
        buy_score = 0
        sell_score = 0
        total_confidence = 0

        for signal, weight in zip(signals, weights):
            direction = signal.get("direction", "HOLD").upper()
            confidence = signal.get("confidence", 0.5)

            if direction == "BUY":
                buy_score += weight * confidence
            elif direction == "SELL":
                sell_score += weight * confidence

            total_confidence += weight * confidence

        # Determine final direction
        if buy_score > sell_score and buy_score > 0.3:
            direction = "BUY"
            final_confidence = buy_score / sum(weights)
        elif sell_score > buy_score and sell_score > 0.3:
            direction = "SELL"
            final_confidence = sell_score / sum(weights)
        else:
            direction = "HOLD"
            final_confidence = 0.5

        return {
            "direction": direction,
            "confidence": round(final_confidence, 2),
            "buy_score": round(buy_score, 2),
            "sell_score": round(sell_score, 2),
            "source": "hybrid_consensus"
        }

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
            "mode": self.mode.value,
            "agent_count": len(self.agents),
            "active_agents": self.get_active_agents(),
            "autonomous_agent": self.autonomous_agent.name if self.autonomous_agent else None,
            "has_toolchain": self.toolchain is not None,
            "pending_messages": self.message_queue.qsize(),
            "signals_generated": len(self.signals),
            "decisions_made": len(self.decision_history)
        }
