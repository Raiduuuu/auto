"""
Competition Framework - Test multiple AI agents/strategies against each other
Inspired by AI-Trader's multi-model competition arena
"""
from typing import Any, Dict, List, Optional, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from loguru import logger

from .base_agent import BaseAgent


class CompetitionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class CompetitorStats:
    """Statistics for a single competitor."""
    agent_name: str
    model_name: str
    initial_balance: float
    current_balance: float
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    signals_generated: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    hold_signals: int = 0
    equity_curve: List[float] = field(default_factory=list)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        if self.trades_count == 0:
            return 0.0
        return (self.winning_trades / self.trades_count) * 100

    @property
    def profit_percent(self) -> float:
        if self.initial_balance == 0:
            return 0.0
        return ((self.current_balance - self.initial_balance) / self.initial_balance) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "model_name": self.model_name,
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "profit_loss": self.total_profit_loss,
            "profit_percent": round(self.profit_percent, 2),
            "trades": self.trades_count,
            "win_rate": round(self.win_rate, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "signals": {
                "total": self.signals_generated,
                "buy": self.buy_signals,
                "sell": self.sell_signals,
                "hold": self.hold_signals
            }
        }


@dataclass
class CompetitionConfig:
    """Configuration for a trading competition."""
    name: str
    instrument: str = "DE40"
    timeframe: str = "M5"
    initial_balance: float = 10000.0
    duration_hours: Optional[int] = None  # None = unlimited
    max_rounds: Optional[int] = None  # None = unlimited
    round_interval_seconds: int = 300  # 5 minutes
    risk_per_trade: float = 1.0  # 1%
    allow_simultaneous_positions: bool = False
    track_paper_trades: bool = True


class TradingCompetition:
    """
    Manages a competition between multiple trading agents.

    Each agent starts with the same initial balance and conditions,
    competing to achieve the highest returns.
    """

    def __init__(self, config: CompetitionConfig):
        self.config = config
        self.status = CompetitionStatus.PENDING
        self.competitors: Dict[str, BaseAgent] = {}
        self.stats: Dict[str, CompetitorStats] = {}
        self.rounds_completed = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.round_results: List[Dict[str, Any]] = []

    def add_competitor(self, agent: BaseAgent, model_name: str = "unknown") -> None:
        """Add an agent to the competition."""
        if self.status != CompetitionStatus.PENDING:
            raise ValueError("Cannot add competitors after competition has started")

        self.competitors[agent.name] = agent
        self.stats[agent.name] = CompetitorStats(
            agent_name=agent.name,
            model_name=model_name,
            initial_balance=self.config.initial_balance,
            current_balance=self.config.initial_balance,
            equity_curve=[self.config.initial_balance]
        )

        logger.info(f"Added competitor: {agent.name} ({model_name})")

    def remove_competitor(self, agent_name: str) -> None:
        """Remove an agent from the competition."""
        if agent_name in self.competitors:
            del self.competitors[agent_name]
            del self.stats[agent_name]
            logger.info(f"Removed competitor: {agent_name}")

    async def start(self) -> None:
        """Start the competition."""
        if not self.competitors:
            raise ValueError("No competitors registered")

        self.status = CompetitionStatus.RUNNING
        self.start_time = datetime.utcnow()

        logger.info(f"Competition '{self.config.name}' started with {len(self.competitors)} competitors")

    async def run_round(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single competition round.

        All agents analyze the same market data and generate signals.
        """
        if self.status != CompetitionStatus.RUNNING:
            return {"error": "Competition not running"}

        round_number = self.rounds_completed + 1
        logger.info(f"Running competition round {round_number}")

        round_result = {
            "round": round_number,
            "timestamp": datetime.utcnow().isoformat(),
            "market_data_summary": self._summarize_market_data(market_data),
            "agent_results": {}
        }

        # Run all agents in parallel
        tasks = []
        for name, agent in self.competitors.items():
            tasks.append(self._run_agent_round(name, agent, market_data))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for name, result in zip(self.competitors.keys(), results):
            if isinstance(result, Exception):
                round_result["agent_results"][name] = {
                    "error": str(result),
                    "signal": None
                }
            else:
                round_result["agent_results"][name] = result
                self._update_stats(name, result)

        self.round_results.append(round_result)
        self.rounds_completed += 1

        # Check if competition should end
        if self._should_end():
            await self.end()

        return round_result

    async def _run_agent_round(
        self,
        agent_name: str,
        agent: BaseAgent,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single agent's analysis for this round."""
        try:
            # Prepare data with competition context
            data = {
                **market_data,
                "instrument": self.config.instrument,
                "timeframe": self.config.timeframe,
                "account_balance": self.stats[agent_name].current_balance,
                "risk_percent": self.config.risk_per_trade,
                "competition_round": self.rounds_completed + 1
            }

            # Run analysis
            analysis = await agent.analyze(data)

            # Extract signal
            signal = analysis.get("signal", {})

            return {
                "signal": signal,
                "analysis_summary": self._summarize_analysis(analysis),
                "execution_time_ms": analysis.get("execution_time_ms", 0)
            }

        except Exception as e:
            logger.error(f"Agent {agent_name} error: {e}")
            raise

    def _update_stats(self, agent_name: str, result: Dict[str, Any]) -> None:
        """Update agent statistics based on round result."""
        stats = self.stats[agent_name]
        signal = result.get("signal", {})

        if signal:
            stats.signals_generated += 1
            direction = signal.get("direction", "HOLD").upper()

            if direction == "BUY":
                stats.buy_signals += 1
            elif direction == "SELL":
                stats.sell_signals += 1
            else:
                stats.hold_signals += 1

            # Simulate paper trade if enabled
            if self.config.track_paper_trades and direction in ["BUY", "SELL"]:
                self._simulate_paper_trade(stats, signal)

        # Update equity curve
        stats.equity_curve.append(stats.current_balance)

        # Update max drawdown
        self._update_drawdown(stats)

    def _simulate_paper_trade(
        self,
        stats: CompetitorStats,
        signal: Dict[str, Any]
    ) -> None:
        """Simulate a paper trade for tracking."""
        # This is a simplified paper trading simulation
        # In reality, you'd track entry/exit and calculate P&L

        trade = {
            "timestamp": datetime.utcnow().isoformat(),
            "direction": signal.get("direction"),
            "confidence": signal.get("confidence", 0),
            "entry_price": signal.get("entry_price"),
            "stop_loss": signal.get("stop_loss"),
            "take_profit": signal.get("take_profit"),
            "status": "paper"
        }

        stats.trade_history.append(trade)
        stats.trades_count += 1

    def _update_drawdown(self, stats: CompetitorStats) -> None:
        """Update maximum drawdown for agent."""
        if len(stats.equity_curve) < 2:
            return

        peak = max(stats.equity_curve)
        current = stats.equity_curve[-1]

        if peak > 0:
            drawdown = (peak - current) / peak * 100
            stats.max_drawdown = max(stats.max_drawdown, drawdown)

    def _summarize_market_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of market data for logging."""
        return {
            "instrument": data.get("instrument", self.config.instrument),
            "current_price": data.get("current_price"),
            "timestamp": datetime.utcnow().isoformat()
        }

    def _summarize_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of agent analysis."""
        signal = analysis.get("signal", {})
        return {
            "direction": signal.get("direction", "HOLD"),
            "confidence": signal.get("confidence", 0),
            "has_entry": signal.get("entry_price") is not None,
            "has_sl_tp": signal.get("stop_loss") is not None
        }

    def _should_end(self) -> bool:
        """Check if competition should end."""
        # Check duration
        if self.config.duration_hours and self.start_time:
            elapsed = datetime.utcnow() - self.start_time
            if elapsed.total_seconds() / 3600 >= self.config.duration_hours:
                return True

        # Check rounds
        if self.config.max_rounds and self.rounds_completed >= self.config.max_rounds:
            return True

        return False

    async def end(self) -> None:
        """End the competition."""
        self.status = CompetitionStatus.COMPLETED
        self.end_time = datetime.utcnow()

        logger.info(f"Competition '{self.config.name}' completed after {self.rounds_completed} rounds")

    def pause(self) -> None:
        """Pause the competition."""
        if self.status == CompetitionStatus.RUNNING:
            self.status = CompetitionStatus.PAUSED
            logger.info(f"Competition '{self.config.name}' paused")

    def resume(self) -> None:
        """Resume a paused competition."""
        if self.status == CompetitionStatus.PAUSED:
            self.status = CompetitionStatus.RUNNING
            logger.info(f"Competition '{self.config.name}' resumed")

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get current competition leaderboard."""
        leaderboard = []

        for name, stats in self.stats.items():
            entry = stats.to_dict()
            entry["rank"] = 0  # Will be set after sorting
            leaderboard.append(entry)

        # Sort by profit percentage (descending)
        leaderboard.sort(key=lambda x: x["profit_percent"], reverse=True)

        # Assign ranks
        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1

        return leaderboard

    def get_competition_summary(self) -> Dict[str, Any]:
        """Get overall competition summary."""
        duration = None
        if self.start_time:
            end = self.end_time or datetime.utcnow()
            duration = (end - self.start_time).total_seconds()

        return {
            "name": self.config.name,
            "status": self.status.value,
            "instrument": self.config.instrument,
            "timeframe": self.config.timeframe,
            "competitors_count": len(self.competitors),
            "rounds_completed": self.rounds_completed,
            "duration_seconds": duration,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "leaderboard": self.get_leaderboard()
        }

    def export_results(self) -> Dict[str, Any]:
        """Export full competition results."""
        return {
            "summary": self.get_competition_summary(),
            "config": {
                "name": self.config.name,
                "instrument": self.config.instrument,
                "timeframe": self.config.timeframe,
                "initial_balance": self.config.initial_balance,
                "duration_hours": self.config.duration_hours,
                "max_rounds": self.config.max_rounds
            },
            "competitor_details": {
                name: stats.to_dict()
                for name, stats in self.stats.items()
            },
            "round_results": self.round_results
        }


async def run_quick_competition(
    agents: List[BaseAgent],
    market_data_provider: Any,
    instrument: str = "DE40",
    rounds: int = 10
) -> Dict[str, Any]:
    """
    Run a quick competition for testing.

    Convenience function for running short competitions.
    """
    config = CompetitionConfig(
        name=f"Quick-{datetime.utcnow().strftime('%Y%m%d-%H%M')}",
        instrument=instrument,
        max_rounds=rounds,
        round_interval_seconds=1
    )

    competition = TradingCompetition(config)

    for agent in agents:
        model_name = getattr(agent, "model_name", "unknown")
        competition.add_competitor(agent, model_name)

    await competition.start()

    for _ in range(rounds):
        # Get fresh market data for each round
        if market_data_provider:
            try:
                market_data = await market_data_provider.get_ohlcv(
                    symbol=instrument,
                    timeframe="M5",
                    limit=100
                )
            except Exception:
                market_data = {}
        else:
            market_data = {}

        await competition.run_round({"ohlcv": market_data})

        await asyncio.sleep(config.round_interval_seconds)

    return competition.export_results()
