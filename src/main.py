"""
TradingAgents - Main Entry Point
Multi-Agent LLM Financial Trading Framework

Supports three operating modes:
- CLASSIC: Multi-agent collaboration (TechnicalAnalyst, SentimentAnalyst, RiskManager, Trader)
- AUTONOMOUS: Pure tool-driven autonomous AI agent (inspired by AI-Trader)
- HYBRID: Both modes running in parallel with weighted consensus
"""
import asyncio
import sys
import argparse
from typing import Optional
from loguru import logger

from config.settings import settings
from src.core.orchestrator import AgentOrchestrator, OrchestratorMode
from src.agents import (
    TechnicalAnalystAgent,
    SentimentAnalystAgent,
    RiskManagerAgent,
    TraderAgent
)
from src.agents.autonomous_agent import AutonomousAgent
from src.data.market_data import MarketDataProvider
from src.data.indicators import TechnicalIndicators
from src.llm.client import LLMClient
from src.ctrader.client import CTraderClient
from src.tools.toolchain import TradingToolchain
from src.core.competition import TradingCompetition, CompetitionConfig
from src.utils.logger import setup_logger


class TradingBot:
    """
    Main trading bot that orchestrates all agents and handles trading operations.

    Supports three modes:
    - classic: Original multi-agent system
    - autonomous: AI-Trader inspired pure tool-driven agent
    - hybrid: Both systems with consensus
    """

    def __init__(self, mode: str = "classic"):
        # Convert string mode to enum
        mode_map = {
            "classic": OrchestratorMode.CLASSIC,
            "autonomous": OrchestratorMode.AUTONOMOUS,
            "hybrid": OrchestratorMode.HYBRID
        }
        self.mode = mode_map.get(mode, OrchestratorMode.CLASSIC)

        self.orchestrator = AgentOrchestrator(mode=self.mode)
        self.market_data: Optional[MarketDataProvider] = None
        self.ctrader: Optional[CTraderClient] = None
        self.llm: Optional[LLMClient] = None
        self.toolchain: Optional[TradingToolchain] = None
        self.autonomous_agent: Optional[AutonomousAgent] = None
        self.is_running = False

    async def initialize(self) -> bool:
        """Initialize all components."""
        logger.info(f"Initializing TradingAgents (mode: {self.mode.value})...")

        # Setup logging
        setup_logger(level=settings.logging.level)

        # Initialize LLM client
        if settings.llm.anthropic_api_key:
            self.llm = LLMClient(
                provider="anthropic",
                api_key=settings.llm.anthropic_api_key,
                model=settings.llm.default_model,
                max_tokens=settings.llm.max_tokens,
                temperature=settings.llm.temperature
            )
            logger.info("LLM client initialized")
        else:
            logger.warning("No LLM API key configured - running without LLM")

        # Initialize cTrader client
        if settings.ctrader.client_id and settings.ctrader.access_token:
            self.ctrader = CTraderClient(
                client_id=settings.ctrader.client_id,
                client_secret=settings.ctrader.client_secret,
                access_token=settings.ctrader.access_token,
                account_id=settings.ctrader.account_id,
                host=settings.ctrader.host,
                port=settings.ctrader.port
            )

            connected = await self.ctrader.connect()
            if connected:
                logger.info("cTrader client connected")
            else:
                logger.warning("cTrader connection failed - using sample data")
                self.ctrader = None
        else:
            logger.warning("No cTrader credentials - using sample data")

        # Initialize market data provider
        self.market_data = MarketDataProvider(ctrader_client=self.ctrader)

        # Initialize toolchain for autonomous mode
        self.toolchain = TradingToolchain(
            ctrader_client=self.ctrader,
            market_data_provider=self.market_data,
            llm_client=self.llm
        )
        self.orchestrator.set_toolchain(self.toolchain)

        # Register agents based on mode
        if self.mode in [OrchestratorMode.CLASSIC, OrchestratorMode.HYBRID]:
            self._register_classic_agents()

        if self.mode in [OrchestratorMode.AUTONOMOUS, OrchestratorMode.HYBRID]:
            self._register_autonomous_agent()

        logger.info("TradingAgents initialized successfully")
        return True

    def _register_classic_agents(self) -> None:
        """Register classic multi-agent team."""
        # Technical Analyst
        technical = TechnicalAnalystAgent(llm_client=self.llm)
        self.orchestrator.register_agent(technical)

        # Sentiment Analyst
        sentiment = SentimentAnalystAgent(llm_client=self.llm)
        self.orchestrator.register_agent(sentiment)

        # Risk Manager
        risk_manager = RiskManagerAgent(
            llm_client=self.llm,
            max_risk_per_trade=settings.trading.default_risk_percent,
            max_daily_loss=settings.trading.max_daily_loss_percent,
            max_drawdown=settings.trading.max_drawdown_percent
        )
        self.orchestrator.register_agent(risk_manager)

        # Trader
        trader = TraderAgent(llm_client=self.llm)
        self.orchestrator.register_agent(trader)

        logger.info(f"Registered {len(self.orchestrator.agents)} classic agents")

    def _register_autonomous_agent(self) -> None:
        """Register autonomous AI agent."""
        self.autonomous_agent = AutonomousAgent(
            name="AutonomousTrader",
            llm_client=self.llm,
            toolchain=self.toolchain,
            model_name="claude"
        )
        self.orchestrator.set_autonomous_agent(self.autonomous_agent)
        logger.info("Registered autonomous agent")

    async def run_trading_cycle(self, instrument: str = "DE40") -> dict:
        """
        Run a complete trading cycle for an instrument.
        """
        logger.info(f"Running trading cycle for {instrument} (mode: {self.mode.value})")

        # Get market data
        ohlcv = await self.market_data.get_ohlcv(
            instrument=instrument,
            timeframe="H1",
            periods=100
        )

        # Calculate indicators
        indicators = TechnicalIndicators.calculate_all(ohlcv)

        # Get current price
        price_data = await self.market_data.get_current_price(instrument)

        # Get account info
        account = await self.market_data.get_account_info()

        # Prepare market data for analysis
        market_data = {
            "instrument": instrument,
            "timeframe": "M5",
            "ohlcv": {
                "open": ohlcv['open'].tolist(),
                "high": ohlcv['high'].tolist(),
                "low": ohlcv['low'].tolist(),
                "close": ohlcv['close'].tolist(),
                "volume": ohlcv['volume'].tolist()
            },
            "indicators": indicators,
            "current_price": price_data.get("bid", indicators.get("current_price", 0)),
            "atr": indicators.get("atr", 0),
            "account": account,
            "account_balance": account.get("balance", 10000),
            "news": [],
            "social": {},
            "fear_greed_index": 50
        }

        # Run analysis cycle
        results = await self.orchestrator.run_analysis_cycle(market_data)

        # Extract signal based on mode
        if self.mode == OrchestratorMode.HYBRID:
            signal = results.get("hybrid_consensus", {})
        elif self.mode == OrchestratorMode.AUTONOMOUS:
            signal = results.get("signal", {})
        else:
            trading_decision = results.get("trading_decision", {})
            signal = trading_decision.get("signal", trading_decision.get("decision", {}))

        # Execute trade if decision is to trade
        if signal and self.ctrader:
            action = signal.get("direction", signal.get("action", "HOLD"))
            if action in ["BUY", "SELL"]:
                order_params = {
                    "symbol": instrument,
                    "direction": action,
                    "volume": signal.get("position_size", 0.1),
                    "stop_loss": signal.get("stop_loss"),
                    "take_profit": signal.get("take_profit")
                }
                execution_result = await self.ctrader.place_order(**order_params)
                results["execution"] = execution_result
                logger.info(f"Order execution: {execution_result}")

        return results

    async def run_competition(
        self,
        rounds: int = 10,
        instrument: str = "DE40"
    ) -> dict:
        """
        Run a competition between classic and autonomous agents.
        """
        logger.info(f"Starting competition: {rounds} rounds on {instrument}")

        config = CompetitionConfig(
            name=f"Classic-vs-Autonomous-{instrument}",
            instrument=instrument,
            max_rounds=rounds,
            initial_balance=10000.0
        )

        competition = TradingCompetition(config)

        # Add classic team as one competitor (using trader agent)
        classic_trader = TraderAgent(llm_client=self.llm)
        classic_trader.name = "ClassicTeam"
        competition.add_competitor(classic_trader, "multi-agent")

        # Add autonomous agent as competitor
        autonomous = AutonomousAgent(
            name="AutonomousAI",
            llm_client=self.llm,
            toolchain=self.toolchain,
            model_name="claude"
        )
        competition.add_competitor(autonomous, "autonomous")

        await competition.start()

        for round_num in range(rounds):
            # Get market data
            ohlcv = await self.market_data.get_ohlcv(
                instrument=instrument,
                timeframe="M5",
                periods=100
            )
            indicators = TechnicalIndicators.calculate_all(ohlcv)

            market_data = {
                "instrument": instrument,
                "ohlcv": ohlcv,
                "indicators": indicators,
                "current_price": indicators.get("current_price", 20000)
            }

            round_result = await competition.run_round(market_data)
            logger.info(f"Round {round_num + 1}/{rounds} completed")

        return competition.export_results()

    async def run(
        self,
        instruments: list = None,
        interval_seconds: int = 300
    ) -> None:
        """
        Run the trading bot continuously.
        """
        if instruments is None:
            instruments = settings.trading.instruments

        self.is_running = True
        logger.info(f"Starting trading bot for {instruments} (mode: {self.mode.value})")

        try:
            while self.is_running:
                for instrument in instruments:
                    try:
                        results = await self.run_trading_cycle(instrument)

                        # Extract signal based on mode
                        if self.mode == OrchestratorMode.HYBRID:
                            signal = results.get("hybrid_consensus", {})
                        elif self.mode == OrchestratorMode.AUTONOMOUS:
                            signal = results.get("signal", {})
                        else:
                            decision = results.get("trading_decision", {}).get("decision", {})
                            signal = decision

                        action = signal.get("direction", signal.get("action", "HOLD"))
                        confidence = signal.get("confidence", 0)

                        logger.info(
                            f"[{self.mode.value.upper()}] {instrument}: {action} "
                            f"(confidence: {confidence:.0%})"
                        )
                    except Exception as e:
                        logger.error(f"Error in trading cycle for {instrument}: {e}")

                # Wait for next cycle
                logger.info(f"Waiting {interval_seconds}s for next cycle...")
                await asyncio.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Shutdown the trading bot."""
        self.is_running = False

        if self.ctrader:
            await self.ctrader.disconnect()

        logger.info("TradingAgents shutdown complete")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TradingAgents - Multi-Agent LLM Trading Framework"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["classic", "autonomous", "hybrid"],
        default="classic",
        help="Operating mode (default: classic)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run single analysis cycle and exit"
    )
    parser.add_argument(
        "--compete",
        action="store_true",
        help="Run competition between classic and autonomous modes"
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=10,
        help="Number of competition rounds (default: 10)"
    )
    parser.add_argument(
        "--instrument", "-i",
        default="DE40",
        help="Trading instrument (default: DE40)"
    )
    parser.add_argument(
        "--interval", "-t",
        type=int,
        default=300,
        help="Interval between cycles in seconds (default: 300)"
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    bot = TradingBot(mode=args.mode)

    if not await bot.initialize():
        logger.error("Failed to initialize trading bot")
        sys.exit(1)

    if args.compete:
        # Run competition mode
        results = await bot.run_competition(
            rounds=args.rounds,
            instrument=args.instrument
        )
        print("\n" + "=" * 60)
        print("       COMPETITION RESULTS")
        print("=" * 60)
        summary = results.get("summary", {})
        print(f"Rounds completed: {summary.get('rounds_completed', 0)}")
        print(f"Status: {summary.get('status', 'unknown')}")
        print("\nLeaderboard:")
        for entry in summary.get("leaderboard", []):
            print(f"  #{entry['rank']} {entry['agent_name']}: "
                  f"{entry['profit_percent']:+.2f}% "
                  f"(Win rate: {entry['win_rate']:.1f}%)")
        await bot.shutdown()

    elif args.once:
        # Run single analysis cycle
        results = await bot.run_trading_cycle(args.instrument)
        print("\n" + "=" * 60)
        print(f"       TRADING ANALYSIS ({args.mode.upper()} MODE)")
        print("=" * 60)

        if args.mode == "hybrid":
            signal = results.get("hybrid_consensus", {})
            print(f"Consensus Direction: {signal.get('direction', 'HOLD')}")
            print(f"Confidence: {signal.get('confidence', 0):.0%}")
            print(f"Buy Score: {signal.get('buy_score', 0):.2f}")
            print(f"Sell Score: {signal.get('sell_score', 0):.2f}")
        elif args.mode == "autonomous":
            signal = results.get("signal", {})
            print(f"Direction: {signal.get('direction', 'HOLD')}")
            print(f"Confidence: {signal.get('confidence', 0):.0%}")
            print(f"Entry: {signal.get('entry_price', 'N/A')}")
            print(f"Stop Loss: {signal.get('stop_loss', 'N/A')}")
            print(f"Take Profit: {signal.get('take_profit', 'N/A')}")
            print(f"Reasoning: {signal.get('reasoning', 'N/A')[:100]}...")
        else:
            decision = results.get("trading_decision", {}).get("decision", {})
            print(f"Action: {decision.get('action', 'HOLD')}")
            print(f"Confidence: {decision.get('confidence', 0):.0%}")
            print(f"Reason: {decision.get('reason', 'N/A')}")

        await bot.shutdown()

    else:
        # Run continuously
        await bot.run(
            instruments=[args.instrument],
            interval_seconds=args.interval
        )


if __name__ == "__main__":
    asyncio.run(main())
