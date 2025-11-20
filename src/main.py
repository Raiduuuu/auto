"""
TradingAgents - Main Entry Point
Multi-Agent LLM Financial Trading Framework
"""
import asyncio
import sys
from typing import Optional
from loguru import logger

from config.settings import settings
from src.core.orchestrator import AgentOrchestrator
from src.agents import (
    TechnicalAnalystAgent,
    SentimentAnalystAgent,
    RiskManagerAgent,
    TraderAgent
)
from src.data.market_data import MarketDataProvider
from src.data.indicators import TechnicalIndicators
from src.llm.client import LLMClient
from src.ctrader.client import CTraderClient
from src.utils.logger import setup_logger


class TradingBot:
    """
    Main trading bot that orchestrates all agents and handles trading operations.
    """

    def __init__(self):
        self.orchestrator = AgentOrchestrator()
        self.market_data: Optional[MarketDataProvider] = None
        self.ctrader: Optional[CTraderClient] = None
        self.llm: Optional[LLMClient] = None
        self.is_running = False

    async def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("Initializing TradingAgents...")

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

        # Register agents
        self._register_agents()

        logger.info("TradingAgents initialized successfully")
        return True

    def _register_agents(self) -> None:
        """Register all trading agents with the orchestrator."""
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

        logger.info(f"Registered {len(self.orchestrator.agents)} agents")

    async def run_trading_cycle(self, instrument: str = "DE40") -> dict:
        """
        Run a complete trading cycle for an instrument.
        """
        logger.info(f"Running trading cycle for {instrument}")

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
            "news": [],  # Would be populated from news feed
            "social": {},  # Would be populated from social data
            "fear_greed_index": 50  # Would be fetched from API
        }

        # Run analysis cycle
        results = await self.orchestrator.run_analysis_cycle(market_data)

        # Execute trade if decision is to trade
        trading_decision = results.get("trading_decision", {})
        order = trading_decision.get("order")

        if order and self.ctrader:
            if trading_decision.get("decision", {}).get("action") in ["BUY", "SELL"]:
                execution_result = await self.ctrader.place_order(
                    symbol=order["instrument"],
                    direction=order["direction"],
                    volume=order["position_size"],
                    stop_loss=order.get("stop_loss"),
                    take_profit=order.get("take_profit")
                )
                results["execution"] = execution_result
                logger.info(f"Order execution: {execution_result}")

        return results

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
        logger.info(f"Starting trading bot for {instruments}")

        try:
            while self.is_running:
                for instrument in instruments:
                    try:
                        results = await self.run_trading_cycle(instrument)
                        decision = results.get("trading_decision", {}).get("decision", {})
                        logger.info(
                            f"{instrument}: {decision.get('action', 'HOLD')} "
                            f"(confidence: {decision.get('confidence', 0):.0%})"
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


async def main():
    """Main entry point."""
    bot = TradingBot()

    if not await bot.initialize():
        logger.error("Failed to initialize trading bot")
        sys.exit(1)

    # Run single analysis cycle for DAX40
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        results = await bot.run_trading_cycle("DE40")
        print("\n=== Trading Analysis Results ===")
        decision = results.get("trading_decision", {}).get("decision", {})
        print(f"Action: {decision.get('action', 'HOLD')}")
        print(f"Confidence: {decision.get('confidence', 0):.0%}")
        print(f"Reason: {decision.get('reason', 'N/A')}")
        await bot.shutdown()
    else:
        # Run continuously
        await bot.run(
            instruments=["DE40"],  # Start with DAX40
            interval_seconds=300  # 5 minute intervals
        )


if __name__ == "__main__":
    asyncio.run(main())
