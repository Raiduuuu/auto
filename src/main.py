"""
TradingAgents - Main Entry Point
Multi-Agent LLM Financial Trading Framework

Enhanced with Advanced AI Components:
- Continuous Learning Engine
- Meta-Learning for Market Regimes
- Real-Time Performance Monitoring
- Causal Market Analysis
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

# Advanced components
from src.advanced.learning.continuous_learning import ContinuousLearningEngine, LearningMode
from src.advanced.learning.meta_learner import MarketRegimeMetaLearner
from src.advanced.monitoring.performance_monitor import RealTimePerformanceMonitor
from src.advanced.analysis.causal_analyzer import CausalMarketAnalyzer


class TradingBot:
    """
    Main trading bot that orchestrates all agents and handles trading operations.

    Enhanced with:
    - ContinuousLearningEngine for real-time adaptive learning
    - MarketRegimeMetaLearner for regime detection and adaptation
    - RealTimePerformanceMonitor for live performance tracking
    - CausalMarketAnalyzer for understanding market dynamics
    """

    def __init__(self, use_advanced: bool = True):
        self.orchestrator = AgentOrchestrator()
        self.market_data: Optional[MarketDataProvider] = None
        self.ctrader: Optional[CTraderClient] = None
        self.llm: Optional[LLMClient] = None
        self.is_running = False
        self.use_advanced = use_advanced

        # Advanced components
        self.learning_engine: Optional[ContinuousLearningEngine] = None
        self.meta_learner: Optional[MarketRegimeMetaLearner] = None
        self.performance_monitor: Optional[RealTimePerformanceMonitor] = None
        self.causal_analyzer: Optional[CausalMarketAnalyzer] = None

        # Trading state
        self.current_regime = None
        self.trade_history = []

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

        # Initialize advanced components
        if self.use_advanced:
            await self._initialize_advanced_components()

        # Register agents
        self._register_agents()

        logger.info("TradingAgents initialized successfully")
        return True

    async def _initialize_advanced_components(self):
        """Initialize advanced AI components."""
        logger.info("Initializing advanced AI components...")

        # Feature names for analysis
        feature_names = [
            "open", "high", "low", "close", "volume",
            "sma_20", "sma_50", "ema_12", "ema_26",
            "rsi", "macd", "macd_signal",
            "atr", "adx", "stoch_k", "stoch_d",
            "bb_upper", "bb_lower", "pivot", "resistance"
        ]

        # Continuous Learning Engine
        self.learning_engine = ContinuousLearningEngine(
            input_dim=20,
            adaptation_rate=1e-5,
            buffer_size=50000,
            mode=LearningMode.ACTIVE,
        )
        logger.info("ContinuousLearningEngine initialized")

        # Meta-Learner for Market Regimes
        self.meta_learner = MarketRegimeMetaLearner(
            input_dim=20,
            hidden_dim=64,
            inner_lr=0.01,
            meta_lr=0.001,
        )
        logger.info("MarketRegimeMetaLearner initialized")

        # Real-Time Performance Monitor
        self.performance_monitor = RealTimePerformanceMonitor(
            collection_interval=1.0,
            analysis_interval=60.0,
        )
        # Register metric collector
        self.performance_monitor.register_collector(self._collect_trading_metrics)
        logger.info("RealTimePerformanceMonitor initialized")

        # Causal Market Analyzer
        self.causal_analyzer = CausalMarketAnalyzer(
            feature_names=feature_names,
        )
        logger.info("CausalMarketAnalyzer initialized")

    def _collect_trading_metrics(self) -> dict:
        """Collect trading metrics for performance monitoring."""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "win_rate": 0.5,
                "profit": 0.0,
            }

        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t.get("profit", 0) > 0)
        total_profit = sum(t.get("profit", 0) for t in self.trade_history)

        return {
            "total_trades": total_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0.5,
            "profit": total_profit,
            "current_regime": self.current_regime.name if self.current_regime else "unknown",
        }

    def _register_agents(self) -> None:
        """Register all trading agents with the orchestrator."""
        # Technical Analyst
        technical = TechnicalAnalystAgent(llm_client=self.llm)
        self.orchestrator.register_agent(technical)

        # Sentiment Analyst
        sentiment = SentimentAnalystAgent(llm_client=self.llm)
        self.orchestrator.register_agent(sentiment)

        # Risk Manager (with advanced risk management)
        risk_manager = RiskManagerAgent(
            llm_client=self.llm,
            max_risk_per_trade=settings.trading.default_risk_percent,
            max_daily_loss=settings.trading.max_daily_loss_percent,
            max_drawdown=settings.trading.max_drawdown_percent,
            use_advanced=self.use_advanced
        )
        self.orchestrator.register_agent(risk_manager)

        # Trader (with advanced prediction engine)
        trader = TraderAgent(
            llm_client=self.llm,
            use_advanced=self.use_advanced
        )
        self.orchestrator.register_agent(trader)

        logger.info(f"Registered {len(self.orchestrator.agents)} agents")
        if self.use_advanced:
            logger.info("Advanced AI features enabled for RiskManager and Trader")

    async def run_trading_cycle(self, instrument: str = "DE40") -> dict:
        """
        Run a complete trading cycle for an instrument.
        Enhanced with advanced AI components for learning and regime detection.
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

        # Advanced: Detect market regime
        if self.use_advanced and self.meta_learner:
            import numpy as np
            ohlcv_array = np.column_stack([
                ohlcv['open'].values[-50:],
                ohlcv['high'].values[-50:],
                ohlcv['low'].values[-50:],
                ohlcv['close'].values[-50:],
                ohlcv['volume'].values[-50:] if 'volume' in ohlcv else np.ones(50)
            ])
            self.current_regime = self.meta_learner.regime_detector.detect(ohlcv_array)
            market_data["regime"] = {
                "name": self.current_regime.name,
                "volatility": self.current_regime.volatility,
                "trend_strength": self.current_regime.trend_strength,
                "confidence": self.current_regime.confidence,
            }
            logger.info(f"Detected market regime: {self.current_regime.name}")

        # Run analysis cycle
        results = await self.orchestrator.run_analysis_cycle(market_data)

        # Add regime info to results
        if self.current_regime:
            results["market_regime"] = market_data.get("regime", {})

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

                # Track trade for learning
                self.trade_history.append({
                    "instrument": instrument,
                    "direction": order["direction"],
                    "entry_price": order["entry_price"],
                    "timestamp": order["created_at"],
                    "regime": self.current_regime.name if self.current_regime else "unknown",
                })

        # Advanced: Learn from trading cycle
        if self.use_advanced and self.learning_engine:
            await self._learn_from_cycle(market_data, results)

        return results

    async def _learn_from_cycle(self, market_data: dict, results: dict):
        """Learn from the trading cycle using advanced learning components."""
        import numpy as np

        # Extract features
        features = []
        indicators = market_data.get("indicators", {})
        for key in ["sma_20", "sma_50", "ema_12", "ema_26", "rsi", "macd", "atr", "adx"]:
            features.append(indicators.get(key, 0))

        # Pad to input_dim
        while len(features) < 20:
            features.append(0)
        features = np.array(features[:20], dtype=np.float64)

        # Get prediction and actual (simplified - would be actual price movement)
        decision = results.get("trading_decision", {}).get("decision", {})
        confidence = decision.get("confidence", 0.5)

        # Learn from experience (simplified - actual outcome would come later)
        prediction = 0.7 if decision.get("action") == "BUY" else (0.3 if decision.get("action") == "SELL" else 0.5)
        self.learning_engine.learn_from_experience(
            features=features,
            prediction=prediction,
            actual=0.5,  # Placeholder - would be actual market movement
            confidence=confidence,
        )

    async def run(
        self,
        instruments: list = None,
        interval_seconds: int = 300
    ) -> None:
        """
        Run the trading bot continuously.
        Enhanced with performance monitoring.
        """
        if instruments is None:
            instruments = settings.trading.instruments

        self.is_running = True
        logger.info(f"Starting trading bot for {instruments}")

        # Start performance monitoring
        if self.use_advanced and self.performance_monitor:
            await self.performance_monitor.start(self)
            logger.info("Performance monitoring started")

        try:
            while self.is_running:
                for instrument in instruments:
                    try:
                        results = await self.run_trading_cycle(instrument)
                        decision = results.get("trading_decision", {}).get("decision", {})

                        # Enhanced logging with regime info
                        regime_info = ""
                        if "market_regime" in results:
                            regime_info = f" [regime: {results['market_regime'].get('name', 'unknown')}]"

                        logger.info(
                            f"{instrument}: {decision.get('action', 'HOLD')} "
                            f"(confidence: {decision.get('confidence', 0):.0%}){regime_info}"
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

        # Stop performance monitoring
        if self.performance_monitor:
            await self.performance_monitor.stop()
            logger.info("Performance monitoring stopped")

        if self.ctrader:
            await self.ctrader.disconnect()

        # Log final statistics
        if self.trade_history:
            total_trades = len(self.trade_history)
            logger.info(f"Total trades executed: {total_trades}")

        if self.learning_engine:
            state = self.learning_engine.get_state()
            logger.info(f"Learning engine: {state['total_experiences']} experiences processed")

        logger.info("TradingAgents shutdown complete")

    def get_advanced_status(self) -> dict:
        """Get status of advanced AI components."""
        status = {
            "advanced_enabled": self.use_advanced,
            "current_regime": self.current_regime.name if self.current_regime else None,
            "total_trades": len(self.trade_history),
        }

        if self.learning_engine:
            status["learning_engine"] = self.learning_engine.get_state()

        if self.meta_learner:
            status["meta_learner"] = self.meta_learner.get_state()

        if self.performance_monitor:
            status["performance"] = self.performance_monitor.get_dashboard_data()

        return status


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
