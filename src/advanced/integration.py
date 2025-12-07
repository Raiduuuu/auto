"""
Advanced Trading System Integration Module

Integrates all advanced components into a unified trading system:
- Continuous Learning Engine
- Neuroplasticity Engine
- Quantum-Inspired Optimizer
- Causal Market Analyzer
- Meta-Learning for Market Regimes
- Advanced Risk Management
- Real-Time Performance Monitoring
- Advanced Prediction Engine
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
import numpy as np
import logging

from .learning.continuous_learning import ContinuousLearningEngine, LearningMode
from .learning.neuroplasticity import NeuroplasticityEngine, AdaptiveNeuroplasticityController
from .learning.meta_learner import MarketRegimeMetaLearner, MarketRegime, Task
from .optimization.quantum_optimizer import QuantumInspiredOptimizer, AdaptiveQuantumOptimizer
from .analysis.causal_analyzer import CausalMarketAnalyzer
from .analysis.advanced_risk import AdvancedRiskManager, MarketConditions, Position
from .monitoring.performance_monitor import RealTimePerformanceMonitor
from .prediction.advanced_prediction import AdvancedPredictionEngine

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Trading signal from the integrated system."""
    direction: str  # "buy", "sell", "hold"
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    horizon: str
    regime: str
    risk_assessment: dict
    predictions: dict
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemState:
    """State of the integrated trading system."""
    is_active: bool
    current_regime: Optional[str]
    learning_mode: str
    total_predictions: int
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    health_score: float
    last_update: datetime


class AdvancedTradingSystem:
    """
    Integrated Advanced Trading System.

    Combines all advanced components for a comprehensive
    AI-powered trading system.
    """

    def __init__(
        self,
        input_dim: int = 50,
        feature_names: list[str] = None,
        config: dict = None,
    ):
        self.input_dim = input_dim
        self.feature_names = feature_names or [f"feature_{i}" for i in range(input_dim)]
        self.config = config or {}

        # Initialize all components
        self._initialize_components()

        # State
        self.is_running = False
        self.current_regime: Optional[MarketRegime] = None
        self.market_data_buffer: list = []
        self.trade_history: list = []

        # Statistics
        self.total_predictions = 0
        self.correct_predictions = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0

        logger.info("AdvancedTradingSystem initialized")

    def _initialize_components(self):
        """Initialize all advanced components."""
        # Continuous Learning
        self.learning_engine = ContinuousLearningEngine(
            input_dim=self.input_dim,
            adaptation_rate=self.config.get("learning_rate", 1e-5),
            buffer_size=self.config.get("buffer_size", 50000),
            mode=LearningMode.ACTIVE,
        )

        # Neuroplasticity
        self.neuroplasticity = NeuroplasticityEngine(
            num_neurons=self.config.get("num_neurons", 100),
            learning_rate=self.config.get("neuro_lr", 0.01),
        )
        self.neuro_controller = AdaptiveNeuroplasticityController(self.neuroplasticity)

        # Quantum Optimizer
        self.optimizer = AdaptiveQuantumOptimizer(
            learning_rate=self.config.get("optimizer_lr", 0.01),
            quantum_strength=self.config.get("quantum_strength", 0.1),
        )

        # Causal Analyzer
        self.causal_analyzer = CausalMarketAnalyzer(
            feature_names=self.feature_names,
        )

        # Meta-Learner
        self.meta_learner = MarketRegimeMetaLearner(
            input_dim=self.input_dim,
            hidden_dim=self.config.get("hidden_dim", 64),
            inner_lr=self.config.get("inner_lr", 0.01),
            meta_lr=self.config.get("meta_lr", 0.001),
        )

        # Risk Manager
        self.risk_manager = AdvancedRiskManager(
            max_portfolio_risk=self.config.get("max_portfolio_risk", 0.02),
            max_single_position=self.config.get("max_single_position", 0.005),
            max_daily_loss=self.config.get("max_daily_loss", 0.05),
            max_drawdown=self.config.get("max_drawdown", 0.10),
        )

        # Performance Monitor
        self.monitor = RealTimePerformanceMonitor(
            collection_interval=self.config.get("monitor_interval", 1.0),
            analysis_interval=self.config.get("analysis_interval", 60.0),
        )

        # Prediction Engine
        self.prediction_engine = AdvancedPredictionEngine(
            input_dim=self.input_dim,
            horizons=self.config.get("horizons", ["1min", "5min", "15min", "1h", "4h", "1d"]),
        )

        # Register metric collector
        self.monitor.register_collector(self._collect_system_metrics)

    def _collect_system_metrics(self) -> dict:
        """Collect metrics for performance monitoring."""
        metrics = {
            "accuracy": self.correct_predictions / max(self.total_predictions, 1),
            "win_rate": self.winning_trades / max(self.total_trades, 1),
            "total_trades": self.total_trades,
            "total_predictions": self.total_predictions,
            "profit": self.total_profit,
        }

        # Add risk manager metrics
        risk_state = self.risk_manager.get_state()
        metrics["open_positions"] = risk_state["positions_count"]
        metrics["drawdown"] = (
            (risk_state["peak_equity"] - risk_state["current_equity"]) /
            risk_state["peak_equity"]
            if risk_state["peak_equity"] > 0 else 0
        )

        return metrics

    async def process_market_data(
        self,
        market_data: dict,
        account_balance: float,
    ) -> Optional[TradingSignal]:
        """
        Process incoming market data and generate trading signal.

        Args:
            market_data: Dictionary containing OHLCV and indicators
            account_balance: Current account balance

        Returns:
            TradingSignal if trade opportunity found, None otherwise
        """
        try:
            # Extract features
            features = self._extract_features(market_data)
            self.market_data_buffer.append(features)

            # Limit buffer size
            if len(self.market_data_buffer) > 1000:
                self.market_data_buffer = self.market_data_buffer[-1000:]

            # Detect market regime
            if len(self.market_data_buffer) >= 50:
                data_array = np.array(self.market_data_buffer[-50:])
                self.current_regime = self.meta_learner.regime_detector.detect(data_array)

            # Get market conditions for risk assessment
            market_conditions = self._assess_market_conditions(market_data)

            # Multi-horizon prediction
            prediction = self.prediction_engine.multi_horizon_prediction(features)
            self.total_predictions += 1

            # Determine trading direction
            direction = self._determine_direction(prediction)
            if direction == "hold":
                return None

            # Get entry price and ATR
            entry_price = market_data.get("close", market_data.get("price", 0))
            atr = market_data.get("atr", entry_price * 0.01)

            # Risk assessment
            risk_assessment = self.risk_manager.assess_trade(
                instrument=market_data.get("instrument", "UNKNOWN"),
                direction="long" if direction == "buy" else "short",
                entry_price=entry_price,
                prediction_confidence=prediction.predictions[prediction.recommended_horizon].confidence,
                market_conditions=market_conditions,
                account_balance=account_balance,
                atr=atr,
            )

            if not risk_assessment.approved:
                logger.info(f"Trade rejected: {risk_assessment.reason}")
                return None

            # Create trading signal
            signal = TradingSignal(
                direction=direction,
                confidence=prediction.predictions[prediction.recommended_horizon].confidence,
                entry_price=entry_price,
                stop_loss=risk_assessment.stop_loss,
                take_profit=risk_assessment.take_profit,
                position_size=risk_assessment.position_size,
                horizon=prediction.recommended_horizon,
                regime=self.current_regime.name if self.current_regime else "unknown",
                risk_assessment=risk_assessment.__dict__,
                predictions={
                    h: {"value": p.value, "confidence": p.confidence}
                    for h, p in prediction.predictions.items()
                },
            )

            return signal

        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return None

    def _extract_features(self, market_data: dict) -> np.ndarray:
        """Extract feature vector from market data."""
        features = []

        # Price features
        for key in ["open", "high", "low", "close", "volume"]:
            if key in market_data:
                features.append(float(market_data[key]))

        # Technical indicators
        indicator_keys = [
            "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
            "rsi", "macd", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower",
            "atr", "adx", "stoch_k", "stoch_d",
        ]
        for key in indicator_keys:
            if key in market_data:
                features.append(float(market_data[key]))

        # Pad or truncate to input_dim
        features = np.array(features, dtype=np.float64)
        if len(features) < self.input_dim:
            features = np.pad(features, (0, self.input_dim - len(features)))
        elif len(features) > self.input_dim:
            features = features[:self.input_dim]

        return features

    def _assess_market_conditions(self, market_data: dict) -> MarketConditions:
        """Assess current market conditions."""
        # Calculate volatility
        atr = market_data.get("atr", 0)
        close = market_data.get("close", 1)
        volatility = atr / close if close > 0 else 0.02

        # Estimate liquidity (simplified)
        volume = market_data.get("volume", 0)
        avg_volume = market_data.get("avg_volume", volume)
        liquidity = min(volume / avg_volume, 1.0) if avg_volume > 0 else 0.5

        # Trend strength from ADX
        adx = market_data.get("adx", 25)
        trend_strength = (adx - 25) / 75  # Normalize ADX

        # Determine correlation regime
        rsi = market_data.get("rsi", 50)
        if rsi > 70:
            correlation_regime = "risk-on"
        elif rsi < 30:
            correlation_regime = "risk-off"
        else:
            correlation_regime = "mixed"

        return MarketConditions(
            volatility=volatility,
            liquidity=liquidity,
            trend_strength=trend_strength,
            correlation_regime=correlation_regime,
            spread_pct=market_data.get("spread_pct", 0.0001),
        )

    def _determine_direction(self, prediction) -> str:
        """Determine trading direction from prediction."""
        # Get the recommended horizon prediction
        recommended_pred = prediction.predictions.get(
            prediction.recommended_horizon
        )

        if recommended_pred is None:
            return "hold"

        # Check consistency
        if prediction.consistency_score < 0.5:
            return "hold"

        # Check confidence
        if recommended_pred.confidence < 0.6:
            return "hold"

        # Determine direction
        if prediction.dominant_direction == "bullish" and recommended_pred.value > 0.6:
            return "buy"
        elif prediction.dominant_direction == "bearish" and recommended_pred.value < 0.4:
            return "sell"

        return "hold"

    async def learn_from_trade(
        self,
        features: np.ndarray,
        prediction: float,
        actual_outcome: float,
        profit: float,
    ):
        """
        Learn from a completed trade.

        Args:
            features: Feature vector used for prediction
            prediction: Predicted value
            actual_outcome: Actual outcome (1 = up, 0 = down)
            profit: Realized profit/loss
        """
        # Update prediction accuracy
        if (prediction > 0.5 and actual_outcome > 0.5) or \
           (prediction < 0.5 and actual_outcome < 0.5):
            self.correct_predictions += 1

        # Update trade statistics
        self.total_trades += 1
        self.total_profit += profit
        if profit > 0:
            self.winning_trades += 1

        # Update learning engine
        self.learning_engine.learn_from_experience(
            features, prediction, actual_outcome, confidence=0.8
        )

        # Update neuroplasticity engine
        input_act = features[:self.neuroplasticity.num_neurons // 2]
        target = np.array([actual_outcome])
        output = self.neuroplasticity.forward_pass(input_act)
        self.neuroplasticity.backward_learning(input_act, target, output[:1])

        # Update prediction engine
        self.prediction_engine.update_models(features, actual_outcome)

        # Update meta-learner if we have regime information
        if self.current_regime:
            task = Task(
                regime=self.current_regime,
                support_data=features.reshape(1, -1),
                support_labels=np.array([actual_outcome]),
                query_data=features.reshape(1, -1),
                query_labels=np.array([actual_outcome]),
            )
            self.meta_learner.add_task(task)

        logger.debug(
            f"Learned from trade: prediction={prediction:.3f}, "
            f"actual={actual_outcome:.3f}, profit={profit:.2f}"
        )

    async def start(self):
        """Start the trading system."""
        self.is_running = True

        # Start performance monitoring
        await self.monitor.start(self)

        # Start continuous meta-learning
        asyncio.create_task(
            self.meta_learner.continuous_meta_learning(update_interval=100)
        )

        logger.info("AdvancedTradingSystem started")

    async def stop(self):
        """Stop the trading system."""
        self.is_running = False
        await self.monitor.stop()
        logger.info("AdvancedTradingSystem stopped")

    def get_state(self) -> SystemState:
        """Get current system state."""
        analysis = self.monitor.get_current_analysis()

        return SystemState(
            is_active=self.is_running,
            current_regime=self.current_regime.name if self.current_regime else None,
            learning_mode=self.learning_engine.mode.value,
            total_predictions=self.total_predictions,
            total_trades=self.total_trades,
            win_rate=self.winning_trades / max(self.total_trades, 1),
            sharpe_ratio=analysis.current_performance.sharpe_ratio if analysis else 0,
            max_drawdown=analysis.current_performance.drawdown if analysis else 0,
            health_score=analysis.health_score if analysis else 0.5,
            last_update=datetime.now(),
        )

    def get_metrics(self) -> dict:
        """Get system metrics for monitoring."""
        return {
            "accuracy": self.correct_predictions / max(self.total_predictions, 1),
            "win_rate": self.winning_trades / max(self.total_trades, 1),
            "total_trades": self.total_trades,
            "total_profit": self.total_profit,
            "sharpe_ratio": self._calculate_sharpe(),
            "current_regime": self.current_regime.name if self.current_regime else "unknown",
        }

    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio from trade history."""
        if len(self.trade_history) < 30:
            return 0.0

        returns = [t.get("return", 0) for t in self.trade_history[-100:]]
        if np.std(returns) == 0:
            return 0.0

        return np.mean(returns) / np.std(returns) * np.sqrt(252)

    def get_component_states(self) -> dict:
        """Get states of all components."""
        return {
            "learning_engine": self.learning_engine.get_state(),
            "neuroplasticity": self.neuroplasticity.get_state(),
            "optimizer": self.optimizer.get_state(),
            "causal_analyzer": self.causal_analyzer.get_summary(),
            "meta_learner": self.meta_learner.get_state(),
            "risk_manager": self.risk_manager.get_state(),
            "monitor": self.monitor.get_state(),
            "prediction_engine": self.prediction_engine.get_state(),
        }


async def create_advanced_trading_system(
    config: dict = None,
) -> AdvancedTradingSystem:
    """
    Factory function to create and initialize an advanced trading system.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized AdvancedTradingSystem
    """
    default_config = {
        "input_dim": 50,
        "learning_rate": 1e-5,
        "buffer_size": 50000,
        "num_neurons": 100,
        "neuro_lr": 0.01,
        "optimizer_lr": 0.01,
        "quantum_strength": 0.1,
        "hidden_dim": 64,
        "inner_lr": 0.01,
        "meta_lr": 0.001,
        "max_portfolio_risk": 0.02,
        "max_single_position": 0.005,
        "max_daily_loss": 0.05,
        "max_drawdown": 0.10,
        "monitor_interval": 1.0,
        "analysis_interval": 60.0,
        "horizons": ["1min", "5min", "15min", "1h", "4h", "1d"],
    }

    if config:
        default_config.update(config)

    # Define feature names
    feature_names = [
        "open", "high", "low", "close", "volume",
        "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
        "rsi", "macd", "macd_signal", "macd_histogram",
        "bb_upper", "bb_middle", "bb_lower",
        "atr", "adx", "stoch_k", "stoch_d",
    ]
    # Pad to input_dim
    while len(feature_names) < default_config["input_dim"]:
        feature_names.append(f"feature_{len(feature_names)}")

    system = AdvancedTradingSystem(
        input_dim=default_config["input_dim"],
        feature_names=feature_names,
        config=default_config,
    )

    return system
