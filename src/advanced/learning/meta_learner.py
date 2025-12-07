"""
Market Regime Meta-Learner for HRM Trading Bot

Implements meta-learning for rapid adaptation to new market conditions:
- MAML (Model-Agnostic Meta-Learning) inspired few-shot adaptation
- Market Regime Detection
- Task Distribution Learning
- Rapid Strategy Switching
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    """Represents a market regime/condition."""
    name: str
    volatility: float  # Average volatility
    trend_strength: float  # Trend strength (-1 to 1)
    volume_profile: str  # low, medium, high
    correlation_regime: str  # risk-on, risk-off, mixed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    confidence: float = 0.5

    def to_features(self) -> np.ndarray:
        """Convert regime to feature vector."""
        volume_map = {"low": 0.0, "medium": 0.5, "high": 1.0}
        corr_map = {"risk-off": -1.0, "mixed": 0.0, "risk-on": 1.0}

        return np.array([
            self.volatility,
            self.trend_strength,
            volume_map.get(self.volume_profile, 0.5),
            corr_map.get(self.correlation_regime, 0.0),
        ])


@dataclass
class Task:
    """A learning task representing a market regime."""
    regime: MarketRegime
    support_data: np.ndarray  # Training data
    support_labels: np.ndarray
    query_data: np.ndarray  # Validation data
    query_labels: np.ndarray


class FastAdaptiveModel:
    """
    Fast adaptive model that can quickly adjust to new market regimes.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # Hidden layer with ReLU
        h = np.maximum(0, x @ self.W1 + self.b1)
        # Output layer with sigmoid
        logits = h @ self.W2 + self.b2
        output = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        return output

    def get_params(self) -> list[np.ndarray]:
        """Get all parameters."""
        return [self.W1, self.b1, self.W2, self.b2]

    def set_params(self, params: list[np.ndarray]):
        """Set all parameters."""
        self.W1, self.b1, self.W2, self.b2 = params

    def clone(self) -> "FastAdaptiveModel":
        """Create a copy of the model."""
        new_model = FastAdaptiveModel(self.input_dim, self.hidden_dim, self.output_dim)
        new_model.W1 = self.W1.copy()
        new_model.b1 = self.b1.copy()
        new_model.W2 = self.W2.copy()
        new_model.b2 = self.b2.copy()
        return new_model

    def compute_gradients(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> tuple[list[np.ndarray], float]:
        """Compute gradients for given data."""
        batch_size = len(x)

        # Forward pass
        h = np.maximum(0, x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        predictions = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

        # Compute loss (binary cross-entropy)
        epsilon = 1e-7
        loss = -np.mean(y * np.log(predictions + epsilon) +
                       (1 - y) * np.log(1 - predictions + epsilon))

        # Backward pass
        d_logits = (predictions - y) / batch_size

        d_W2 = h.T @ d_logits
        d_b2 = np.sum(d_logits, axis=0)

        d_h = d_logits @ self.W2.T
        d_h[h <= 0] = 0  # ReLU gradient

        d_W1 = x.T @ d_h
        d_b1 = np.sum(d_h, axis=0)

        gradients = [d_W1, d_b1, d_W2, d_b2]
        return gradients, loss


class MarketRegimeDetector:
    """
    Detects current market regime from market data.
    """

    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period
        self.regime_history: deque = deque(maxlen=1000)
        self.current_regime: Optional[MarketRegime] = None

    def detect(self, market_data: np.ndarray) -> MarketRegime:
        """
        Detect current market regime from recent data.

        Args:
            market_data: Recent market data (OHLCV)

        Returns:
            Detected MarketRegime
        """
        if len(market_data) < self.lookback_period:
            return MarketRegime(
                name="unknown",
                volatility=0.0,
                trend_strength=0.0,
                volume_profile="medium",
                correlation_regime="mixed",
            )

        recent_data = market_data[-self.lookback_period:]

        # Calculate volatility (using returns)
        if recent_data.shape[1] >= 4:  # Has OHLC
            close_prices = recent_data[:, 3]  # Close column
        else:
            close_prices = recent_data[:, 0]

        returns = np.diff(close_prices) / close_prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized

        # Calculate trend strength
        x = np.arange(len(close_prices))
        slope, _ = np.polyfit(x, close_prices, 1)
        trend_strength = np.clip(slope / np.mean(close_prices) * 100, -1, 1)

        # Volume profile
        if recent_data.shape[1] >= 5:  # Has volume
            volume = recent_data[:, 4]
            avg_volume = np.mean(volume)
            if avg_volume > np.percentile(volume, 75):
                volume_profile = "high"
            elif avg_volume < np.percentile(volume, 25):
                volume_profile = "low"
            else:
                volume_profile = "medium"
        else:
            volume_profile = "medium"

        # Correlation regime (simplified - based on volatility and trend)
        if volatility > 0.3 and trend_strength < 0:
            correlation_regime = "risk-off"
        elif volatility < 0.15 and trend_strength > 0:
            correlation_regime = "risk-on"
        else:
            correlation_regime = "mixed"

        # Determine regime name
        if volatility > 0.3:
            if trend_strength > 0.3:
                regime_name = "volatile_bull"
            elif trend_strength < -0.3:
                regime_name = "volatile_bear"
            else:
                regime_name = "volatile_range"
        elif volatility < 0.1:
            if trend_strength > 0.3:
                regime_name = "quiet_bull"
            elif trend_strength < -0.3:
                regime_name = "quiet_bear"
            else:
                regime_name = "quiet_range"
        else:
            if trend_strength > 0.3:
                regime_name = "trending_bull"
            elif trend_strength < -0.3:
                regime_name = "trending_bear"
            else:
                regime_name = "ranging"

        regime = MarketRegime(
            name=regime_name,
            volatility=float(volatility),
            trend_strength=float(trend_strength),
            volume_profile=volume_profile,
            correlation_regime=correlation_regime,
            start_time=datetime.now(),
            confidence=0.8,
        )

        self.regime_history.append(regime)
        self.current_regime = regime

        return regime

    def get_regime_transition_prob(self, from_regime: str, to_regime: str) -> float:
        """Get probability of transitioning from one regime to another."""
        if len(self.regime_history) < 2:
            return 0.5

        transitions = 0
        total = 0

        regime_list = list(self.regime_history)
        for i in range(len(regime_list) - 1):
            if regime_list[i].name == from_regime:
                total += 1
                if regime_list[i + 1].name == to_regime:
                    transitions += 1

        return transitions / total if total > 0 else 0.5


class MarketRegimeMetaLearner:
    """
    Meta-Learning system for rapid adaptation to new market regimes.

    Implements MAML-inspired algorithm for few-shot learning on market data.
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 64,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        inner_steps: int = 5,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps

        # Base model
        self.base_model = FastAdaptiveModel(input_dim, hidden_dim)

        # Regime detector
        self.regime_detector = MarketRegimeDetector()

        # Task buffer for meta-training
        self.task_buffer: deque = deque(maxlen=100)

        # Regime-specific adapted models
        self.adapted_models: dict[str, FastAdaptiveModel] = {}

        # Statistics
        self.meta_updates = 0
        self.total_adaptations = 0

        logger.info(
            f"MarketRegimeMetaLearner initialized: "
            f"inner_lr={inner_lr}, meta_lr={meta_lr}"
        )

    def few_shot_adaptation(
        self,
        support_data: np.ndarray,
        support_labels: np.ndarray,
        query_data: np.ndarray,
        num_steps: Optional[int] = None
    ) -> tuple[FastAdaptiveModel, float]:
        """
        Adapt model to new regime with few examples.

        Args:
            support_data: Few-shot training examples
            support_labels: Labels for training examples
            query_data: Query examples for evaluation
            num_steps: Number of adaptation steps

        Returns:
            Tuple of (adapted model, query loss)
        """
        steps = num_steps or self.inner_steps

        # Clone base model for adaptation
        adapted_model = self.base_model.clone()

        # Inner loop: adapt to support set
        for step in range(steps):
            gradients, loss = adapted_model.compute_gradients(support_data, support_labels)

            # Update parameters
            params = adapted_model.get_params()
            new_params = [p - self.inner_lr * g for p, g in zip(params, gradients)]
            adapted_model.set_params(new_params)

        # Evaluate on query set
        if len(query_data) > 0:
            predictions = adapted_model.forward(query_data)
            query_loss = np.mean((predictions.flatten() - support_labels[:len(predictions)]) ** 2)
        else:
            query_loss = 0.0

        self.total_adaptations += 1
        return adapted_model, query_loss

    def meta_update(self, tasks: list[Task]) -> float:
        """
        Perform meta-update across multiple tasks/regimes.

        Args:
            tasks: List of tasks (market regimes) to learn from

        Returns:
            Average meta loss
        """
        if not tasks:
            return 0.0

        # Accumulate gradients across tasks
        meta_gradients = [np.zeros_like(p) for p in self.base_model.get_params()]
        total_loss = 0.0

        for task in tasks:
            # Adapt to task
            adapted_model, query_loss = self.few_shot_adaptation(
                task.support_data,
                task.support_labels,
                task.query_data,
            )

            # Compute gradients on query set with adapted model
            gradients, loss = adapted_model.compute_gradients(
                task.query_data,
                task.query_labels
            )

            # Accumulate gradients
            for i, g in enumerate(gradients):
                meta_gradients[i] += g

            total_loss += loss

        # Average gradients
        n_tasks = len(tasks)
        meta_gradients = [g / n_tasks for g in meta_gradients]

        # Update base model
        params = self.base_model.get_params()
        new_params = [p - self.meta_lr * g for p, g in zip(params, meta_gradients)]
        self.base_model.set_params(new_params)

        self.meta_updates += 1
        avg_loss = total_loss / n_tasks

        logger.debug(f"Meta update {self.meta_updates}: loss={avg_loss:.4f}")
        return avg_loss

    def adapt_to_regime(
        self,
        regime: MarketRegime,
        market_data: np.ndarray,
        labels: np.ndarray
    ) -> FastAdaptiveModel:
        """
        Adapt model to specific market regime.

        Args:
            regime: Target market regime
            market_data: Market data for adaptation
            labels: Labels for market data

        Returns:
            Adapted model for this regime
        """
        # Split data into support and query
        n = len(market_data)
        support_size = min(20, n // 2)

        support_data = market_data[:support_size]
        support_labels = labels[:support_size]
        query_data = market_data[support_size:]

        # Perform few-shot adaptation
        adapted_model, loss = self.few_shot_adaptation(
            support_data,
            support_labels,
            query_data,
            num_steps=10  # More steps for regime adaptation
        )

        # Cache adapted model
        self.adapted_models[regime.name] = adapted_model

        logger.info(f"Adapted to regime '{regime.name}' with loss={loss:.4f}")
        return adapted_model

    def predict(
        self,
        features: np.ndarray,
        regime: Optional[MarketRegime] = None
    ) -> np.ndarray:
        """
        Make prediction, optionally using regime-specific model.

        Args:
            features: Input features
            regime: Optional regime to use specific model

        Returns:
            Predictions
        """
        if regime and regime.name in self.adapted_models:
            model = self.adapted_models[regime.name]
        else:
            model = self.base_model

        return model.forward(features)

    def add_task(self, task: Task):
        """Add task to buffer for meta-training."""
        self.task_buffer.append(task)

    async def continuous_meta_learning(self, update_interval: int = 100):
        """
        Continuously perform meta-learning updates.

        Args:
            update_interval: Number of tasks between meta-updates
        """
        while True:
            if len(self.task_buffer) >= update_interval:
                tasks = list(self.task_buffer)[-update_interval:]
                loss = self.meta_update(tasks)
                logger.info(f"Continuous meta-update: loss={loss:.4f}")

            await asyncio.sleep(60)  # Check every minute

    def get_state(self) -> dict:
        """Get meta-learner state."""
        return {
            "meta_updates": self.meta_updates,
            "total_adaptations": self.total_adaptations,
            "num_cached_regimes": len(self.adapted_models),
            "task_buffer_size": len(self.task_buffer),
            "base_model_params": [p.tolist() for p in self.base_model.get_params()],
        }

    def load_state(self, state: dict):
        """Load meta-learner state."""
        self.meta_updates = state.get("meta_updates", 0)
        self.total_adaptations = state.get("total_adaptations", 0)

        if "base_model_params" in state:
            params = [np.array(p) for p in state["base_model_params"]]
            self.base_model.set_params(params)


class RegimeAwareTrader:
    """
    Trading strategy that uses meta-learner for regime-aware predictions.
    """

    def __init__(self, meta_learner: MarketRegimeMetaLearner):
        self.meta_learner = meta_learner
        self.current_regime: Optional[MarketRegime] = None
        self.regime_confidence_threshold = 0.7

    async def trade_with_regime_awareness(
        self,
        market_data: np.ndarray,
        features: np.ndarray
    ) -> dict:
        """
        Make trading decision with regime awareness.

        Args:
            market_data: Raw market data for regime detection
            features: Processed features for prediction

        Returns:
            Trading decision with regime information
        """
        # Detect current regime
        regime = self.meta_learner.regime_detector.detect(market_data)

        # Check if regime changed
        regime_changed = (
            self.current_regime is None or
            self.current_regime.name != regime.name
        )

        if regime_changed:
            logger.info(f"Regime changed to: {regime.name}")
            self.current_regime = regime

            # Quick adaptation to new regime if we have data
            if len(market_data) > 20:
                # Create pseudo-labels from price direction
                labels = (np.diff(market_data[-21:, 3]) > 0).astype(float)
                self.meta_learner.adapt_to_regime(
                    regime,
                    features[-20:],
                    labels
                )

        # Make prediction with regime-aware model
        prediction = self.meta_learner.predict(features[-1:], regime)

        # Adjust confidence based on regime stability
        confidence = float(prediction[0, 0]) if prediction.size > 0 else 0.5
        if regime.confidence < self.regime_confidence_threshold:
            confidence *= regime.confidence

        return {
            "prediction": float(prediction[0, 0]) if prediction.size > 0 else 0.5,
            "confidence": confidence,
            "regime": regime.name,
            "regime_confidence": regime.confidence,
            "regime_volatility": regime.volatility,
            "regime_trend": regime.trend_strength,
        }
