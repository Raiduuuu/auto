"""
Continuous Learning Engine for HRM Trading Bot

Provides real-time adaptive learning from market data streams.
Implements online learning algorithms for continuous model improvement.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Callable, Optional
import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Learning modes for the continuous learning engine."""
    PASSIVE = "passive"  # Learn from observations only
    ACTIVE = "active"  # Learn and provide recommendations
    AGGRESSIVE = "aggressive"  # Learn with higher adaptation rate


@dataclass
class Experience:
    """Represents a single learning experience."""
    features: np.ndarray
    prediction: float
    actual: float
    confidence: float
    error: float
    timestamp: datetime
    metadata: dict = field(default_factory=dict)

    @property
    def reward(self) -> float:
        """Calculate reward based on prediction accuracy."""
        return 1.0 - min(abs(self.error), 1.0)


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
        }


class PerformanceTracker:
    """Tracks and analyzes model performance over time."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions: deque = deque(maxlen=window_size)
        self.actuals: deque = deque(maxlen=window_size)
        self.returns: deque = deque(maxlen=window_size)
        self.equity_curve: list = [1.0]

    def update(self, prediction: float, actual: float, return_pct: float = 0.0):
        """Update tracker with new prediction-actual pair."""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.returns.append(return_pct)

        # Update equity curve
        new_equity = self.equity_curve[-1] * (1 + return_pct)
        self.equity_curve.append(new_equity)

    def get_metrics(self) -> PerformanceMetrics:
        """Calculate current performance metrics."""
        if len(self.predictions) < 10:
            return PerformanceMetrics()

        predictions = np.array(self.predictions)
        actuals = np.array(self.actuals)
        returns = np.array(self.returns)

        # Binary classification metrics (for direction prediction)
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)

        correct = pred_direction == actual_direction
        accuracy = np.mean(correct)

        # Win rate
        wins = np.sum(returns > 0)
        losses = np.sum(returns < 0)
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.5

        # Sharpe ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = np.max(drawdown)

        # Profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return PerformanceMetrics(
            accuracy=float(accuracy),
            win_rate=float(win_rate),
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_dd),
            profit_factor=float(profit_factor),
        )


class OnlineLearningModel:
    """Online learning model with incremental updates."""

    def __init__(self, input_dim: int, learning_rate: float = 1e-4):
        self.input_dim = input_dim
        self.learning_rate = learning_rate

        # Initialize weights with small random values
        self.weights = np.random.randn(input_dim) * 0.01
        self.bias = 0.0

        # Running statistics for normalization
        self.running_mean = np.zeros(input_dim)
        self.running_var = np.ones(input_dim)
        self.n_samples = 0

        # Momentum for gradient updates
        self.momentum = 0.9
        self.velocity_w = np.zeros(input_dim)
        self.velocity_b = 0.0

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        """Make prediction with confidence estimate."""
        # Normalize features
        normalized = (features - self.running_mean) / (np.sqrt(self.running_var) + 1e-8)

        # Linear prediction with sigmoid activation
        logit = np.dot(normalized, self.weights) + self.bias
        prediction = 1 / (1 + np.exp(-np.clip(logit, -500, 500)))

        # Confidence based on weight magnitude and feature alignment
        weight_magnitude = np.linalg.norm(self.weights)
        feature_alignment = abs(np.dot(normalized, self.weights / (weight_magnitude + 1e-8)))
        confidence = min(0.5 + feature_alignment * 0.5, 0.99)

        return prediction, confidence

    def update(self, features: np.ndarray, target: float, learning_rate: Optional[float] = None):
        """Incremental update using online gradient descent."""
        lr = learning_rate or self.learning_rate

        # Update running statistics
        self.n_samples += 1
        delta = features - self.running_mean
        self.running_mean += delta / self.n_samples
        self.running_var = ((self.n_samples - 1) * self.running_var + delta * (features - self.running_mean)) / self.n_samples

        # Forward pass
        normalized = (features - self.running_mean) / (np.sqrt(self.running_var) + 1e-8)
        logit = np.dot(normalized, self.weights) + self.bias
        prediction = 1 / (1 + np.exp(-np.clip(logit, -500, 500)))

        # Compute gradient (binary cross-entropy)
        error = prediction - target
        grad_w = error * normalized
        grad_b = error

        # Momentum update
        self.velocity_w = self.momentum * self.velocity_w - lr * grad_w
        self.velocity_b = self.momentum * self.velocity_b - lr * grad_b

        self.weights += self.velocity_w
        self.bias += self.velocity_b

        return abs(error)


class ContinuousLearningEngine:
    """
    Continuous Learning Engine for real-time adaptive learning.

    Implements online learning from market data streams with:
    - Experience replay for stable learning
    - Adaptive learning rates based on market conditions
    - Immediate learning from critical errors
    - Performance-based model selection
    """

    def __init__(
        self,
        input_dim: int = 50,
        adaptation_rate: float = 1e-5,
        buffer_size: int = 50000,
        batch_size: int = 32,
        error_threshold: float = 0.1,
        mode: LearningMode = LearningMode.ACTIVE,
    ):
        self.input_dim = input_dim
        self.base_adaptation_rate = adaptation_rate
        self.adaptation_rate = adaptation_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.error_threshold = error_threshold
        self.mode = mode

        # Core learning model
        self.model = OnlineLearningModel(input_dim, adaptation_rate)

        # Experience replay buffer
        self.experience_buffer: deque = deque(maxlen=buffer_size)

        # Performance tracking
        self.performance_tracker = PerformanceTracker()

        # Learning state
        self.total_experiences = 0
        self.learning_steps = 0
        self.last_adaptation_time = datetime.now()
        self.adaptation_interval = timedelta(seconds=60)

        # Market regime detection
        self.volatility_window: deque = deque(maxlen=100)
        self.trend_window: deque = deque(maxlen=50)

        # Callbacks
        self.on_learn_callback: Optional[Callable] = None
        self.on_adapt_callback: Optional[Callable] = None

        logger.info(f"ContinuousLearningEngine initialized with mode={mode.value}")

    async def continuous_adaptation(self, market_stream: AsyncIterator[dict]):
        """
        Continuously adapt to market data streams.

        Args:
            market_stream: Async iterator yielding market data dictionaries
        """
        logger.info("Starting continuous adaptation loop")

        async for market_data in market_stream:
            try:
                # Extract features and make prediction
                features = self._extract_features(market_data)
                prediction, confidence = self.model.predict(features)

                # Store prediction for later validation
                prediction_record = {
                    "timestamp": datetime.now(),
                    "prediction": prediction,
                    "confidence": confidence,
                    "features": features,
                    "market_data": market_data,
                }

                # Wait for actual outcome (non-blocking)
                actual_result = await self._wait_for_outcome(
                    prediction_record["timestamp"],
                    market_data.get("instrument", "unknown")
                )

                if actual_result is not None:
                    # Learn from experience
                    self.learn_from_experience(
                        features,
                        prediction,
                        actual_result,
                        confidence
                    )

                    # Check if we should perform batch adaptation
                    if self._should_adapt():
                        await self._adapt_model()

            except Exception as e:
                logger.error(f"Error in continuous adaptation: {e}")
                continue

    def learn_from_experience(
        self,
        features: np.ndarray,
        prediction: float,
        actual: float,
        confidence: float
    ):
        """Learn from a single trading experience."""
        error = abs(prediction - actual)

        experience = Experience(
            features=features,
            prediction=prediction,
            actual=actual,
            confidence=confidence,
            error=error,
            timestamp=datetime.now(),
        )

        self.experience_buffer.append(experience)
        self.total_experiences += 1

        # Update performance tracker
        self.performance_tracker.update(
            prediction=prediction,
            actual=actual,
            return_pct=(1 if prediction > 0.5 and actual > 0.5 else -1) * 0.01
        )

        # Immediate learning for critical errors
        if error > self.error_threshold:
            self._immediate_learning_update(experience)

        # Trigger callback if set
        if self.on_learn_callback:
            self.on_learn_callback(experience)

        logger.debug(f"Learned from experience: error={error:.4f}, confidence={confidence:.4f}")

    def _immediate_learning_update(self, experience: Experience):
        """Perform immediate learning update for critical errors."""
        # Use higher learning rate for critical errors
        critical_lr = self.adaptation_rate * 10

        # Update model immediately
        self.model.update(
            experience.features,
            experience.actual,
            learning_rate=critical_lr
        )

        self.learning_steps += 1
        logger.info(f"Immediate learning update: error={experience.error:.4f}")

    def _should_adapt(self) -> bool:
        """Check if model should perform batch adaptation."""
        # Time-based check
        time_elapsed = datetime.now() - self.last_adaptation_time
        if time_elapsed < self.adaptation_interval:
            return False

        # Buffer size check
        if len(self.experience_buffer) < self.batch_size * 2:
            return False

        return True

    async def _adapt_model(self):
        """Perform batch adaptation from experience buffer."""
        if len(self.experience_buffer) < self.batch_size:
            return

        logger.info("Starting batch adaptation")

        # Sample random batch from buffer
        indices = np.random.choice(
            len(self.experience_buffer),
            min(self.batch_size * 4, len(self.experience_buffer)),
            replace=False
        )

        batch_loss = 0.0
        for idx in indices:
            experience = self.experience_buffer[idx]

            # Prioritize recent and high-error experiences
            recency_weight = 1.0 / (1.0 + (datetime.now() - experience.timestamp).total_seconds() / 3600)
            error_weight = 1.0 + experience.error

            effective_lr = self.adaptation_rate * recency_weight * error_weight

            # Update model
            loss = self.model.update(
                experience.features,
                experience.actual,
                learning_rate=effective_lr
            )
            batch_loss += loss

        self.learning_steps += len(indices)
        self.last_adaptation_time = datetime.now()

        # Adapt learning rate based on performance
        self._adapt_learning_rate()

        if self.on_adapt_callback:
            await self.on_adapt_callback(batch_loss / len(indices))

        logger.info(f"Batch adaptation complete: avg_loss={batch_loss/len(indices):.4f}")

    def _adapt_learning_rate(self):
        """Dynamically adapt learning rate based on performance."""
        metrics = self.performance_tracker.get_metrics()

        # Increase learning rate if performance is poor
        if metrics.accuracy < 0.5:
            self.adaptation_rate = min(self.adaptation_rate * 1.5, self.base_adaptation_rate * 10)
        # Decrease learning rate if performance is good (fine-tuning)
        elif metrics.accuracy > 0.7:
            self.adaptation_rate = max(self.adaptation_rate * 0.9, self.base_adaptation_rate * 0.1)
        else:
            # Gradually return to base rate
            self.adaptation_rate = 0.9 * self.adaptation_rate + 0.1 * self.base_adaptation_rate

    def _extract_features(self, market_data: dict) -> np.ndarray:
        """Extract feature vector from market data."""
        features = []

        # Price features
        if "close" in market_data:
            features.append(market_data["close"])
        if "open" in market_data:
            features.append(market_data["open"])
        if "high" in market_data:
            features.append(market_data["high"])
        if "low" in market_data:
            features.append(market_data["low"])

        # Volume features
        if "volume" in market_data:
            features.append(market_data["volume"])

        # Technical indicators
        for indicator in ["sma_20", "sma_50", "ema_12", "ema_26", "rsi", "macd", "atr"]:
            if indicator in market_data:
                features.append(market_data[indicator])

        # Pad or truncate to input_dim
        features = np.array(features, dtype=np.float64)
        if len(features) < self.input_dim:
            features = np.pad(features, (0, self.input_dim - len(features)))
        elif len(features) > self.input_dim:
            features = features[:self.input_dim]

        return features

    async def _wait_for_outcome(
        self,
        prediction_timestamp: datetime,
        instrument: str,
        timeout_seconds: int = 60
    ) -> Optional[float]:
        """Wait for actual market outcome."""
        # In real implementation, this would wait for actual price movement
        # For now, simulate with a small delay
        await asyncio.sleep(0.1)

        # Return None to indicate we need actual data
        # This will be integrated with the market data system
        return None

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        """Make prediction with the current model."""
        return self.model.predict(features)

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.performance_tracker.get_metrics()

    def get_state(self) -> dict:
        """Get current engine state for persistence."""
        return {
            "total_experiences": self.total_experiences,
            "learning_steps": self.learning_steps,
            "adaptation_rate": self.adaptation_rate,
            "buffer_size": len(self.experience_buffer),
            "performance": self.performance_tracker.get_metrics().to_dict(),
            "model_weights": self.model.weights.tolist(),
            "model_bias": self.model.bias,
        }

    def load_state(self, state: dict):
        """Load engine state from persistence."""
        self.total_experiences = state.get("total_experiences", 0)
        self.learning_steps = state.get("learning_steps", 0)
        self.adaptation_rate = state.get("adaptation_rate", self.base_adaptation_rate)

        if "model_weights" in state:
            self.model.weights = np.array(state["model_weights"])
        if "model_bias" in state:
            self.model.bias = state["model_bias"]

        logger.info(f"Loaded state: {self.total_experiences} experiences, {self.learning_steps} learning steps")


class AdaptiveLearningScheduler:
    """Schedules learning rate adaptation based on market conditions."""

    def __init__(self, base_lr: float = 1e-4):
        self.base_lr = base_lr
        self.current_lr = base_lr
        self.volatility_history: deque = deque(maxlen=100)
        self.performance_history: deque = deque(maxlen=100)

    def update_volatility(self, volatility: float):
        """Update volatility estimate."""
        self.volatility_history.append(volatility)

    def update_performance(self, performance: float):
        """Update performance estimate."""
        self.performance_history.append(performance)

    def get_learning_rate(self) -> float:
        """Get adapted learning rate."""
        if len(self.volatility_history) < 10:
            return self.base_lr

        avg_volatility = np.mean(self.volatility_history)

        # Lower learning rate in high volatility
        volatility_factor = 1.0 / (1.0 + avg_volatility * 10)

        # Increase learning rate if performance is poor
        if len(self.performance_history) >= 10:
            avg_performance = np.mean(self.performance_history)
            performance_factor = 2.0 - avg_performance if avg_performance < 0.5 else 1.0
        else:
            performance_factor = 1.0

        self.current_lr = self.base_lr * volatility_factor * performance_factor
        return self.current_lr
