"""
Advanced Prediction Engine for HRM Trading Bot

Implements multi-horizon prediction with:
- Ensemble Models
- Uncertainty Quantification
- Prediction Calibration
- Multi-timeframe Analysis
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A single prediction with metadata."""
    value: float
    confidence: float
    uncertainty: float
    timestamp: datetime
    horizon: str
    model_name: str
    features_used: list[str] = field(default_factory=list)


@dataclass
class EnsemblePrediction:
    """Ensemble prediction combining multiple models."""
    prediction: float
    uncertainty: float
    confidence: float
    individual_predictions: list[float]
    model_weights: list[float]
    model_names: list[str]
    agreement_score: float  # How much models agree


@dataclass
class MultiHorizonPrediction:
    """Predictions across multiple time horizons."""
    predictions: dict[str, Prediction]
    dominant_direction: str  # "bullish", "bearish", "neutral"
    consistency_score: float  # Agreement across horizons
    recommended_horizon: str


class BasePredictor:
    """Base class for prediction models."""

    def __init__(self, name: str, input_dim: int):
        self.name = name
        self.input_dim = input_dim
        self.prediction_history: deque = deque(maxlen=1000)
        self.accuracy_history: deque = deque(maxlen=100)

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        """
        Make prediction.

        Returns:
            Tuple of (prediction, confidence)
        """
        raise NotImplementedError

    def update_accuracy(self, predicted: float, actual: float):
        """Update accuracy tracking."""
        correct = (predicted > 0.5) == (actual > 0.5)
        self.accuracy_history.append(float(correct))

    def get_recent_accuracy(self) -> float:
        """Get recent accuracy."""
        if not self.accuracy_history:
            return 0.5
        return np.mean(self.accuracy_history)


class LinearPredictor(BasePredictor):
    """Simple linear regression predictor."""

    def __init__(self, name: str, input_dim: int, learning_rate: float = 0.01):
        super().__init__(name, input_dim)
        self.weights = np.random.randn(input_dim) * 0.01
        self.bias = 0.0
        self.learning_rate = learning_rate

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        """Make prediction."""
        logit = np.dot(features, self.weights) + self.bias
        prediction = 1 / (1 + np.exp(-np.clip(logit, -500, 500)))

        # Confidence based on distance from decision boundary
        confidence = abs(prediction - 0.5) * 2

        return float(prediction), float(confidence)

    def update(self, features: np.ndarray, target: float):
        """Update model with new observation."""
        prediction, _ = self.predict(features)
        error = prediction - target

        self.weights -= self.learning_rate * error * features
        self.bias -= self.learning_rate * error


class MomentumPredictor(BasePredictor):
    """Momentum-based predictor using price momentum."""

    def __init__(self, name: str, input_dim: int, lookback: int = 20):
        super().__init__(name, input_dim)
        self.lookback = lookback
        self.price_history: deque = deque(maxlen=lookback)

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        """Predict based on momentum."""
        # Assume first feature is price or return
        if len(features) > 0:
            self.price_history.append(features[0])

        if len(self.price_history) < 5:
            return 0.5, 0.3

        prices = np.array(self.price_history)

        # Calculate momentum
        short_ma = np.mean(prices[-5:])
        long_ma = np.mean(prices)

        momentum = (short_ma - long_ma) / (long_ma + 1e-10)

        # Convert to probability
        prediction = 1 / (1 + np.exp(-momentum * 10))

        # Confidence based on trend strength
        trend_strength = abs(momentum)
        confidence = min(trend_strength * 5, 0.9)

        return float(prediction), float(confidence)


class MeanReversionPredictor(BasePredictor):
    """Mean reversion predictor."""

    def __init__(self, name: str, input_dim: int, lookback: int = 50):
        super().__init__(name, input_dim)
        self.lookback = lookback
        self.value_history: deque = deque(maxlen=lookback)

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        """Predict mean reversion."""
        if len(features) > 0:
            self.value_history.append(features[0])

        if len(self.value_history) < 20:
            return 0.5, 0.3

        values = np.array(self.value_history)
        mean = np.mean(values)
        std = np.std(values)
        current = values[-1]

        if std == 0:
            return 0.5, 0.3

        # Z-score
        z_score = (current - mean) / std

        # Mean reversion: predict opposite of deviation
        # High positive z-score -> predict down (< 0.5)
        prediction = 1 / (1 + np.exp(z_score))

        # Confidence based on z-score magnitude
        confidence = min(abs(z_score) / 3, 0.9)

        return float(prediction), float(confidence)


class UncertaintyQuantifier:
    """
    Quantifies prediction uncertainty.
    """

    def __init__(self):
        self.calibration_history: deque = deque(maxlen=1000)

    def estimate_uncertainty(
        self,
        predictions: list[float],
        confidences: list[float]
    ) -> float:
        """
        Estimate uncertainty from ensemble predictions.

        Args:
            predictions: List of predictions from different models
            confidences: List of model confidences

        Returns:
            Uncertainty estimate (0 = certain, 1 = uncertain)
        """
        if not predictions:
            return 1.0

        # Variance-based uncertainty
        pred_variance = np.var(predictions)

        # Disagreement-based uncertainty
        mean_pred = np.mean(predictions)
        disagreement = np.mean([abs(p - mean_pred) for p in predictions])

        # Confidence-weighted uncertainty
        avg_confidence = np.mean(confidences) if confidences else 0.5
        confidence_uncertainty = 1 - avg_confidence

        # Combine uncertainty measures
        uncertainty = (
            0.4 * pred_variance +
            0.3 * disagreement +
            0.3 * confidence_uncertainty
        )

        return float(min(uncertainty, 1.0))

    def calibrate(
        self,
        predicted_prob: float,
        actual: float
    ):
        """Update calibration with new observation."""
        self.calibration_history.append({
            "predicted": predicted_prob,
            "actual": actual,
        })

    def get_calibration_error(self) -> float:
        """Calculate Expected Calibration Error."""
        if len(self.calibration_history) < 100:
            return 0.0

        history = list(self.calibration_history)
        predictions = [h["predicted"] for h in history]
        actuals = [h["actual"] for h in history]

        # Bin predictions
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean([actuals[j] for j in range(len(actuals)) if bin_mask[j]])
                bin_confidence = np.mean([predictions[j] for j in range(len(predictions)) if bin_mask[j]])
                bin_weight = np.sum(bin_mask) / len(predictions)
                ece += bin_weight * abs(bin_accuracy - bin_confidence)

        return float(ece)


class PredictionCalibrator:
    """
    Calibrates raw predictions for better probability estimates.
    """

    def __init__(self):
        self.calibration_params: dict[str, dict] = {}
        self.observation_history: dict[str, deque] = {}

    def add_observation(
        self,
        horizon: str,
        raw_prediction: float,
        actual: float
    ):
        """Add observation for calibration."""
        if horizon not in self.observation_history:
            self.observation_history[horizon] = deque(maxlen=1000)

        self.observation_history[horizon].append({
            "raw": raw_prediction,
            "actual": actual,
        })

        # Recalibrate periodically
        if len(self.observation_history[horizon]) % 100 == 0:
            self._fit_calibration(horizon)

    def _fit_calibration(self, horizon: str):
        """Fit Platt scaling calibration."""
        history = list(self.observation_history[horizon])
        if len(history) < 50:
            return

        raw_preds = np.array([h["raw"] for h in history])
        actuals = np.array([h["actual"] for h in history])

        # Fit logistic regression for calibration
        # p_calibrated = 1 / (1 + exp(-(a * raw_pred + b)))

        # Simple gradient descent
        a, b = 1.0, 0.0
        lr = 0.01

        for _ in range(100):
            logits = a * raw_preds + b
            preds = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

            error = preds - actuals
            grad_a = np.mean(error * raw_preds)
            grad_b = np.mean(error)

            a -= lr * grad_a
            b -= lr * grad_b

        self.calibration_params[horizon] = {"a": a, "b": b}

    def calibrate(self, raw_prediction: float, horizon: str) -> float:
        """Calibrate a raw prediction."""
        if horizon not in self.calibration_params:
            return raw_prediction

        params = self.calibration_params[horizon]
        logit = params["a"] * raw_prediction + params["b"]
        calibrated = 1 / (1 + np.exp(-np.clip(logit, -500, 500)))

        return float(calibrated)


class AdvancedPredictionEngine:
    """
    Advanced prediction engine with ensemble models and multi-horizon forecasting.

    Features:
    - Multiple predictor models
    - Ensemble combination
    - Uncertainty quantification
    - Prediction calibration
    - Multi-timeframe analysis
    """

    def __init__(
        self,
        input_dim: int = 20,
        horizons: list[str] = None,
    ):
        self.input_dim = input_dim
        self.horizons = horizons or ["1min", "5min", "15min", "1h", "4h", "1d"]

        # Initialize predictors for each horizon
        self.predictors: dict[str, list[BasePredictor]] = {}
        self._initialize_predictors()

        # Components
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.prediction_calibrator = PredictionCalibrator()

        # Model weights (performance-based)
        self.model_weights: dict[str, dict[str, float]] = {}

        # Prediction history
        self.prediction_history: deque = deque(maxlen=10000)

        logger.info(
            f"AdvancedPredictionEngine initialized with "
            f"{len(self.horizons)} horizons"
        )

    def _initialize_predictors(self):
        """Initialize prediction models for each horizon."""
        for horizon in self.horizons:
            self.predictors[horizon] = [
                LinearPredictor(f"linear_{horizon}", self.input_dim),
                MomentumPredictor(f"momentum_{horizon}", self.input_dim),
                MeanReversionPredictor(f"mean_reversion_{horizon}", self.input_dim),
            ]
            self.model_weights[horizon] = {
                p.name: 1.0 / len(self.predictors[horizon])
                for p in self.predictors[horizon]
            }

    def multi_horizon_prediction(
        self,
        features: np.ndarray
    ) -> MultiHorizonPrediction:
        """
        Make predictions across multiple time horizons.

        Args:
            features: Input feature vector

        Returns:
            MultiHorizonPrediction with predictions for all horizons
        """
        predictions = {}

        for horizon in self.horizons:
            ensemble = self.ensemble_prediction(features, horizon)

            # Calibrate
            calibrated_pred = self.prediction_calibrator.calibrate(
                ensemble.prediction, horizon
            )

            predictions[horizon] = Prediction(
                value=calibrated_pred,
                confidence=ensemble.confidence,
                uncertainty=ensemble.uncertainty,
                timestamp=datetime.now(),
                horizon=horizon,
                model_name="ensemble",
            )

        # Determine dominant direction
        avg_pred = np.mean([p.value for p in predictions.values()])
        if avg_pred > 0.6:
            dominant_direction = "bullish"
        elif avg_pred < 0.4:
            dominant_direction = "bearish"
        else:
            dominant_direction = "neutral"

        # Calculate consistency across horizons
        pred_values = [p.value for p in predictions.values()]
        consistency = 1 - np.std(pred_values)

        # Recommend best horizon based on confidence and consistency
        horizon_scores = {}
        for horizon, pred in predictions.items():
            score = pred.confidence * (1 - pred.uncertainty) * consistency
            horizon_scores[horizon] = score

        recommended = max(horizon_scores, key=horizon_scores.get)

        return MultiHorizonPrediction(
            predictions=predictions,
            dominant_direction=dominant_direction,
            consistency_score=float(consistency),
            recommended_horizon=recommended,
        )

    def ensemble_prediction(
        self,
        features: np.ndarray,
        horizon: str = "1h"
    ) -> EnsemblePrediction:
        """
        Make ensemble prediction for specific horizon.

        Args:
            features: Input feature vector
            horizon: Time horizon

        Returns:
            EnsemblePrediction combining all models
        """
        if horizon not in self.predictors:
            horizon = self.horizons[0]

        individual_predictions = []
        individual_confidences = []
        model_names = []

        for predictor in self.predictors[horizon]:
            pred, conf = predictor.predict(features)
            individual_predictions.append(pred)
            individual_confidences.append(conf)
            model_names.append(predictor.name)

        # Get model weights
        weights = [
            self.model_weights[horizon].get(name, 1.0)
            for name in model_names
        ]
        weights = np.array(weights) / np.sum(weights)

        # Weighted average prediction
        ensemble_pred = np.average(individual_predictions, weights=weights)

        # Uncertainty
        uncertainty = self.uncertainty_quantifier.estimate_uncertainty(
            individual_predictions, individual_confidences
        )

        # Confidence: weighted average of confidences adjusted by agreement
        agreement = 1 - np.std(individual_predictions)
        confidence = np.average(individual_confidences, weights=weights) * agreement

        return EnsemblePrediction(
            prediction=float(ensemble_pred),
            uncertainty=float(uncertainty),
            confidence=float(confidence),
            individual_predictions=individual_predictions,
            model_weights=weights.tolist(),
            model_names=model_names,
            agreement_score=float(agreement),
        )

    def update_models(
        self,
        features: np.ndarray,
        actual: float,
        horizon: str = "1h"
    ):
        """
        Update models with actual outcome.

        Args:
            features: Feature vector used for prediction
            actual: Actual outcome
            horizon: Time horizon
        """
        if horizon not in self.predictors:
            return

        for predictor in self.predictors[horizon]:
            # Update accuracy tracking
            pred, _ = predictor.predict(features)
            predictor.update_accuracy(pred, actual)

            # Update model if it supports it
            if hasattr(predictor, "update"):
                predictor.update(features, actual)

        # Update model weights based on accuracy
        self._update_model_weights(horizon)

        # Update calibration
        ensemble = self.ensemble_prediction(features, horizon)
        self.prediction_calibrator.add_observation(horizon, ensemble.prediction, actual)

    def _update_model_weights(self, horizon: str):
        """Update model weights based on recent accuracy."""
        accuracies = []
        names = []

        for predictor in self.predictors[horizon]:
            acc = predictor.get_recent_accuracy()
            accuracies.append(acc)
            names.append(predictor.name)

        # Convert accuracies to weights (softmax-like)
        accuracies = np.array(accuracies)
        weights = np.exp(accuracies * 2)  # Temperature = 0.5
        weights = weights / np.sum(weights)

        for name, weight in zip(names, weights):
            self.model_weights[horizon][name] = float(weight)

    def add_predictor(self, predictor: BasePredictor, horizon: str):
        """Add a custom predictor for a horizon."""
        if horizon not in self.predictors:
            self.predictors[horizon] = []
            self.model_weights[horizon] = {}

        self.predictors[horizon].append(predictor)
        self.model_weights[horizon][predictor.name] = 1.0 / len(self.predictors[horizon])

    def get_model_performance(self) -> dict:
        """Get performance metrics for all models."""
        performance = {}

        for horizon in self.horizons:
            performance[horizon] = {}
            for predictor in self.predictors.get(horizon, []):
                performance[horizon][predictor.name] = {
                    "accuracy": predictor.get_recent_accuracy(),
                    "weight": self.model_weights[horizon].get(predictor.name, 0),
                }

        return performance

    def get_calibration_info(self) -> dict:
        """Get calibration information."""
        return {
            "calibration_error": self.uncertainty_quantifier.get_calibration_error(),
            "horizon_params": dict(self.prediction_calibrator.calibration_params),
        }

    def get_state(self) -> dict:
        """Get engine state."""
        return {
            "horizons": self.horizons,
            "model_weights": self.model_weights,
            "prediction_count": len(self.prediction_history),
            "calibration": self.get_calibration_info(),
        }


class PredictionValidator:
    """
    Validates predictions against actual outcomes.
    """

    def __init__(self, validation_delay: dict[str, timedelta] = None):
        self.validation_delay = validation_delay or {
            "1min": timedelta(minutes=1),
            "5min": timedelta(minutes=5),
            "15min": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1),
        }
        self.pending_validations: deque = deque(maxlen=10000)
        self.validation_results: deque = deque(maxlen=10000)

    def add_pending(
        self,
        prediction: Prediction,
        features: np.ndarray
    ):
        """Add prediction pending validation."""
        self.pending_validations.append({
            "prediction": prediction,
            "features": features,
            "validate_at": datetime.now() + self.validation_delay.get(
                prediction.horizon, timedelta(hours=1)
            ),
        })

    async def validate_pending(
        self,
        get_actual_fn: Callable
    ):
        """
        Validate pending predictions.

        Args:
            get_actual_fn: Async function to get actual outcome
        """
        current_time = datetime.now()
        to_validate = []

        for pending in self.pending_validations:
            if pending["validate_at"] <= current_time:
                to_validate.append(pending)

        for pending in to_validate:
            try:
                actual = await get_actual_fn(
                    pending["prediction"].timestamp,
                    pending["prediction"].horizon
                )

                if actual is not None:
                    self.validation_results.append({
                        "prediction": pending["prediction"].value,
                        "actual": actual,
                        "horizon": pending["prediction"].horizon,
                        "confidence": pending["prediction"].confidence,
                        "correct": (pending["prediction"].value > 0.5) == (actual > 0.5),
                        "error": abs(pending["prediction"].value - actual),
                        "timestamp": datetime.now(),
                    })

                self.pending_validations.remove(pending)

            except Exception as e:
                logger.error(f"Validation failed: {e}")

    def get_validation_metrics(self) -> dict:
        """Get validation metrics by horizon."""
        metrics = {}

        for horizon in set(r["horizon"] for r in self.validation_results):
            horizon_results = [r for r in self.validation_results if r["horizon"] == horizon]

            if horizon_results:
                metrics[horizon] = {
                    "accuracy": np.mean([r["correct"] for r in horizon_results]),
                    "mae": np.mean([r["error"] for r in horizon_results]),
                    "count": len(horizon_results),
                }

        return metrics
