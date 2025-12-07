"""
Trader Agent - Executes trading decisions

Enhanced with Advanced Prediction Engine features:
- Multi-horizon ensemble predictions
- Uncertainty quantification
- Prediction calibration
"""
from typing import Any, Dict, Optional
from datetime import datetime
import numpy as np
from ..core.base_agent import BaseAgent, AgentMessage, AgentRole, TradingSignal
from ..advanced.prediction.advanced_prediction import AdvancedPredictionEngine


class TraderAgent(BaseAgent):
    """
    Agent specialized in making final trading decisions and execution.
    Synthesizes inputs from all other agents.

    Enhanced with AdvancedPredictionEngine for:
    - Multi-horizon price predictions
    - Ensemble model predictions
    - Uncertainty quantification
    - Prediction calibration
    """

    def __init__(self, llm_client: Any = None, use_advanced: bool = True):
        super().__init__(
            name="Trader",
            role=AgentRole.TRADER,
            description="Makes final trading decisions and manages execution",
            llm_client=llm_client
        )
        self.pending_orders = []
        self.executed_orders = []
        self.use_advanced = use_advanced

        # Initialize Advanced Prediction Engine
        if use_advanced:
            self.prediction_engine = AdvancedPredictionEngine(
                input_dim=20,
                horizons=["1min", "5min", "15min", "1h", "4h", "1d"]
            )
        else:
            self.prediction_engine = None

        # Learning state
        self.prediction_history = []

    def get_system_prompt(self) -> str:
        return """You are an expert Trader responsible for final trading decisions.
Your role is to synthesize analysis from multiple sources and decide on trade execution.

You receive input from:
- Technical Analyst: Chart patterns, indicators, price levels
- Sentiment Analyst: Market mood, news impact
- Risk Manager: Position sizing, risk limits

Your responsibilities:
1. Evaluate all agent inputs
2. Make final BUY/SELL/HOLD decision
3. Determine precise entry, stop loss, take profit
4. Time the execution
5. Document reasoning

Decision framework:
- Require minimum 60% confidence for entry
- All risk checks must pass
- Technical and sentiment should align
- Use limit orders when possible

Be decisive but disciplined. Document every decision clearly."""

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make final trading decision based on all inputs.
        Uses AdvancedPredictionEngine when available for enhanced analysis.
        """
        analyses = data.get("analyses", {})
        market_data = data.get("market_data", {})
        risk_assessment = data.get("risk_assessment", {})

        instrument = market_data.get("instrument", "DE40")
        current_price = market_data.get("current_price", 0)

        # Aggregate signals from analysts
        signals = self._aggregate_signals(analyses)

        # Check risk approval
        risk_approved = risk_assessment.get("approval", {}).get("approved", False)

        # Get advanced predictions if available
        advanced_predictions = None
        if self.use_advanced and self.prediction_engine:
            features = self._extract_features(market_data)
            advanced_predictions = self._get_advanced_predictions(features)
            # Enhance signals with prediction engine output
            signals = self._enhance_signals_with_predictions(signals, advanced_predictions)

        # Make final decision
        decision = self._make_decision(signals, risk_approved, risk_assessment)

        # Generate order if decision is to trade
        order = None
        if decision["action"] in ["BUY", "SELL"]:
            order = self._generate_order(
                instrument=instrument,
                direction=decision["action"],
                current_price=current_price,
                risk_data=risk_assessment.get("position_sizing", {})
            )

        # LLM reasoning if available
        reasoning = ""
        if self.llm_client:
            reasoning = await self._get_llm_decision(analyses, decision)

        # Store prediction for learning
        if advanced_predictions:
            self.prediction_history.append({
                "timestamp": datetime.utcnow(),
                "instrument": instrument,
                "predictions": advanced_predictions,
                "decision": decision["action"],
                "current_price": current_price,
            })

        result = {
            "agent": self.name,
            "instrument": instrument,
            "decision": decision,
            "order": order,
            "signals_summary": signals,
            "reasoning": reasoning,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Add advanced prediction info
        if advanced_predictions:
            result["advanced_predictions"] = advanced_predictions
            result["recommended_horizon"] = advanced_predictions.get("recommended_horizon", "1h")

        return result

    def _extract_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extract features for prediction engine."""
        features = []

        # Price features
        indicators = market_data.get("indicators", {})
        ohlcv = market_data.get("ohlcv", {})

        # Current price
        features.append(market_data.get("current_price", 0))

        # Moving averages
        for ma in ["sma_20", "sma_50", "ema_12", "ema_26"]:
            features.append(indicators.get(ma, 0))

        # Momentum indicators
        features.append(indicators.get("rsi", 50))
        features.append(indicators.get("macd", 0))
        features.append(indicators.get("macd_signal", 0))

        # Volatility
        features.append(indicators.get("atr", 0))
        features.append(indicators.get("bb_upper", 0))
        features.append(indicators.get("bb_lower", 0))

        # Volume
        if "volume" in ohlcv and len(ohlcv["volume"]) > 0:
            features.append(ohlcv["volume"][-1])
        else:
            features.append(0)

        # Pad to expected input dim
        while len(features) < 20:
            features.append(0)

        return np.array(features[:20], dtype=np.float64)

    def _get_advanced_predictions(self, features: np.ndarray) -> Dict[str, Any]:
        """Get multi-horizon predictions from prediction engine."""
        if not self.prediction_engine:
            return {}

        try:
            # Get multi-horizon prediction
            multi_pred = self.prediction_engine.multi_horizon_prediction(features)

            predictions = {}
            for horizon, pred in multi_pred.predictions.items():
                predictions[horizon] = {
                    "value": pred.value,
                    "confidence": pred.confidence,
                    "uncertainty": pred.uncertainty,
                }

            return {
                "horizons": predictions,
                "dominant_direction": multi_pred.dominant_direction,
                "consistency_score": multi_pred.consistency_score,
                "recommended_horizon": multi_pred.recommended_horizon,
            }
        except Exception as e:
            return {"error": str(e)}

    def _enhance_signals_with_predictions(
        self,
        signals: Dict[str, Any],
        predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance analyst signals with prediction engine output."""
        if not predictions or "error" in predictions:
            return signals

        # Get recommended horizon prediction
        horizons = predictions.get("horizons", {})
        recommended = predictions.get("recommended_horizon", "1h")

        if recommended in horizons:
            pred = horizons[recommended]
            pred_value = pred["value"]
            pred_confidence = pred["confidence"]

            # Adjust signals based on prediction
            if pred_value > 0.6 and pred_confidence > 0.5:
                signals["buy_score"] += pred_confidence * 0.5
                signals["prediction_boost"] = "bullish"
            elif pred_value < 0.4 and pred_confidence > 0.5:
                signals["sell_score"] += pred_confidence * 0.5
                signals["prediction_boost"] = "bearish"
            else:
                signals["prediction_boost"] = "neutral"

            # Adjust confidence based on consistency
            consistency = predictions.get("consistency_score", 0.5)
            signals["avg_confidence"] = (signals["avg_confidence"] + consistency) / 2
            signals["prediction_confidence"] = pred_confidence
            signals["prediction_uncertainty"] = pred.get("uncertainty", 0.5)

        return signals

    def learn_from_outcome(self, prediction_record: Dict[str, Any], actual_outcome: float):
        """Learn from trade outcome to improve predictions."""
        if not self.prediction_engine:
            return

        features = prediction_record.get("features")
        if features is not None:
            # Update prediction engine
            self.prediction_engine.update_models(
                features=features,
                actual=actual_outcome,
                horizon=prediction_record.get("horizon", "1h")
            )

    def _aggregate_signals(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate signals from all analysts."""
        buy_votes = 0
        sell_votes = 0
        hold_votes = 0
        total_confidence = 0
        signal_count = 0

        for agent_name, analysis in analyses.items():
            if not isinstance(analysis, dict):
                continue

            signal = analysis.get("signal", {})
            if not signal:
                continue

            direction = signal.get("direction", "HOLD")
            confidence = signal.get("confidence", 0.5)

            if direction == "BUY":
                buy_votes += confidence
            elif direction == "SELL":
                sell_votes += confidence
            else:
                hold_votes += confidence

            total_confidence += confidence
            signal_count += 1

        avg_confidence = total_confidence / signal_count if signal_count > 0 else 0

        return {
            "buy_score": buy_votes,
            "sell_score": sell_votes,
            "hold_score": hold_votes,
            "avg_confidence": avg_confidence,
            "signal_count": signal_count
        }

    def _make_decision(
        self,
        signals: Dict[str, Any],
        risk_approved: bool,
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make final trading decision."""
        buy_score = signals.get("buy_score", 0)
        sell_score = signals.get("sell_score", 0)
        avg_confidence = signals.get("avg_confidence", 0)

        # Decision logic
        if not risk_approved:
            return {
                "action": "HOLD",
                "reason": "Risk check failed",
                "confidence": 0,
                "risk_details": risk_assessment.get("approval", {})
            }

        min_confidence = 0.6
        min_score_diff = 0.3

        if buy_score > sell_score + min_score_diff and avg_confidence >= min_confidence:
            action = "BUY"
            confidence = min(0.95, (buy_score / (buy_score + sell_score + 0.001)) * avg_confidence)
        elif sell_score > buy_score + min_score_diff and avg_confidence >= min_confidence:
            action = "SELL"
            confidence = min(0.95, (sell_score / (buy_score + sell_score + 0.001)) * avg_confidence)
        else:
            action = "HOLD"
            confidence = avg_confidence

        reason = self._generate_reason(action, signals, risk_assessment)

        return {
            "action": action,
            "confidence": round(confidence, 2),
            "reason": reason,
            "signals": signals
        }

    def _generate_reason(
        self,
        action: str,
        signals: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> str:
        """Generate human-readable reason for decision."""
        if action == "HOLD":
            if signals["avg_confidence"] < 0.6:
                return "Insufficient confidence - signals not aligned"
            return "No clear directional bias"

        direction = "bullish" if action == "BUY" else "bearish"
        confidence = signals["avg_confidence"]
        risk_level = risk_assessment.get("risk_assessment", {}).get("risk_level", "unknown")

        return f"Strong {direction} signal (conf: {confidence:.0%}), risk: {risk_level}"

    def _generate_order(
        self,
        instrument: str,
        direction: str,
        current_price: float,
        risk_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate order details."""
        return {
            "instrument": instrument,
            "direction": direction,
            "order_type": "MARKET",
            "entry_price": current_price,
            "position_size": risk_data.get("position_size", 0.01),
            "stop_loss": risk_data.get("stop_loss", 0),
            "take_profit": risk_data.get("take_profit", 0),
            "risk_amount": risk_data.get("risk_amount", 0),
            "status": "PENDING",
            "created_at": datetime.utcnow().isoformat()
        }

    async def _get_llm_decision(
        self,
        analyses: Dict[str, Any],
        decision: Dict[str, Any]
    ) -> str:
        """Get LLM reasoning for the decision."""
        if not self.llm_client:
            return ""

        prompt = f"""Based on the following analysis and decision, provide brief reasoning:

Analysis Summary:
{analyses}

Decision: {decision['action']}
Confidence: {decision['confidence']}

Explain in 2-3 sentences why this is the correct decision."""

        try:
            response = await self.llm_client.analyze(
                system_prompt=self.get_system_prompt(),
                user_prompt=prompt
            )
            return response
        except Exception:
            return ""

    def add_pending_order(self, order: Dict[str, Any]) -> None:
        """Add order to pending list."""
        self.pending_orders.append(order)

    def execute_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Mark order as executed."""
        for i, order in enumerate(self.pending_orders):
            if order.get("id") == order_id:
                executed = self.pending_orders.pop(i)
                executed["status"] = "EXECUTED"
                executed["executed_at"] = datetime.utcnow().isoformat()
                self.executed_orders.append(executed)
                return executed
        return None

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message."""
        if message.message_type == "trade_request":
            decision = await self.analyze(message.content)
            return await self.send_message(
                receiver=message.sender,
                content={"decision": decision},
                message_type="trade_decision"
            )
        return None
