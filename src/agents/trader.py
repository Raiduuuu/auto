"""
Trader Agent - Executes trading decisions
"""
from typing import Any, Dict, Optional
from datetime import datetime, timezone
from ..core.base_agent import BaseAgent, AgentMessage, AgentRole, TradingSignal


class TraderAgent(BaseAgent):
    """
    Agent specialized in making final trading decisions and execution.
    Synthesizes inputs from all other agents.
    """

    def __init__(self, llm_client: Any = None):
        super().__init__(
            name="Trader",
            role=AgentRole.TRADER,
            description="Makes final trading decisions and manages execution",
            llm_client=llm_client
        )
        self.pending_orders = []
        self.executed_orders = []

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

        return {
            "agent": self.name,
            "instrument": instrument,
            "decision": decision,
            "order": order,
            "signals_summary": signals,
            "reasoning": reasoning,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

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
            "created_at": datetime.now(timezone.utc).isoformat()
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
                executed["executed_at"] = datetime.now(timezone.utc).isoformat()
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
