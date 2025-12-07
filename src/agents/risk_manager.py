"""
Risk Manager Agent - Manages risk and position sizing

Enhanced with Advanced Risk Management features:
- Kelly Criterion position sizing
- Portfolio VaR calculations
- Circuit breakers
- Emergency protocols
"""
from typing import Any, Dict, Optional
from ..core.base_agent import BaseAgent, AgentMessage, AgentRole
from ..advanced.analysis.advanced_risk import (
    AdvancedRiskManager,
    MarketConditions,
    Position,
)


class RiskManagerAgent(BaseAgent):
    """
    Agent specialized in risk management and position sizing.
    Ensures trades comply with risk parameters.

    Enhanced with AdvancedRiskManager for:
    - Dynamic Kelly Criterion position sizing
    - Portfolio VaR calculations
    - Circuit breakers and emergency protocols
    - Correlation and liquidity risk analysis
    """

    def __init__(
        self,
        llm_client: Any = None,
        max_risk_per_trade: float = 1.0,
        max_daily_loss: float = 5.0,
        max_drawdown: float = 10.0,
        use_advanced: bool = True
    ):
        super().__init__(
            name="RiskManager",
            role=AgentRole.RISK_MANAGER,
            description="Manages risk, position sizing, and trading limits",
            llm_client=llm_client
        )
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.daily_pnl = 0.0
        self.open_positions = []
        self.use_advanced = use_advanced

        # Initialize Advanced Risk Manager
        if use_advanced:
            self.advanced_risk = AdvancedRiskManager(
                max_portfolio_risk=max_risk_per_trade / 100,
                max_single_position=max_risk_per_trade / 200,
                max_daily_loss=max_daily_loss / 100,
                max_drawdown=max_drawdown / 100,
            )
        else:
            self.advanced_risk = None

    def get_system_prompt(self) -> str:
        return """You are an expert Risk Manager for a trading operation.
Your role is to protect capital and ensure sustainable trading.

You specialize in:
- Position sizing calculations
- Stop loss placement
- Risk/reward ratio analysis
- Portfolio risk assessment
- Drawdown management
- Correlation risk

Your rules:
- Maximum risk per trade: {max_risk}% of capital
- Maximum daily loss: {max_daily}% of capital
- Maximum drawdown: {max_dd}% of capital

When assessing, provide:
1. Position size recommendation
2. Stop loss level
3. Risk/reward ratio
4. Overall risk assessment
5. Approval/rejection with reasoning

Be conservative and always prioritize capital preservation.""".format(
            max_risk=self.max_risk_per_trade,
            max_daily=self.max_daily_loss,
            max_dd=self.max_drawdown
        )

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform risk analysis on proposed trade.
        Uses AdvancedRiskManager when available for enhanced analysis.
        """
        market_data = data.get("market_data", {})
        analyses = data.get("analyses", {})
        account = data.get("account", {})

        # Extract relevant data
        balance = account.get("balance", 10000)
        current_price = market_data.get("current_price", 0)
        atr = market_data.get("atr", 0)
        instrument = market_data.get("instrument", "DE40")

        # Use advanced risk manager if available
        if self.use_advanced and self.advanced_risk:
            return await self._analyze_advanced(
                market_data, analyses, account, instrument, balance, current_price, atr
            )

        # Fallback to basic analysis
        # Calculate position size
        position_sizing = self._calculate_position_size(
            balance=balance,
            entry_price=current_price,
            stop_loss_distance=atr * 2,  # 2 ATR stop
            risk_percent=self.max_risk_per_trade
        )

        # Assess overall risk
        risk_assessment = self._assess_risk(
            balance=balance,
            position_size=position_sizing["position_size"],
            current_price=current_price,
            analyses=analyses
        )

        # Check trading limits
        limits_check = self._check_limits(balance)

        # Generate approval
        approval = self._generate_approval(risk_assessment, limits_check)

        return {
            "agent": self.name,
            "instrument": instrument,
            "position_sizing": position_sizing,
            "risk_assessment": risk_assessment,
            "limits_check": limits_check,
            "approval": approval
        }

    async def _analyze_advanced(
        self,
        market_data: Dict[str, Any],
        analyses: Dict[str, Any],
        account: Dict[str, Any],
        instrument: str,
        balance: float,
        current_price: float,
        atr: float
    ) -> Dict[str, Any]:
        """
        Perform advanced risk analysis using AdvancedRiskManager.
        """
        # Calculate prediction confidence from analyses
        confidences = []
        for agent_name, analysis in analyses.items():
            if isinstance(analysis, dict) and "signal" in analysis:
                conf = analysis["signal"].get("confidence", 0.5)
                confidences.append(conf)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Create market conditions
        market_conditions = MarketConditions(
            volatility=atr / current_price if current_price > 0 else 0.02,
            liquidity=0.7,  # Default, would be from market data
            trend_strength=0.0,  # Would be calculated from indicators
            correlation_regime="mixed",
            spread_pct=market_data.get("spread_pct", 0.0001),
        )

        # Check for emergency conditions
        emergency_action = self.advanced_risk.emergency_protocols(market_conditions)
        if emergency_action.action == "halt_trading":
            return {
                "agent": self.name,
                "instrument": instrument,
                "position_sizing": {"position_size": 0, "risk_amount": 0},
                "risk_assessment": {"risk_level": "critical", "risk_score": 100},
                "limits_check": {"can_trade": False},
                "approval": {
                    "approved": False,
                    "reason": f"Emergency halt: {emergency_action.reason}"
                },
                "emergency_action": emergency_action.__dict__
            }

        # Determine trade direction from analyses
        direction = "long"  # Default
        buy_votes = 0
        sell_votes = 0
        for agent_name, analysis in analyses.items():
            if isinstance(analysis, dict) and "signal" in analysis:
                signal_dir = analysis["signal"].get("direction", "HOLD")
                if signal_dir == "BUY":
                    buy_votes += 1
                elif signal_dir == "SELL":
                    sell_votes += 1
        if sell_votes > buy_votes:
            direction = "short"

        # Get advanced risk assessment
        risk_assessment = self.advanced_risk.assess_trade(
            instrument=instrument,
            direction=direction,
            entry_price=current_price,
            prediction_confidence=avg_confidence,
            market_conditions=market_conditions,
            account_balance=balance,
            atr=atr if atr > 0 else current_price * 0.01,
        )

        # Convert to expected format
        position_sizing = {
            "position_size": round(risk_assessment.position_size / current_price, 4) if current_price > 0 else 0,
            "risk_amount": round(risk_assessment.total_risk * balance, 2),
            "risk_percent": risk_assessment.total_risk * 100,
            "stop_loss": round(risk_assessment.stop_loss, 2),
            "take_profit": round(risk_assessment.take_profit, 2),
            "stop_distance": abs(current_price - risk_assessment.stop_loss),
            "risk_reward_ratio": 1.5
        }

        risk_info = {
            "risk_score": int(risk_assessment.total_risk * 100),
            "risk_level": "high" if risk_assessment.total_risk > 0.015 else "medium" if risk_assessment.total_risk > 0.008 else "low",
            "exposure_percent": round(risk_assessment.position_size / balance * 100, 2) if balance > 0 else 0,
            "avg_confidence": round(avg_confidence, 2),
            "open_positions": len(self.open_positions),
            "var_contribution": risk_assessment.var_contribution,
            "correlation_risk": risk_assessment.correlation_risk,
            "liquidity_risk": risk_assessment.liquidity_risk,
        }

        # Get portfolio risk metrics
        portfolio_metrics = self.advanced_risk.assess_portfolio_risk()

        limits_check = self._check_limits(balance)
        limits_check["portfolio_var_95"] = portfolio_metrics.portfolio_var_95
        limits_check["current_drawdown"] = portfolio_metrics.current_drawdown
        limits_check["sharpe_ratio"] = portfolio_metrics.sharpe_ratio

        approval = {
            "approved": risk_assessment.approved,
            "reason": risk_assessment.reason,
            "warnings": risk_assessment.warnings,
            "risk_level": risk_info["risk_level"],
            "confidence": avg_confidence
        }

        return {
            "agent": self.name,
            "instrument": instrument,
            "position_sizing": position_sizing,
            "risk_assessment": risk_info,
            "limits_check": limits_check,
            "approval": approval,
            "portfolio_metrics": portfolio_metrics.to_dict(),
            "market_conditions": {
                "volatility": market_conditions.volatility,
                "liquidity": market_conditions.liquidity,
            }
        }

    def _calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        stop_loss_distance: float,
        risk_percent: float
    ) -> Dict[str, Any]:
        """Calculate optimal position size based on risk parameters."""
        if stop_loss_distance <= 0 or entry_price <= 0:
            return {
                "position_size": 0,
                "risk_amount": 0,
                "stop_loss": 0,
                "error": "Invalid price or stop loss"
            }

        # Risk amount in currency
        risk_amount = balance * (risk_percent / 100)

        # Position size calculation
        position_size = risk_amount / stop_loss_distance

        # Stop loss price
        stop_loss = entry_price - stop_loss_distance

        # Take profit (1.5:1 reward/risk)
        take_profit = entry_price + (stop_loss_distance * 1.5)

        return {
            "position_size": round(position_size, 4),
            "risk_amount": round(risk_amount, 2),
            "risk_percent": risk_percent,
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "stop_distance": round(stop_loss_distance, 2),
            "risk_reward_ratio": 1.5
        }

    def _assess_risk(
        self,
        balance: float,
        position_size: float,
        current_price: float,
        analyses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess overall risk of the trade."""
        # Calculate exposure
        exposure = position_size * current_price
        exposure_percent = (exposure / balance) * 100 if balance > 0 else 0

        # Analyze confidence from other agents
        confidences = []
        for agent_name, analysis in analyses.items():
            if isinstance(analysis, dict) and "signal" in analysis:
                conf = analysis["signal"].get("confidence", 0.5)
                confidences.append(conf)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Risk score (0-100, higher = more risky)
        risk_score = 50

        # Adjust for exposure
        if exposure_percent > 50:
            risk_score += 20
        elif exposure_percent > 30:
            risk_score += 10

        # Adjust for confidence
        if avg_confidence < 0.5:
            risk_score += 15
        elif avg_confidence > 0.7:
            risk_score -= 10

        # Adjust for daily PnL
        if self.daily_pnl < -self.max_daily_loss / 2:
            risk_score += 20

        risk_level = (
            "high" if risk_score > 70 else
            "medium" if risk_score > 40 else
            "low"
        )

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "exposure_percent": round(exposure_percent, 2),
            "avg_confidence": round(avg_confidence, 2),
            "open_positions": len(self.open_positions)
        }

    def _check_limits(self, balance: float) -> Dict[str, Any]:
        """Check if trading limits are breached."""
        daily_loss_percent = abs(self.daily_pnl / balance * 100) if balance > 0 else 0

        return {
            "daily_loss_limit": {
                "current": round(daily_loss_percent, 2),
                "max": self.max_daily_loss,
                "breached": daily_loss_percent >= self.max_daily_loss
            },
            "max_positions": {
                "current": len(self.open_positions),
                "max": 5,
                "breached": len(self.open_positions) >= 5
            },
            "can_trade": daily_loss_percent < self.max_daily_loss and len(self.open_positions) < 5
        }

    def _generate_approval(
        self,
        risk_assessment: Dict[str, Any],
        limits_check: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate trade approval decision."""
        if not limits_check["can_trade"]:
            return {
                "approved": False,
                "reason": "Trading limits breached",
                "details": limits_check
            }

        if risk_assessment["risk_level"] == "high":
            return {
                "approved": False,
                "reason": f"Risk too high: {risk_assessment['risk_score']}/100",
                "suggestion": "Reduce position size or wait for better setup"
            }

        if risk_assessment["avg_confidence"] < 0.4:
            return {
                "approved": False,
                "reason": f"Low confidence: {risk_assessment['avg_confidence']}",
                "suggestion": "Wait for stronger signal alignment"
            }

        return {
            "approved": True,
            "reason": "Trade within risk parameters",
            "risk_level": risk_assessment["risk_level"],
            "confidence": risk_assessment["avg_confidence"]
        }

    def update_daily_pnl(self, pnl: float) -> None:
        """Update daily PnL tracker."""
        self.daily_pnl += pnl

    def reset_daily_stats(self) -> None:
        """Reset daily statistics."""
        self.daily_pnl = 0.0

    def add_position(self, position: Dict[str, Any]) -> None:
        """Track an open position."""
        self.open_positions.append(position)

    def remove_position(self, position_id: str) -> None:
        """Remove a closed position."""
        self.open_positions = [
            p for p in self.open_positions if p.get("id") != position_id
        ]

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message."""
        if message.message_type == "risk_check":
            analysis = await self.analyze(message.content)
            return await self.send_message(
                receiver=message.sender,
                content={"risk_analysis": analysis},
                message_type="risk_response"
            )
        return None
