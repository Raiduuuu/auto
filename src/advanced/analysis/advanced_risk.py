"""
Advanced Risk Manager for HRM Trading Bot

Implements professional-grade risk management:
- Dynamic Position Sizing (Kelly Criterion)
- Portfolio VaR (Value at Risk)
- Correlation-based Risk Assessment
- Liquidity Risk Analysis
- Emergency Protocols and Circuit Breakers
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open trading position."""
    instrument: str
    direction: str  # "long" or "short"
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    unrealized_pnl: float = 0.0

    @property
    def risk_amount(self) -> float:
        """Calculate risk amount based on stop loss."""
        if self.direction == "long":
            return (self.entry_price - self.stop_loss) * self.size
        else:
            return (self.stop_loss - self.entry_price) * self.size


@dataclass
class MarketConditions:
    """Current market conditions."""
    volatility: float
    liquidity: float  # 0 to 1 scale
    trend_strength: float  # -1 to 1
    correlation_regime: str  # "risk-on", "risk-off", "mixed"
    vix_level: Optional[float] = None
    spread_pct: float = 0.0


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    portfolio_var_95: float  # 95% VaR
    portfolio_var_99: float  # 99% VaR
    expected_shortfall: float  # CVaR
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: float
    correlation_risk: float
    liquidity_risk: float
    concentration_risk: float

    def to_dict(self) -> dict:
        return {
            "portfolio_var_95": self.portfolio_var_95,
            "portfolio_var_99": self.portfolio_var_99,
            "expected_shortfall": self.expected_shortfall,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "beta": self.beta,
            "correlation_risk": self.correlation_risk,
            "liquidity_risk": self.liquidity_risk,
            "concentration_risk": self.concentration_risk,
        }


@dataclass
class RiskAssessment:
    """Result of risk assessment for a trade."""
    approved: bool
    total_risk: float
    var_contribution: float
    correlation_risk: float
    liquidity_risk: float
    position_size: float
    stop_loss: float
    take_profit: float
    reason: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class EmergencyAction:
    """Emergency action to take."""
    action: str  # "reduce_positions", "halt_trading", "hedge", "continue"
    factor: float = 1.0
    duration: Optional[timedelta] = None
    reason: str = ""
    affected_positions: list[str] = field(default_factory=list)


class CircuitBreakers:
    """
    Circuit breakers for automatic trading halts.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.05,
        max_drawdown_pct: float = 0.10,
        max_volatility: float = 0.05,
        cooldown_period: timedelta = timedelta(hours=1),
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_volatility = max_volatility
        self.cooldown_period = cooldown_period

        self.is_halted = False
        self.halt_start: Optional[datetime] = None
        self.halt_reason: str = ""
        self.triggered_breakers: list[str] = []

    def check(
        self,
        daily_loss_pct: float,
        current_drawdown: float,
        volatility: float
    ) -> tuple[bool, str]:
        """
        Check if any circuit breaker should be triggered.

        Returns:
            Tuple of (should_halt, reason)
        """
        # Check if currently halted
        if self.is_halted:
            if datetime.now() - self.halt_start >= self.cooldown_period:
                self.reset()
            else:
                remaining = self.cooldown_period - (datetime.now() - self.halt_start)
                return True, f"Trading halted, {remaining.seconds}s remaining"

        # Daily loss check
        if daily_loss_pct >= self.max_daily_loss_pct:
            self._trigger("daily_loss", f"Daily loss {daily_loss_pct:.2%} exceeds limit")
            return True, self.halt_reason

        # Drawdown check
        if current_drawdown >= self.max_drawdown_pct:
            self._trigger("drawdown", f"Drawdown {current_drawdown:.2%} exceeds limit")
            return True, self.halt_reason

        # Volatility check
        if volatility >= self.max_volatility:
            self._trigger("volatility", f"Volatility {volatility:.2%} exceeds limit")
            return True, self.halt_reason

        return False, ""

    def _trigger(self, breaker_name: str, reason: str):
        """Trigger a circuit breaker."""
        self.is_halted = True
        self.halt_start = datetime.now()
        self.halt_reason = reason
        self.triggered_breakers.append(breaker_name)
        logger.warning(f"Circuit breaker triggered: {reason}")

    def reset(self):
        """Reset circuit breakers."""
        self.is_halted = False
        self.halt_start = None
        self.halt_reason = ""


class VaRCalculator:
    """
    Value at Risk calculator using multiple methods.
    """

    def __init__(self, confidence_levels: list[float] = None):
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.returns_history: deque = deque(maxlen=1000)

    def add_return(self, return_pct: float):
        """Add a return observation."""
        self.returns_history.append(return_pct)

    def calculate_historical_var(
        self,
        portfolio_value: float,
        confidence: float = 0.95
    ) -> float:
        """Calculate VaR using historical simulation."""
        if len(self.returns_history) < 30:
            return 0.0

        returns = np.array(self.returns_history)
        var_pct = np.percentile(returns, (1 - confidence) * 100)

        return abs(var_pct * portfolio_value)

    def calculate_parametric_var(
        self,
        portfolio_value: float,
        confidence: float = 0.95
    ) -> float:
        """Calculate VaR using parametric (normal) method."""
        if len(self.returns_history) < 30:
            return 0.0

        returns = np.array(self.returns_history)
        mu = np.mean(returns)
        sigma = np.std(returns)

        z_score = stats.norm.ppf(1 - confidence)
        var_pct = mu + z_score * sigma

        return abs(var_pct * portfolio_value)

    def calculate_expected_shortfall(
        self,
        portfolio_value: float,
        confidence: float = 0.95
    ) -> float:
        """Calculate Expected Shortfall (CVaR)."""
        if len(self.returns_history) < 30:
            return 0.0

        returns = np.array(self.returns_history)
        var_pct = np.percentile(returns, (1 - confidence) * 100)

        # ES is the average of returns below VaR
        tail_returns = returns[returns <= var_pct]
        if len(tail_returns) == 0:
            return abs(var_pct * portfolio_value)

        es_pct = np.mean(tail_returns)
        return abs(es_pct * portfolio_value)


class PositionSizer:
    """
    Dynamic position sizing using multiple methods.
    """

    def __init__(
        self,
        max_position_size: float = 0.05,
        max_total_exposure: float = 0.50,
    ):
        self.max_position_size = max_position_size
        self.max_total_exposure = max_total_exposure

    def kelly_criterion(
        self,
        win_probability: float,
        win_loss_ratio: float,
        fraction: float = 0.25  # Use fractional Kelly for safety
    ) -> float:
        """
        Calculate position size using Kelly Criterion.

        Args:
            win_probability: Probability of winning trade
            win_loss_ratio: Average win / average loss
            fraction: Fraction of Kelly to use (0.25 = quarter Kelly)

        Returns:
            Optimal position size as fraction of capital
        """
        if win_probability <= 0 or win_probability >= 1:
            return 0.0

        if win_loss_ratio <= 0:
            return 0.0

        # Kelly formula: f = (bp - q) / b
        # where b = win_loss_ratio, p = win_prob, q = loss_prob
        b = win_loss_ratio
        p = win_probability
        q = 1 - p

        kelly_fraction = (b * p - q) / b

        # Apply fractional Kelly and cap
        position_size = kelly_fraction * fraction
        position_size = max(0, min(position_size, self.max_position_size))

        return position_size

    def volatility_adjusted_size(
        self,
        base_size: float,
        current_volatility: float,
        target_volatility: float = 0.15
    ) -> float:
        """
        Adjust position size based on volatility.

        Args:
            base_size: Base position size
            current_volatility: Current annualized volatility
            target_volatility: Target volatility level

        Returns:
            Volatility-adjusted position size
        """
        if current_volatility <= 0:
            return base_size

        vol_adjustment = target_volatility / current_volatility
        adjusted_size = base_size * vol_adjustment

        return min(adjusted_size, self.max_position_size)

    def calculate_optimal_size(
        self,
        prediction_confidence: float,
        historical_win_rate: float,
        historical_win_loss_ratio: float,
        current_volatility: float,
        current_exposure: float,
        account_balance: float
    ) -> float:
        """
        Calculate optimal position size considering all factors.

        Args:
            prediction_confidence: Model's confidence in prediction
            historical_win_rate: Historical win rate
            historical_win_loss_ratio: Historical win/loss ratio
            current_volatility: Current market volatility
            current_exposure: Current total exposure
            account_balance: Current account balance

        Returns:
            Optimal position size in currency units
        """
        # Base size from Kelly
        kelly_size = self.kelly_criterion(
            win_probability=historical_win_rate,
            win_loss_ratio=historical_win_loss_ratio,
        )

        # Adjust for confidence
        confidence_adjusted = kelly_size * prediction_confidence

        # Adjust for volatility
        vol_adjusted = self.volatility_adjusted_size(
            confidence_adjusted,
            current_volatility,
        )

        # Check exposure limit
        available_exposure = self.max_total_exposure - current_exposure
        final_size = min(vol_adjusted, available_exposure)

        # Convert to currency
        position_value = final_size * account_balance

        return max(0, position_value)


class AdvancedRiskManager:
    """
    Comprehensive risk management system.

    Features:
    - Dynamic position sizing
    - Portfolio VaR calculation
    - Correlation risk assessment
    - Liquidity risk monitoring
    - Emergency protocols
    - Circuit breakers
    """

    def __init__(
        self,
        max_portfolio_risk: float = 0.02,
        max_single_position: float = 0.005,
        max_daily_loss: float = 0.05,
        max_drawdown: float = 0.10,
    ):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_single_position = max_single_position
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown

        # Components
        self.circuit_breakers = CircuitBreakers(
            max_daily_loss_pct=max_daily_loss,
            max_drawdown_pct=max_drawdown,
        )
        self.var_calculator = VaRCalculator()
        self.position_sizer = PositionSizer(
            max_position_size=max_single_position,
            max_total_exposure=0.5,
        )

        # State
        self.positions: dict[str, Position] = {}
        self.daily_pnl: float = 0.0
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0
        self.returns_history: deque = deque(maxlen=252)

        # Correlation tracking
        self.price_history: dict[str, deque] = {}
        self.correlation_matrix: Optional[np.ndarray] = None

        # Statistics
        self.trade_count = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0

        logger.info(
            f"AdvancedRiskManager initialized: "
            f"max_risk={max_portfolio_risk:.1%}, max_position={max_single_position:.1%}"
        )

    def dynamic_position_sizing(
        self,
        prediction_confidence: float,
        market_conditions: MarketConditions,
        account_balance: float
    ) -> float:
        """
        Calculate dynamic position size based on all factors.

        Args:
            prediction_confidence: Model's prediction confidence
            market_conditions: Current market conditions
            account_balance: Current account balance

        Returns:
            Optimal position size in currency units
        """
        # Calculate current exposure
        current_exposure = sum(
            pos.size * pos.current_price / account_balance
            for pos in self.positions.values()
        )

        # Historical statistics
        if self.trade_count > 10:
            win_rate = self.winning_trades / self.trade_count
            if self.total_loss > 0:
                win_loss_ratio = abs(self.total_profit / self.total_loss)
            else:
                win_loss_ratio = 2.0
        else:
            win_rate = 0.5
            win_loss_ratio = 1.5

        # Calculate optimal size
        position_size = self.position_sizer.calculate_optimal_size(
            prediction_confidence=prediction_confidence,
            historical_win_rate=win_rate,
            historical_win_loss_ratio=win_loss_ratio,
            current_volatility=market_conditions.volatility,
            current_exposure=current_exposure,
            account_balance=account_balance,
        )

        # Additional adjustments based on market conditions
        if market_conditions.liquidity < 0.3:
            position_size *= 0.5  # Reduce for low liquidity
        if market_conditions.volatility > 0.03:
            position_size *= 0.7  # Reduce for high volatility

        return position_size

    def assess_portfolio_risk(
        self,
        new_position: Optional[Position] = None
    ) -> RiskMetrics:
        """
        Comprehensive portfolio risk assessment.

        Args:
            new_position: Optional new position to include in assessment

        Returns:
            RiskMetrics with all risk measures
        """
        positions = list(self.positions.values())
        if new_position:
            positions.append(new_position)

        # Calculate VaR
        var_95 = self.var_calculator.calculate_historical_var(
            self.current_equity, 0.95
        )
        var_99 = self.var_calculator.calculate_historical_var(
            self.current_equity, 0.99
        )
        es = self.var_calculator.calculate_expected_shortfall(
            self.current_equity, 0.95
        )

        # Drawdown
        if self.peak_equity > 0:
            current_dd = (self.peak_equity - self.current_equity) / self.peak_equity
        else:
            current_dd = 0.0

        max_dd = self._calculate_max_drawdown()

        # Risk ratios
        sharpe = self._calculate_sharpe_ratio()
        sortino = self._calculate_sortino_ratio()
        calmar = abs(sharpe / max_dd) if max_dd > 0 else 0.0

        # Correlation risk
        corr_risk = self._calculate_correlation_risk(positions)

        # Liquidity risk
        liq_risk = self._calculate_liquidity_risk(positions)

        # Concentration risk
        conc_risk = self._calculate_concentration_risk(positions)

        return RiskMetrics(
            portfolio_var_95=var_95,
            portfolio_var_99=var_99,
            expected_shortfall=es,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            beta=1.0,  # Simplified
            correlation_risk=corr_risk,
            liquidity_risk=liq_risk,
            concentration_risk=conc_risk,
        )

    def assess_trade(
        self,
        instrument: str,
        direction: str,
        entry_price: float,
        prediction_confidence: float,
        market_conditions: MarketConditions,
        account_balance: float,
        atr: float,
    ) -> RiskAssessment:
        """
        Assess risk of a proposed trade.

        Args:
            instrument: Trading instrument
            direction: "long" or "short"
            entry_price: Proposed entry price
            prediction_confidence: Model confidence
            market_conditions: Current conditions
            account_balance: Account balance
            atr: Average True Range

        Returns:
            RiskAssessment with approval decision
        """
        warnings = []

        # Check circuit breakers
        halted, reason = self.circuit_breakers.check(
            self.daily_pnl / account_balance if account_balance > 0 else 0,
            (self.peak_equity - self.current_equity) / self.peak_equity if self.peak_equity > 0 else 0,
            market_conditions.volatility,
        )
        if halted:
            return RiskAssessment(
                approved=False,
                total_risk=0,
                var_contribution=0,
                correlation_risk=0,
                liquidity_risk=0,
                position_size=0,
                stop_loss=0,
                take_profit=0,
                reason=reason,
            )

        # Calculate position size
        position_size = self.dynamic_position_sizing(
            prediction_confidence,
            market_conditions,
            account_balance,
        )

        if position_size == 0:
            return RiskAssessment(
                approved=False,
                total_risk=0,
                var_contribution=0,
                correlation_risk=0,
                liquidity_risk=0,
                position_size=0,
                stop_loss=0,
                take_profit=0,
                reason="Position size calculated as zero",
            )

        # Calculate stop loss and take profit
        sl_distance = 2 * atr
        tp_distance = 3 * atr  # 1.5:1 reward/risk

        if direction == "long":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        # Create hypothetical position
        hyp_position = Position(
            instrument=instrument,
            direction=direction,
            size=position_size / entry_price,  # Convert to units
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now(),
        )

        # Assess portfolio risk with new position
        risk_metrics = self.assess_portfolio_risk(hyp_position)

        # Calculate risk contributions
        var_contribution = (
            risk_metrics.portfolio_var_95 -
            self.var_calculator.calculate_historical_var(self.current_equity, 0.95)
        )

        # Check limits
        total_risk = (
            var_contribution / account_balance +
            risk_metrics.correlation_risk * 0.3 +
            risk_metrics.liquidity_risk * 0.2
        )

        # Generate warnings
        if risk_metrics.correlation_risk > 0.5:
            warnings.append(f"High correlation risk: {risk_metrics.correlation_risk:.2f}")
        if risk_metrics.liquidity_risk > 0.5:
            warnings.append(f"High liquidity risk: {risk_metrics.liquidity_risk:.2f}")
        if market_conditions.volatility > 0.03:
            warnings.append(f"High volatility: {market_conditions.volatility:.2%}")

        # Make decision
        approved = (
            total_risk <= self.max_portfolio_risk and
            position_size / account_balance <= self.max_single_position and
            prediction_confidence >= 0.5
        )

        if not approved:
            if total_risk > self.max_portfolio_risk:
                reason = f"Total risk {total_risk:.2%} exceeds limit {self.max_portfolio_risk:.2%}"
            elif position_size / account_balance > self.max_single_position:
                reason = f"Position size exceeds limit"
            else:
                reason = f"Low confidence: {prediction_confidence:.2%}"
        else:
            reason = "Trade approved"

        return RiskAssessment(
            approved=approved,
            total_risk=total_risk,
            var_contribution=var_contribution,
            correlation_risk=risk_metrics.correlation_risk,
            liquidity_risk=risk_metrics.liquidity_risk,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
            warnings=warnings,
        )

    def emergency_protocols(
        self,
        market_conditions: MarketConditions
    ) -> EmergencyAction:
        """
        Determine emergency action based on market conditions.

        Args:
            market_conditions: Current market conditions

        Returns:
            EmergencyAction to take
        """
        # High volatility protocol
        if market_conditions.volatility > 0.05:
            return EmergencyAction(
                action="reduce_positions",
                factor=0.5,
                reason="extreme_volatility",
                affected_positions=list(self.positions.keys()),
            )

        # Low liquidity protocol
        if market_conditions.liquidity < 0.2:
            return EmergencyAction(
                action="halt_trading",
                duration=timedelta(hours=1),
                reason="low_liquidity",
            )

        # Market crash detection (high vol + strong downtrend)
        if market_conditions.volatility > 0.04 and market_conditions.trend_strength < -0.5:
            return EmergencyAction(
                action="hedge",
                factor=0.8,
                reason="potential_crash",
                affected_positions=[
                    pos.instrument for pos in self.positions.values()
                    if pos.direction == "long"
                ],
            )

        return EmergencyAction(
            action="continue",
            reason="normal_conditions",
        )

    def update_position(
        self,
        instrument: str,
        current_price: float
    ):
        """Update position with current price."""
        if instrument not in self.positions:
            return

        pos = self.positions[instrument]
        pos.current_price = current_price

        # Calculate unrealized PnL
        if pos.direction == "long":
            pos.unrealized_pnl = (current_price - pos.entry_price) * pos.size
        else:
            pos.unrealized_pnl = (pos.entry_price - current_price) * pos.size

    def close_position(
        self,
        instrument: str,
        close_price: float
    ) -> float:
        """
        Close a position and update statistics.

        Returns:
            Realized PnL
        """
        if instrument not in self.positions:
            return 0.0

        pos = self.positions[instrument]

        # Calculate realized PnL
        if pos.direction == "long":
            pnl = (close_price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - close_price) * pos.size

        # Update statistics
        self.trade_count += 1
        self.daily_pnl += pnl

        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
        else:
            self.total_loss += abs(pnl)

        # Update equity
        self.current_equity += pnl
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        # Add to returns history
        if self.current_equity > 0:
            return_pct = pnl / (self.current_equity - pnl)
            self.returns_history.append(return_pct)
            self.var_calculator.add_return(return_pct)

        # Remove position
        del self.positions[instrument]

        logger.info(f"Closed position {instrument}: PnL={pnl:.2f}")
        return pnl

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from returns history."""
        if len(self.returns_history) < 2:
            return 0.0

        returns = np.array(self.returns_history)
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak

        return float(np.max(drawdown))

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        if len(self.returns_history) < 30:
            return 0.0

        returns = np.array(self.returns_history)
        if np.std(returns) == 0:
            return 0.0

        return float(np.mean(returns) / np.std(returns) * np.sqrt(252))

    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (downside risk adjusted)."""
        if len(self.returns_history) < 30:
            return 0.0

        returns = np.array(self.returns_history)
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0 or np.std(negative_returns) == 0:
            return float(np.mean(returns) * np.sqrt(252))

        return float(np.mean(returns) / np.std(negative_returns) * np.sqrt(252))

    def _calculate_correlation_risk(self, positions: list[Position]) -> float:
        """Calculate correlation-based risk."""
        if len(positions) < 2:
            return 0.0

        # Simplified: based on number of same-direction positions
        long_count = sum(1 for p in positions if p.direction == "long")
        short_count = len(positions) - long_count

        # High correlation if all same direction
        imbalance = abs(long_count - short_count) / len(positions)
        return float(imbalance)

    def _calculate_liquidity_risk(self, positions: list[Position]) -> float:
        """Calculate liquidity risk."""
        if not positions:
            return 0.0

        # Simplified: larger positions = higher liquidity risk
        total_value = sum(p.size * p.current_price for p in positions)
        avg_position = total_value / len(positions)

        # Normalize to 0-1
        return float(min(avg_position / 100000, 1.0))

    def _calculate_concentration_risk(self, positions: list[Position]) -> float:
        """Calculate concentration risk."""
        if not positions:
            return 0.0

        total_value = sum(p.size * p.current_price for p in positions)
        if total_value == 0:
            return 0.0

        # Herfindahl index
        weights = [p.size * p.current_price / total_value for p in positions]
        hhi = sum(w ** 2 for w in weights)

        return float(hhi)

    def get_state(self) -> dict:
        """Get risk manager state."""
        return {
            "positions_count": len(self.positions),
            "daily_pnl": self.daily_pnl,
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "trade_count": self.trade_count,
            "win_rate": self.winning_trades / self.trade_count if self.trade_count > 0 else 0,
            "total_profit": self.total_profit,
            "total_loss": self.total_loss,
            "circuit_breaker_halted": self.circuit_breakers.is_halted,
        }

    def reset_daily(self):
        """Reset daily statistics."""
        self.daily_pnl = 0.0
        self.circuit_breakers.reset()
