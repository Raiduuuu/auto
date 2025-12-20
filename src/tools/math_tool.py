"""
Math Tool - Financial calculations for trading
"""
from typing import Any, Dict, List, Optional
from loguru import logger
import math

from .base_tool import BaseTool, ToolResult, ToolStatus


class MathTool(BaseTool):
    """
    Tool for performing financial and trading-related calculations.
    Supports position sizing, risk calculations, and statistical analysis.
    """

    def __init__(self):
        super().__init__(
            name="calculate",
            description="Perform trading calculations: position sizing, risk/reward, profit/loss, statistics"
        )

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "calculation": {
                    "type": "string",
                    "enum": [
                        "position_size",
                        "risk_reward",
                        "profit_loss",
                        "pip_value",
                        "margin_required",
                        "percentage_change",
                        "compound_return",
                        "sharpe_ratio",
                        "max_drawdown",
                        "volatility"
                    ],
                    "description": "Type of calculation to perform"
                },
                "params": {
                    "type": "object",
                    "description": "Parameters for the calculation"
                }
            },
            "required": ["calculation", "params"]
        }

    async def execute(self, **kwargs) -> ToolResult:
        calculation = kwargs.get("calculation")
        params = kwargs.get("params", {})

        logger.info(f"Math tool: {calculation}")

        try:
            if calculation == "position_size":
                result = self._calculate_position_size(params)
            elif calculation == "risk_reward":
                result = self._calculate_risk_reward(params)
            elif calculation == "profit_loss":
                result = self._calculate_profit_loss(params)
            elif calculation == "pip_value":
                result = self._calculate_pip_value(params)
            elif calculation == "margin_required":
                result = self._calculate_margin(params)
            elif calculation == "percentage_change":
                result = self._calculate_percentage_change(params)
            elif calculation == "compound_return":
                result = self._calculate_compound_return(params)
            elif calculation == "sharpe_ratio":
                result = self._calculate_sharpe_ratio(params)
            elif calculation == "max_drawdown":
                result = self._calculate_max_drawdown(params)
            elif calculation == "volatility":
                result = self._calculate_volatility(params)
            else:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolStatus.ERROR,
                    error=f"Unknown calculation: {calculation}"
                )

            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                data={
                    "calculation": calculation,
                    "params": params,
                    "result": result
                }
            )

        except Exception as e:
            logger.error(f"Math tool error: {e}")
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                error=str(e)
            )

    def _calculate_position_size(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate position size based on risk parameters.

        Params:
            account_balance: Total account balance
            risk_percent: Risk per trade (e.g., 1.0 for 1%)
            entry_price: Entry price
            stop_loss: Stop loss price
            pip_value: Value per pip (optional)
        """
        balance = params.get("account_balance", 10000)
        risk_percent = params.get("risk_percent", 1.0)
        entry = params.get("entry_price")
        stop_loss = params.get("stop_loss")
        pip_value = params.get("pip_value", 1.0)

        if not entry or not stop_loss:
            return {"error": "entry_price and stop_loss required"}

        risk_amount = balance * (risk_percent / 100)
        stop_distance = abs(entry - stop_loss)

        if stop_distance == 0:
            return {"error": "stop_loss cannot equal entry_price"}

        # Position size in units
        position_size = risk_amount / (stop_distance * pip_value)

        # Round to appropriate lot size
        lots = position_size / 100000  # Standard lot
        mini_lots = position_size / 10000
        micro_lots = position_size / 1000

        return {
            "risk_amount": round(risk_amount, 2),
            "stop_distance": round(stop_distance, 5),
            "position_size_units": round(position_size, 2),
            "lots": round(lots, 2),
            "mini_lots": round(mini_lots, 2),
            "micro_lots": round(micro_lots, 2),
            "recommended_lots": round(lots, 2)
        }

    def _calculate_risk_reward(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate risk/reward ratio.

        Params:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
        """
        entry = params.get("entry_price")
        stop_loss = params.get("stop_loss")
        take_profit = params.get("take_profit")

        if not all([entry, stop_loss, take_profit]):
            return {"error": "entry_price, stop_loss, and take_profit required"}

        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)

        if risk == 0:
            return {"error": "risk cannot be zero"}

        ratio = reward / risk

        return {
            "risk_points": round(risk, 5),
            "reward_points": round(reward, 5),
            "risk_reward_ratio": round(ratio, 2),
            "ratio_display": f"1:{round(ratio, 1)}",
            "favorable": ratio >= 1.5
        }

    def _calculate_profit_loss(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate profit/loss for a trade.

        Params:
            entry_price: Entry price
            exit_price: Exit price
            position_size: Position size in lots
            direction: "buy" or "sell"
            pip_value: Value per pip (default: 10 for standard lot)
        """
        entry = params.get("entry_price")
        exit_price = params.get("exit_price")
        size = params.get("position_size", 1.0)
        direction = params.get("direction", "buy")
        pip_value = params.get("pip_value", 10)

        if not entry or not exit_price:
            return {"error": "entry_price and exit_price required"}

        if direction.lower() == "buy":
            pips = exit_price - entry
        else:
            pips = entry - exit_price

        profit_loss = pips * size * pip_value

        return {
            "entry_price": entry,
            "exit_price": exit_price,
            "direction": direction,
            "pips": round(pips, 5),
            "profit_loss": round(profit_loss, 2),
            "profitable": profit_loss > 0
        }

    def _calculate_pip_value(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate pip value for an instrument.

        Params:
            instrument: Instrument symbol
            lot_size: Lot size (default: 1.0)
            account_currency: Account currency (default: EUR)
        """
        instrument = params.get("instrument", "DE40")
        lot_size = params.get("lot_size", 1.0)
        account_currency = params.get("account_currency", "EUR")

        # Pip values for common instruments (per standard lot)
        pip_values = {
            "DE40": 1.0,      # €1 per point per lot
            "US500": 1.0,    # $1 per point per lot
            "US30": 1.0,     # $1 per point per lot
            "EURUSD": 10.0,  # $10 per pip per lot
            "GBPUSD": 10.0,
            "XAUUSD": 1.0,   # $1 per 0.01 move per lot
        }

        base_pip_value = pip_values.get(instrument, 1.0)
        total_pip_value = base_pip_value * lot_size

        return {
            "instrument": instrument,
            "lot_size": lot_size,
            "pip_value_per_lot": base_pip_value,
            "total_pip_value": round(total_pip_value, 2),
            "account_currency": account_currency
        }

    def _calculate_margin(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate required margin.

        Params:
            price: Current price
            lot_size: Lot size
            leverage: Leverage ratio (e.g., 30 for 30:1)
            contract_size: Contract size (default: 100000)
        """
        price = params.get("price", 20000)
        lot_size = params.get("lot_size", 1.0)
        leverage = params.get("leverage", 30)
        contract_size = params.get("contract_size", 1)  # 1 unit for indices

        notional_value = price * lot_size * contract_size
        margin_required = notional_value / leverage

        return {
            "price": price,
            "lot_size": lot_size,
            "leverage": f"{leverage}:1",
            "notional_value": round(notional_value, 2),
            "margin_required": round(margin_required, 2)
        }

    def _calculate_percentage_change(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate percentage change between two values."""
        old_value = params.get("old_value")
        new_value = params.get("new_value")

        if not old_value or old_value == 0:
            return {"error": "old_value required and cannot be zero"}

        change = ((new_value - old_value) / old_value) * 100

        return {
            "old_value": old_value,
            "new_value": new_value,
            "change": round(new_value - old_value, 5),
            "percentage_change": round(change, 2),
            "direction": "up" if change > 0 else "down" if change < 0 else "flat"
        }

    def _calculate_compound_return(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate compound return.

        Params:
            initial_value: Starting value
            final_value: Ending value
            periods: Number of periods
        """
        initial = params.get("initial_value")
        final = params.get("final_value")
        periods = params.get("periods", 1)

        if not initial or initial <= 0:
            return {"error": "initial_value must be positive"}

        total_return = (final / initial) - 1
        cagr = (pow(final / initial, 1 / periods) - 1) * 100 if periods > 0 else 0

        return {
            "initial_value": initial,
            "final_value": final,
            "periods": periods,
            "total_return_percent": round(total_return * 100, 2),
            "compound_annual_return": round(cagr, 2)
        }

    def _calculate_sharpe_ratio(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate Sharpe ratio.

        Params:
            returns: List of period returns
            risk_free_rate: Annual risk-free rate (default: 0.02)
            periods_per_year: Number of periods per year (default: 252)
        """
        returns = params.get("returns", [])
        risk_free = params.get("risk_free_rate", 0.02)
        periods = params.get("periods_per_year", 252)

        if len(returns) < 2:
            return {"error": "At least 2 returns required"}

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            return {"error": "Standard deviation is zero"}

        # Annualize
        annual_return = mean_return * periods
        annual_std = std_dev * math.sqrt(periods)
        sharpe = (annual_return - risk_free) / annual_std

        return {
            "mean_return": round(mean_return, 6),
            "std_dev": round(std_dev, 6),
            "annualized_return": round(annual_return * 100, 2),
            "annualized_std": round(annual_std * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "quality": "excellent" if sharpe > 2 else "good" if sharpe > 1 else "acceptable" if sharpe > 0.5 else "poor"
        }

    def _calculate_max_drawdown(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate maximum drawdown.

        Params:
            equity_curve: List of equity values
        """
        equity = params.get("equity_curve", [])

        if len(equity) < 2:
            return {"error": "At least 2 equity values required"}

        peak = equity[0]
        max_dd = 0
        max_dd_start = 0
        max_dd_end = 0
        current_dd_start = 0

        for i, value in enumerate(equity):
            if value > peak:
                peak = value
                current_dd_start = i
            else:
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
                    max_dd_start = current_dd_start
                    max_dd_end = i

        return {
            "max_drawdown_percent": round(max_dd * 100, 2),
            "drawdown_start_index": max_dd_start,
            "drawdown_end_index": max_dd_end,
            "peak_value": peak,
            "trough_value": equity[max_dd_end] if max_dd_end < len(equity) else None,
            "acceptable": max_dd < 0.2  # Less than 20% is generally acceptable
        }

    def _calculate_volatility(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate price volatility.

        Params:
            prices: List of prices
            periods_per_year: Number of periods per year (default: 252)
        """
        prices = params.get("prices", [])
        periods = params.get("periods_per_year", 252)

        if len(prices) < 2:
            return {"error": "At least 2 prices required"}

        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])

        if not returns:
            return {"error": "Could not calculate returns"}

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        daily_vol = math.sqrt(variance)
        annual_vol = daily_vol * math.sqrt(periods)

        return {
            "daily_volatility": round(daily_vol * 100, 4),
            "annual_volatility": round(annual_vol * 100, 2),
            "volatility_level": "high" if annual_vol > 0.3 else "medium" if annual_vol > 0.15 else "low",
            "sample_size": len(returns)
        }
