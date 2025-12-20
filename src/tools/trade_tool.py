"""
Trade Tool - Execute trading operations
"""
from typing import Any, Dict, Optional
from loguru import logger

from .base_tool import BaseTool, ToolResult, ToolStatus


class TradeTool(BaseTool):
    """
    Tool for executing trades via cTrader.
    Supports market orders, limit orders, and position management.
    """

    def __init__(self, ctrader_client: Any = None):
        super().__init__(
            name="trade",
            description="Execute trading operations: place orders, close positions, modify orders"
        )
        self.ctrader = ctrader_client

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["buy", "sell", "close", "modify", "cancel"],
                    "description": "Trading action to perform"
                },
                "instrument": {
                    "type": "string",
                    "description": "Instrument symbol (e.g., DE40, EURUSD)"
                },
                "volume": {
                    "type": "number",
                    "description": "Trade volume in lots"
                },
                "order_type": {
                    "type": "string",
                    "enum": ["market", "limit", "stop"],
                    "default": "market"
                },
                "price": {
                    "type": "number",
                    "description": "Price for limit/stop orders"
                },
                "stop_loss": {
                    "type": "number",
                    "description": "Stop loss price"
                },
                "take_profit": {
                    "type": "number",
                    "description": "Take profit price"
                },
                "position_id": {
                    "type": "string",
                    "description": "Position ID for close/modify operations"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for the trade (for logging)"
                }
            },
            "required": ["action", "instrument"]
        }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action")
        instrument = kwargs.get("instrument")
        volume = kwargs.get("volume", 0.1)
        order_type = kwargs.get("order_type", "market")
        price = kwargs.get("price")
        stop_loss = kwargs.get("stop_loss")
        take_profit = kwargs.get("take_profit")
        position_id = kwargs.get("position_id")
        reason = kwargs.get("reason", "")

        logger.info(f"Trade tool executing: {action} {instrument} vol={volume}")

        if not self.ctrader:
            # Simulation mode
            return await self._simulate_trade(
                action, instrument, volume, order_type,
                price, stop_loss, take_profit, reason
            )

        try:
            if action == "buy":
                result = await self._execute_buy(
                    instrument, volume, order_type, price, stop_loss, take_profit
                )
            elif action == "sell":
                result = await self._execute_sell(
                    instrument, volume, order_type, price, stop_loss, take_profit
                )
            elif action == "close":
                result = await self._execute_close(position_id, volume)
            elif action == "modify":
                result = await self._execute_modify(position_id, stop_loss, take_profit)
            elif action == "cancel":
                result = await self._execute_cancel(position_id)
            else:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolStatus.ERROR,
                    error=f"Unknown action: {action}"
                )

            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                data={
                    "action": action,
                    "instrument": instrument,
                    "volume": volume,
                    "result": result,
                    "reason": reason
                }
            )

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                error=str(e)
            )

    async def _simulate_trade(
        self, action: str, instrument: str, volume: float,
        order_type: str, price: Optional[float],
        stop_loss: Optional[float], take_profit: Optional[float],
        reason: str
    ) -> ToolResult:
        """Simulate trade for testing."""
        simulated_price = price or 20000.0  # Default DAX price

        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            data={
                "action": action,
                "instrument": instrument,
                "volume": volume,
                "order_type": order_type,
                "executed_price": simulated_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "reason": reason,
                "simulated": True,
                "order_id": f"SIM-{instrument}-{action.upper()}-001"
            }
        )

    async def _execute_buy(
        self, instrument: str, volume: float, order_type: str,
        price: Optional[float], stop_loss: Optional[float], take_profit: Optional[float]
    ) -> Dict[str, Any]:
        """Execute buy order."""
        if order_type == "market":
            order = await self.ctrader.place_market_order(
                symbol=instrument,
                side="BUY",
                volume=volume,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        else:
            order = await self.ctrader.place_limit_order(
                symbol=instrument,
                side="BUY",
                volume=volume,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        return order

    async def _execute_sell(
        self, instrument: str, volume: float, order_type: str,
        price: Optional[float], stop_loss: Optional[float], take_profit: Optional[float]
    ) -> Dict[str, Any]:
        """Execute sell order."""
        if order_type == "market":
            order = await self.ctrader.place_market_order(
                symbol=instrument,
                side="SELL",
                volume=volume,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        else:
            order = await self.ctrader.place_limit_order(
                symbol=instrument,
                side="SELL",
                volume=volume,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        return order

    async def _execute_close(
        self, position_id: str, volume: Optional[float]
    ) -> Dict[str, Any]:
        """Close a position."""
        return await self.ctrader.close_position(
            position_id=position_id,
            volume=volume
        )

    async def _execute_modify(
        self, position_id: str, stop_loss: Optional[float], take_profit: Optional[float]
    ) -> Dict[str, Any]:
        """Modify position stop loss / take profit."""
        return await self.ctrader.modify_position(
            position_id=position_id,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    async def _execute_cancel(self, order_id: str) -> Dict[str, Any]:
        """Cancel a pending order."""
        return await self.ctrader.cancel_order(order_id=order_id)
