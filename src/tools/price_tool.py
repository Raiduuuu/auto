"""
Price Tool - Get market price data
"""
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd

from .base_tool import BaseTool, ToolResult, ToolStatus


class PriceTool(BaseTool):
    """
    Tool for retrieving market price data.
    Supports real-time quotes, historical OHLCV, and technical indicators.
    """

    def __init__(self, market_data_provider: Any = None):
        super().__init__(
            name="get_price",
            description="Get market prices: real-time quotes, historical OHLCV data, and technical indicators"
        )
        self.market_data = market_data_provider

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "instrument": {
                    "type": "string",
                    "description": "Instrument symbol (e.g., DE40, EURUSD, XAUUSD)"
                },
                "data_type": {
                    "type": "string",
                    "enum": ["quote", "ohlcv", "indicators"],
                    "default": "quote",
                    "description": "Type of price data to retrieve"
                },
                "timeframe": {
                    "type": "string",
                    "enum": ["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
                    "default": "M5",
                    "description": "Timeframe for OHLCV data"
                },
                "periods": {
                    "type": "integer",
                    "default": 100,
                    "description": "Number of periods to retrieve"
                },
                "indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of indicators to calculate (e.g., ['sma_20', 'rsi', 'macd'])"
                }
            },
            "required": ["instrument"]
        }

    async def execute(self, **kwargs) -> ToolResult:
        instrument = kwargs.get("instrument")
        data_type = kwargs.get("data_type", "quote")
        timeframe = kwargs.get("timeframe", "M5")
        periods = kwargs.get("periods", 100)
        indicators = kwargs.get("indicators", [])

        logger.info(f"Price tool: {data_type} for {instrument}")

        try:
            if data_type == "quote":
                data = await self._get_quote(instrument)
            elif data_type == "ohlcv":
                data = await self._get_ohlcv(instrument, timeframe, periods)
            elif data_type == "indicators":
                data = await self._get_indicators(instrument, timeframe, periods, indicators)
            else:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolStatus.ERROR,
                    error=f"Unknown data_type: {data_type}"
                )

            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                data=data
            )

        except Exception as e:
            logger.error(f"Price tool error: {e}")
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                error=str(e)
            )

    async def _get_quote(self, instrument: str) -> Dict[str, Any]:
        """Get real-time quote."""
        if self.market_data:
            try:
                quote = await self.market_data.get_quote(instrument)
                return {
                    "instrument": instrument,
                    "bid": quote.get("bid"),
                    "ask": quote.get("ask"),
                    "spread": quote.get("ask", 0) - quote.get("bid", 0),
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception:
                pass

        # Fallback: simulated quote
        import random
        base_prices = {
            "DE40": 20000.0,
            "US500": 5800.0,
            "EURUSD": 1.0850,
            "XAUUSD": 2650.0
        }
        base = base_prices.get(instrument, 1000.0)
        bid = base + random.uniform(-10, 10)
        spread = base * 0.0001  # 1 pip spread

        return {
            "instrument": instrument,
            "bid": round(bid, 2),
            "ask": round(bid + spread, 2),
            "spread": round(spread, 5),
            "timestamp": datetime.utcnow().isoformat(),
            "simulated": True
        }

    async def _get_ohlcv(
        self, instrument: str, timeframe: str, periods: int
    ) -> Dict[str, Any]:
        """Get historical OHLCV data."""
        if self.market_data:
            try:
                df = await self.market_data.get_ohlcv(
                    symbol=instrument,
                    timeframe=timeframe,
                    limit=periods
                )
                if df is not None and not df.empty:
                    return self._dataframe_to_dict(df, instrument, timeframe)
            except Exception:
                pass

        # Fallback: generate sample data
        return self._generate_sample_ohlcv(instrument, timeframe, periods)

    async def _get_indicators(
        self, instrument: str, timeframe: str, periods: int, indicators: List[str]
    ) -> Dict[str, Any]:
        """Get OHLCV data with calculated indicators."""
        ohlcv_result = await self._get_ohlcv(instrument, timeframe, periods)

        if not indicators:
            indicators = ["sma_20", "sma_50", "rsi", "macd", "atr", "bbands"]

        # Calculate indicators
        from ..data.indicators import TechnicalIndicators

        # Convert back to dataframe
        df = pd.DataFrame(ohlcv_result.get("candles", []))
        if df.empty:
            return ohlcv_result

        indicator_values = {}

        for ind in indicators:
            try:
                if ind.startswith("sma_"):
                    period = int(ind.split("_")[1])
                    values = TechnicalIndicators.sma(df["close"], period)
                    indicator_values[ind] = round(values.iloc[-1], 2) if not values.empty else None
                elif ind.startswith("ema_"):
                    period = int(ind.split("_")[1])
                    values = TechnicalIndicators.ema(df["close"], period)
                    indicator_values[ind] = round(values.iloc[-1], 2) if not values.empty else None
                elif ind == "rsi":
                    values = TechnicalIndicators.rsi(df["close"])
                    indicator_values["rsi"] = round(values.iloc[-1], 2) if not values.empty else None
                elif ind == "macd":
                    macd_line, signal, hist = TechnicalIndicators.macd(df["close"])
                    indicator_values["macd"] = {
                        "macd_line": round(macd_line.iloc[-1], 4) if not macd_line.empty else None,
                        "signal": round(signal.iloc[-1], 4) if not signal.empty else None,
                        "histogram": round(hist.iloc[-1], 4) if not hist.empty else None
                    }
                elif ind == "atr":
                    values = TechnicalIndicators.atr(df["high"], df["low"], df["close"])
                    indicator_values["atr"] = round(values.iloc[-1], 2) if not values.empty else None
                elif ind == "bbands":
                    upper, middle, lower = TechnicalIndicators.bollinger_bands(df["close"])
                    indicator_values["bbands"] = {
                        "upper": round(upper.iloc[-1], 2) if not upper.empty else None,
                        "middle": round(middle.iloc[-1], 2) if not middle.empty else None,
                        "lower": round(lower.iloc[-1], 2) if not lower.empty else None
                    }
                elif ind == "stochastic":
                    k, d = TechnicalIndicators.stochastic(df["high"], df["low"], df["close"])
                    indicator_values["stochastic"] = {
                        "k": round(k.iloc[-1], 2) if not k.empty else None,
                        "d": round(d.iloc[-1], 2) if not d.empty else None
                    }
                elif ind == "adx":
                    values = TechnicalIndicators.adx(df["high"], df["low"], df["close"])
                    indicator_values["adx"] = round(values.iloc[-1], 2) if not values.empty else None
            except Exception as e:
                logger.warning(f"Could not calculate {ind}: {e}")
                indicator_values[ind] = None

        ohlcv_result["indicators"] = indicator_values
        return ohlcv_result

    def _dataframe_to_dict(
        self, df: pd.DataFrame, instrument: str, timeframe: str
    ) -> Dict[str, Any]:
        """Convert DataFrame to dict format."""
        candles = []
        for _, row in df.tail(50).iterrows():  # Last 50 candles
            candles.append({
                "timestamp": row.get("timestamp", row.name).isoformat() if hasattr(row.get("timestamp", row.name), "isoformat") else str(row.get("timestamp", row.name)),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0))
            })

        latest = df.iloc[-1]
        return {
            "instrument": instrument,
            "timeframe": timeframe,
            "latest": {
                "open": float(latest["open"]),
                "high": float(latest["high"]),
                "low": float(latest["low"]),
                "close": float(latest["close"]),
                "volume": float(latest.get("volume", 0))
            },
            "candles": candles,
            "count": len(candles)
        }

    def _generate_sample_ohlcv(
        self, instrument: str, timeframe: str, periods: int
    ) -> Dict[str, Any]:
        """Generate sample OHLCV data for testing."""
        import random
        import numpy as np

        base_prices = {
            "DE40": 20000.0,
            "US500": 5800.0,
            "EURUSD": 1.0850,
            "XAUUSD": 2650.0
        }
        base = base_prices.get(instrument, 1000.0)
        volatility = base * 0.001  # 0.1% volatility

        candles = []
        price = base

        timeframe_minutes = {
            "M1": 1, "M5": 5, "M15": 15, "M30": 30,
            "H1": 60, "H4": 240, "D1": 1440
        }
        minutes = timeframe_minutes.get(timeframe, 5)

        now = datetime.utcnow()

        for i in range(min(periods, 50)):
            # Random walk
            change = random.gauss(0, volatility)
            open_price = price
            close_price = price + change
            high_price = max(open_price, close_price) + abs(random.gauss(0, volatility * 0.5))
            low_price = min(open_price, close_price) - abs(random.gauss(0, volatility * 0.5))

            ts = now - timedelta(minutes=minutes * (periods - i))

            candles.append({
                "timestamp": ts.isoformat(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": random.randint(1000, 10000)
            })

            price = close_price

        latest = candles[-1] if candles else {}
        return {
            "instrument": instrument,
            "timeframe": timeframe,
            "latest": {
                "open": latest.get("open"),
                "high": latest.get("high"),
                "low": latest.get("low"),
                "close": latest.get("close"),
                "volume": latest.get("volume")
            },
            "candles": candles,
            "count": len(candles),
            "simulated": True
        }
