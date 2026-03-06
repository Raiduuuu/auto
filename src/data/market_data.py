"""
Market Data Provider - Handles market data retrieval and caching
"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional
import pandas as pd
from loguru import logger


class MarketDataProvider:
    """
    Provides market data for trading agents.
    Supports multiple data sources and caching.
    """

    def __init__(self, ctrader_client: Any = None):
        self.ctrader_client = ctrader_client
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 60  # seconds

    async def get_ohlcv(
        self,
        instrument: str,
        timeframe: str = "H1",
        periods: int = 100
    ) -> pd.DataFrame:
        """
        Get OHLCV data for an instrument.
        """
        cache_key = f"{instrument}_{timeframe}_{periods}"

        # Check cache
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if datetime.now(timezone.utc) - cached["timestamp"] < timedelta(seconds=self.cache_ttl):
                return cached["data"]

        # Fetch from cTrader
        if self.ctrader_client:
            try:
                data = await self.ctrader_client.get_historical_data(
                    instrument=instrument,
                    timeframe=timeframe,
                    periods=periods
                )
                df = self._to_dataframe(data)
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                df = self._generate_sample_data(periods)
        else:
            # Generate sample data for testing
            df = self._generate_sample_data(periods)

        # Cache the data
        self.cache[cache_key] = {
            "data": df,
            "timestamp": datetime.now(timezone.utc)
        }

        return df

    async def get_current_price(self, instrument: str) -> Dict[str, float]:
        """Get current bid/ask prices."""
        if self.ctrader_client:
            try:
                return await self.ctrader_client.get_quote(instrument)
            except Exception as e:
                logger.error(f"Error getting price: {e}")

        # Sample price for testing
        return {
            "bid": 18500.0,
            "ask": 18501.0,
            "spread": 1.0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        if self.ctrader_client:
            try:
                return await self.ctrader_client.get_account()
            except Exception as e:
                logger.error(f"Error getting account: {e}")

        # Sample account for testing
        return {
            "balance": 10000.0,
            "equity": 10000.0,
            "margin_used": 0.0,
            "free_margin": 10000.0,
            "margin_level": 0.0,
            "currency": "EUR"
        }

    def _to_dataframe(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert raw data to DataFrame."""
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df

    def _generate_sample_data(self, periods: int) -> pd.DataFrame:
        """Generate sample OHLCV data for testing."""
        import numpy as np

        base_price = 18500.0
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=periods, freq='h')

        # Generate random walk
        returns = np.random.normal(0, 0.001, periods)
        prices = base_price * (1 + returns).cumprod()

        data = {
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, periods))),
            'close': prices * (1 + np.random.normal(0, 0.001, periods)),
            'volume': np.random.randint(1000, 10000, periods)
        }

        df = pd.DataFrame(data, index=dates)
        return df

    def clear_cache(self) -> None:
        """Clear data cache."""
        self.cache = {}

    async def subscribe_to_prices(
        self,
        instruments: List[str],
        callback: Callable
    ) -> None:
        """Subscribe to real-time price updates."""
        if self.ctrader_client:
            await self.ctrader_client.subscribe_prices(instruments, callback)
        else:
            logger.warning("No cTrader client - price subscription unavailable")
