"""
Technical Indicators - Calculate various technical indicators
"""
import pandas as pd
import numpy as np
from typing import Dict, Any


class TechnicalIndicators:
    """
    Calculate technical indicators for trading analysis.
    """

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all indicators and return as dictionary."""
        indicators = {}

        if df.empty:
            return indicators

        close = df['close']
        high = df['high']
        low = df['low']

        # Moving Averages
        indicators['sma_20'] = TechnicalIndicators.sma(close, 20)
        indicators['sma_50'] = TechnicalIndicators.sma(close, 50)
        indicators['sma_200'] = TechnicalIndicators.sma(close, 200)
        indicators['ema_12'] = TechnicalIndicators.ema(close, 12)
        indicators['ema_26'] = TechnicalIndicators.ema(close, 26)

        # RSI
        indicators['rsi'] = TechnicalIndicators.rsi(close)

        # MACD
        macd_data = TechnicalIndicators.macd(close)
        indicators['macd'] = macd_data['macd']
        indicators['macd_signal'] = macd_data['signal']
        indicators['macd_histogram'] = macd_data['histogram']

        # Bollinger Bands
        bb_data = TechnicalIndicators.bollinger_bands(close)
        indicators['bb_upper'] = bb_data['upper']
        indicators['bb_middle'] = bb_data['middle']
        indicators['bb_lower'] = bb_data['lower']

        # ATR
        indicators['atr'] = TechnicalIndicators.atr(high, low, close)

        # Stochastic
        stoch = TechnicalIndicators.stochastic(high, low, close)
        indicators['stoch_k'] = stoch['k']
        indicators['stoch_d'] = stoch['d']

        # ADX
        indicators['adx'] = TechnicalIndicators.adx(high, low, close)

        # Current price
        indicators['current_price'] = float(close.iloc[-1])

        return indicators

    @staticmethod
    def sma(series: pd.Series, period: int) -> float:
        """Simple Moving Average."""
        if len(series) < period:
            return float(series.mean()) if len(series) > 0 else 0.0
        return float(series.rolling(window=period).mean().iloc[-1])

    @staticmethod
    def ema(series: pd.Series, period: int) -> float:
        """Exponential Moving Average."""
        if len(series) < period:
            return float(series.mean()) if len(series) > 0 else 0.0
        return float(series.ewm(span=period, adjust=False).mean().iloc[-1])

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> float:
        """Relative Strength Index."""
        if len(series) < period + 1:
            return 50.0

        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, float]:
        """MACD indicator."""
        if len(series) < slow:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}

        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            'macd': float(macd_line.iloc[-1]),
            'signal': float(signal_line.iloc[-1]),
            'histogram': float(histogram.iloc[-1])
        }

    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, float]:
        """Bollinger Bands."""
        if len(series) < period:
            current = float(series.iloc[-1]) if len(series) > 0 else 0.0
            return {'upper': current, 'middle': current, 'lower': current}

        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return {
            'upper': float(upper.iloc[-1]),
            'middle': float(middle.iloc[-1]),
            'lower': float(lower.iloc[-1])
        }

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> float:
        """Average True Range."""
        if len(close) < 2:
            return 0.0

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0.0

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, float]:
        """Stochastic Oscillator."""
        if len(close) < k_period:
            return {'k': 50.0, 'd': 50.0}

        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()

        return {
            'k': float(k.iloc[-1]) if not np.isnan(k.iloc[-1]) else 50.0,
            'd': float(d.iloc[-1]) if not np.isnan(d.iloc[-1]) else 50.0
        }

    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> float:
        """Average Directional Index."""
        if len(close) < period * 2:
            return 25.0

        # Calculate +DM and -DM
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Directional indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 25.0

    @staticmethod
    def pivot_points(
        high: float,
        low: float,
        close: float
    ) -> Dict[str, float]:
        """Calculate pivot points."""
        pivot = (high + low + close) / 3

        return {
            'pivot': pivot,
            'r1': 2 * pivot - low,
            'r2': pivot + (high - low),
            'r3': high + 2 * (pivot - low),
            's1': 2 * pivot - high,
            's2': pivot - (high - low),
            's3': low - 2 * (high - pivot)
        }
