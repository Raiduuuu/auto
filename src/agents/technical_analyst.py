"""
Technical Analyst Agent - Analyzes price charts and technical indicators
"""
from typing import Any, Dict, Optional
from ..core.base_agent import BaseAgent, AgentMessage, AgentRole


class TechnicalAnalystAgent(BaseAgent):
    """
    Agent specialized in technical analysis of financial instruments.
    Analyzes price patterns, trends, and technical indicators.
    """

    def __init__(self, llm_client: Any = None):
        super().__init__(
            name="TechnicalAnalyst",
            role=AgentRole.TECHNICAL_ANALYST,
            description="Analyzes price charts, patterns, and technical indicators",
            llm_client=llm_client
        )
        self.indicators = [
            "SMA", "EMA", "RSI", "MACD", "Bollinger Bands",
            "ATR", "Stochastic", "ADX", "Fibonacci"
        ]

    def get_system_prompt(self) -> str:
        return """You are an expert Technical Analyst for financial markets.
Your role is to analyze price charts, patterns, and technical indicators to identify trading opportunities.

You specialize in:
- Trend analysis (uptrend, downtrend, sideways)
- Support and resistance levels
- Chart patterns (head & shoulders, triangles, flags, etc.)
- Technical indicators (RSI, MACD, Moving Averages, Bollinger Bands, etc.)
- Price action and candlestick patterns
- Volume analysis

When analyzing, provide:
1. Current trend direction and strength
2. Key support/resistance levels
3. Indicator readings and their implications
4. Pattern recognition
5. Entry/exit suggestions with reasoning
6. Confidence level (0-100%)

Be precise, objective, and back your analysis with specific indicator values and price levels.
Focus on the DAX40 (DE40) and other major indices/forex pairs."""

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform technical analysis on market data.
        """
        instrument = data.get("instrument", "DE40")
        ohlcv = data.get("ohlcv", {})
        indicators = data.get("indicators", {})

        # Prepare analysis context
        analysis_context = self._prepare_analysis_context(ohlcv, indicators)

        # If LLM client available, use it for deeper analysis
        if self.llm_client:
            llm_analysis = await self._get_llm_analysis(analysis_context)
            analysis_context["llm_analysis"] = llm_analysis

        # Generate signal based on indicators
        signal = self._generate_signal(indicators)

        return {
            "agent": self.name,
            "instrument": instrument,
            "analysis": analysis_context,
            "signal": signal,
            "indicators_used": self.indicators
        }

    def _prepare_analysis_context(
        self,
        ohlcv: Dict[str, Any],
        indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare context for analysis."""
        context = {
            "price_data": {
                "current_price": ohlcv.get("close", [0])[-1] if ohlcv.get("close") else 0,
                "high": max(ohlcv.get("high", [0])) if ohlcv.get("high") else 0,
                "low": min(ohlcv.get("low", [0])) if ohlcv.get("low") else 0,
            },
            "trend": self._analyze_trend(indicators),
            "momentum": self._analyze_momentum(indicators),
            "volatility": self._analyze_volatility(indicators),
            "support_resistance": self._find_support_resistance(ohlcv)
        }
        return context

    def _analyze_trend(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trend using moving averages."""
        sma_20 = indicators.get("sma_20", 0)
        sma_50 = indicators.get("sma_50", 0)
        sma_200 = indicators.get("sma_200", 0)
        current_price = indicators.get("current_price", 0)

        if current_price > sma_20 > sma_50 > sma_200:
            trend = "strong_uptrend"
            strength = 0.9
        elif current_price > sma_20 > sma_50:
            trend = "uptrend"
            strength = 0.7
        elif current_price < sma_20 < sma_50 < sma_200:
            trend = "strong_downtrend"
            strength = 0.9
        elif current_price < sma_20 < sma_50:
            trend = "downtrend"
            strength = 0.7
        else:
            trend = "sideways"
            strength = 0.5

        return {"direction": trend, "strength": strength}

    def _analyze_momentum(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum using RSI and MACD."""
        rsi = indicators.get("rsi", 50)
        macd = indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", 0)

        # RSI analysis
        if rsi > 70:
            rsi_signal = "overbought"
        elif rsi < 30:
            rsi_signal = "oversold"
        else:
            rsi_signal = "neutral"

        # MACD analysis
        if macd > macd_signal:
            macd_signal_type = "bullish"
        elif macd < macd_signal:
            macd_signal_type = "bearish"
        else:
            macd_signal_type = "neutral"

        return {
            "rsi": rsi,
            "rsi_signal": rsi_signal,
            "macd": macd,
            "macd_signal_type": macd_signal_type
        }

    def _analyze_volatility(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility using ATR and Bollinger Bands."""
        atr = indicators.get("atr", 0)
        bb_upper = indicators.get("bb_upper", 0)
        bb_lower = indicators.get("bb_lower", 0)
        bb_middle = indicators.get("bb_middle", 0)

        return {
            "atr": atr,
            "bollinger_bands": {
                "upper": bb_upper,
                "middle": bb_middle,
                "lower": bb_lower
            }
        }

    def _find_support_resistance(self, ohlcv: Dict[str, Any]) -> Dict[str, Any]:
        """Find support and resistance levels."""
        highs = ohlcv.get("high", [])
        lows = ohlcv.get("low", [])

        if not highs or not lows:
            return {"support": [], "resistance": []}

        # Simple pivot points
        resistance = sorted(set(highs), reverse=True)[:3]
        support = sorted(set(lows))[:3]

        return {
            "support": support,
            "resistance": resistance
        }

    def _generate_signal(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal based on indicators."""
        trend = self._analyze_trend(indicators)
        momentum = self._analyze_momentum(indicators)

        # Scoring system
        score = 0

        # Trend contribution
        if "uptrend" in trend["direction"]:
            score += 2 if "strong" in trend["direction"] else 1
        elif "downtrend" in trend["direction"]:
            score -= 2 if "strong" in trend["direction"] else 1

        # Momentum contribution
        if momentum["rsi_signal"] == "oversold":
            score += 1
        elif momentum["rsi_signal"] == "overbought":
            score -= 1

        if momentum["macd_signal_type"] == "bullish":
            score += 1
        elif momentum["macd_signal_type"] == "bearish":
            score -= 1

        # Determine direction
        if score >= 2:
            direction = "BUY"
            confidence = min(0.9, 0.5 + score * 0.1)
        elif score <= -2:
            direction = "SELL"
            confidence = min(0.9, 0.5 + abs(score) * 0.1)
        else:
            direction = "HOLD"
            confidence = 0.5

        return {
            "direction": direction,
            "confidence": confidence,
            "score": score
        }

    async def _get_llm_analysis(self, context: Dict[str, Any]) -> str:
        """Get LLM-based analysis."""
        if not self.llm_client:
            return ""

        prompt = f"""Analyze the following technical data and provide trading insights:

{context}

Provide a concise technical analysis summary with:
1. Overall market bias
2. Key levels to watch
3. Recommended action"""

        try:
            response = await self.llm_client.analyze(
                system_prompt=self.get_system_prompt(),
                user_prompt=prompt
            )
            return response
        except Exception as e:
            return f"LLM analysis unavailable: {e}"

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message from another agent."""
        if message.message_type == "request_analysis":
            data = message.content.get("data", {})
            analysis = await self.analyze(data)
            return await self.send_message(
                receiver=message.sender,
                content={"analysis": analysis},
                message_type="analysis_response"
            )
        return None
