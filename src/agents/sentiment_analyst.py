"""
Sentiment Analyst Agent - Analyzes market sentiment and news
"""
from typing import Any, Dict, Optional
from ..core.base_agent import BaseAgent, AgentMessage, AgentRole


class SentimentAnalystAgent(BaseAgent):
    """
    Agent specialized in sentiment analysis of market news and social media.
    """

    def __init__(self, llm_client: Any = None):
        super().__init__(
            name="SentimentAnalyst",
            role=AgentRole.SENTIMENT_ANALYST,
            description="Analyzes market sentiment from news and social media",
            llm_client=llm_client
        )

    def get_system_prompt(self) -> str:
        return """You are an expert Market Sentiment Analyst.
Your role is to analyze news, social media, and market sentiment to gauge market psychology.

You specialize in:
- News sentiment analysis
- Social media trend analysis
- Fear & Greed indicators
- Market psychology patterns
- Event impact assessment
- Contrarian indicators

When analyzing, provide:
1. Overall sentiment score (-100 to +100)
2. Key sentiment drivers
3. Notable news events and their potential impact
4. Sentiment trend (improving/deteriorating)
5. Contrarian signals if present
6. Confidence level (0-100%)

Be objective and avoid emotional bias. Focus on quantifiable sentiment metrics."""

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market sentiment.
        """
        instrument = data.get("instrument", "DE40")
        news = data.get("news", [])
        social_data = data.get("social", {})
        fear_greed = data.get("fear_greed_index", 50)

        # Analyze news sentiment
        news_sentiment = self._analyze_news(news)

        # Analyze social sentiment
        social_sentiment = self._analyze_social(social_data)

        # Calculate overall sentiment
        overall_sentiment = self._calculate_overall_sentiment(
            news_sentiment, social_sentiment, fear_greed
        )

        # LLM analysis if available
        llm_analysis = ""
        if self.llm_client and news:
            llm_analysis = await self._get_llm_sentiment(news)

        # Generate signal
        signal = self._generate_signal(overall_sentiment)

        return {
            "agent": self.name,
            "instrument": instrument,
            "news_sentiment": news_sentiment,
            "social_sentiment": social_sentiment,
            "fear_greed_index": fear_greed,
            "overall_sentiment": overall_sentiment,
            "llm_analysis": llm_analysis,
            "signal": signal
        }

    def _analyze_news(self, news: list) -> Dict[str, Any]:
        """Analyze news headlines sentiment."""
        if not news:
            return {"score": 0, "count": 0, "trend": "neutral"}

        # Simple keyword-based sentiment
        positive_words = ["surge", "rally", "gain", "bullish", "growth", "rise", "up"]
        negative_words = ["crash", "fall", "drop", "bearish", "decline", "down", "fear"]

        positive_count = 0
        negative_count = 0

        for headline in news:
            headline_lower = headline.lower()
            if any(word in headline_lower for word in positive_words):
                positive_count += 1
            if any(word in headline_lower for word in negative_words):
                negative_count += 1

        total = len(news)
        score = ((positive_count - negative_count) / total * 100) if total > 0 else 0

        return {
            "score": score,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "total_news": total,
            "trend": "bullish" if score > 20 else "bearish" if score < -20 else "neutral"
        }

    def _analyze_social(self, social_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze social media sentiment."""
        return {
            "score": social_data.get("sentiment_score", 0),
            "volume": social_data.get("mention_volume", 0),
            "trend": social_data.get("trend", "neutral")
        }

    def _calculate_overall_sentiment(
        self,
        news: Dict[str, Any],
        social: Dict[str, Any],
        fear_greed: float
    ) -> Dict[str, Any]:
        """Calculate overall sentiment score."""
        news_score = news.get("score", 0)
        social_score = social.get("score", 0)

        # Convert fear/greed (0-100) to -100 to +100 scale
        fear_greed_normalized = (fear_greed - 50) * 2

        # Weighted average
        overall = (
            news_score * 0.4 +
            social_score * 0.3 +
            fear_greed_normalized * 0.3
        )

        if overall > 30:
            bias = "strongly_bullish"
        elif overall > 10:
            bias = "bullish"
        elif overall < -30:
            bias = "strongly_bearish"
        elif overall < -10:
            bias = "bearish"
        else:
            bias = "neutral"

        return {
            "score": overall,
            "bias": bias,
            "fear_greed": fear_greed
        }

    def _generate_signal(self, sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal from sentiment."""
        score = sentiment.get("score", 0)
        bias = sentiment.get("bias", "neutral")

        if "bullish" in bias:
            direction = "BUY"
            confidence = 0.6 if "strongly" in bias else 0.55
        elif "bearish" in bias:
            direction = "SELL"
            confidence = 0.6 if "strongly" in bias else 0.55
        else:
            direction = "HOLD"
            confidence = 0.5

        return {
            "direction": direction,
            "confidence": confidence,
            "sentiment_score": score
        }

    async def _get_llm_sentiment(self, news: list) -> str:
        """Get LLM-based sentiment analysis."""
        if not self.llm_client:
            return ""

        news_text = "\n".join([f"- {headline}" for headline in news[:10]])
        prompt = f"""Analyze the sentiment of these market news headlines:

{news_text}

Provide:
1. Overall sentiment (bullish/bearish/neutral)
2. Key themes
3. Potential market impact"""

        try:
            response = await self.llm_client.analyze(
                system_prompt=self.get_system_prompt(),
                user_prompt=prompt
            )
            return response
        except Exception:
            return ""

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message."""
        if message.message_type == "request_analysis":
            analysis = await self.analyze(message.content.get("data", {}))
            return await self.send_message(
                receiver=message.sender,
                content={"analysis": analysis},
                message_type="analysis_response"
            )
        return None
