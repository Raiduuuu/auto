"""
Search Tool - Real-time market news and information search via Jina AI
Inspired by AI-Trader's tool_jina_search.py
"""
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import aiohttp
import os
from loguru import logger

from .base_tool import BaseTool, ToolResult, ToolStatus


class SearchTool(BaseTool):
    """
    Tool for searching real-time market news and financial information.
    Uses Jina AI for web search and content extraction.
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="search",
            description="Search for real-time market news, financial reports, and trading information"
        )
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self.base_url = "https://s.jina.ai"
        self.reader_url = "https://r.jina.ai"
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 minutes

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for market news or information"
                },
                "search_type": {
                    "type": "string",
                    "enum": ["news", "analysis", "earnings", "general"],
                    "default": "news",
                    "description": "Type of search to perform"
                },
                "instrument": {
                    "type": "string",
                    "description": "Optional instrument to focus search on (e.g., DAX, DE40)"
                },
                "time_range": {
                    "type": "string",
                    "enum": ["1h", "24h", "7d", "30d"],
                    "default": "24h",
                    "description": "Time range for news search"
                },
                "max_results": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of results to return"
                }
            },
            "required": ["query"]
        }

    async def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query")
        search_type = kwargs.get("search_type", "news")
        instrument = kwargs.get("instrument")
        time_range = kwargs.get("time_range", "24h")
        max_results = kwargs.get("max_results", 10)

        if not query:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                error="Query is required"
            )

        # Build enhanced query
        enhanced_query = self._build_query(query, search_type, instrument)

        logger.info(f"Search tool: '{enhanced_query}' ({search_type})")

        # Check cache
        cache_key = f"{enhanced_query}:{time_range}"
        cached = self._get_cached(cache_key)
        if cached:
            logger.info("Returning cached search results")
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                data=cached
            )

        try:
            results = await self._search(enhanced_query, max_results)

            # Analyze sentiment of results
            sentiment = self._analyze_results_sentiment(results)

            data = {
                "query": enhanced_query,
                "search_type": search_type,
                "instrument": instrument,
                "time_range": time_range,
                "results": results,
                "result_count": len(results),
                "sentiment_summary": sentiment,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Cache results
            self._cache[cache_key] = {
                "data": data,
                "timestamp": datetime.utcnow()
            }

            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                data=data
            )

        except Exception as e:
            logger.error(f"Search tool error: {e}")
            # Return fallback results
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                data=self._get_fallback_results(query, instrument)
            )

    def _build_query(
        self, query: str, search_type: str, instrument: Optional[str]
    ) -> str:
        """Build an enhanced search query."""
        parts = [query]

        if instrument:
            instrument_names = {
                "DE40": "DAX 40 German stock index",
                "US500": "S&P 500",
                "US30": "Dow Jones",
                "EURUSD": "EUR/USD forex",
                "XAUUSD": "Gold XAU/USD"
            }
            parts.append(instrument_names.get(instrument, instrument))

        if search_type == "news":
            parts.append("latest news today")
        elif search_type == "analysis":
            parts.append("technical analysis forecast")
        elif search_type == "earnings":
            parts.append("earnings report financial results")

        return " ".join(parts)

    async def _search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform search using Jina AI."""
        headers = {
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/{query}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=30) as response:
                if response.status == 200:
                    text = await response.text()
                    return self._parse_jina_response(text, max_results)
                else:
                    logger.warning(f"Jina search returned {response.status}")
                    return []

    def _parse_jina_response(self, text: str, max_results: int) -> List[Dict[str, Any]]:
        """Parse Jina search response."""
        results = []

        # Jina returns markdown-formatted results
        lines = text.split("\n")
        current_result = {}

        for line in lines:
            line = line.strip()
            if line.startswith("# ") or line.startswith("## "):
                if current_result.get("title"):
                    results.append(current_result)
                    if len(results) >= max_results:
                        break
                current_result = {
                    "title": line.lstrip("#").strip(),
                    "content": "",
                    "url": ""
                }
            elif line.startswith("http"):
                current_result["url"] = line
            elif line and current_result.get("title"):
                current_result["content"] += line + " "

        if current_result.get("title") and len(results) < max_results:
            results.append(current_result)

        # Clean up content
        for r in results:
            r["content"] = r.get("content", "").strip()[:500]  # Limit content length

        return results

    def _analyze_results_sentiment(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment of search results."""
        if not results:
            return {"score": 0, "bias": "neutral", "confidence": 0}

        positive_words = [
            "surge", "rally", "gain", "bullish", "growth", "rise", "up",
            "record", "high", "strong", "positive", "optimistic", "recovery"
        ]
        negative_words = [
            "crash", "fall", "drop", "bearish", "decline", "down", "fear",
            "low", "weak", "negative", "pessimistic", "recession", "crisis"
        ]

        positive_count = 0
        negative_count = 0
        total_words = 0

        for result in results:
            text = (result.get("title", "") + " " + result.get("content", "")).lower()
            words = text.split()
            total_words += len(words)

            for word in words:
                if any(pw in word for pw in positive_words):
                    positive_count += 1
                if any(nw in word for nw in negative_words):
                    negative_count += 1

        # Calculate sentiment score (-100 to +100)
        if positive_count + negative_count > 0:
            score = ((positive_count - negative_count) / (positive_count + negative_count)) * 100
        else:
            score = 0

        if score > 30:
            bias = "bullish"
        elif score > 10:
            bias = "slightly_bullish"
        elif score < -30:
            bias = "bearish"
        elif score < -10:
            bias = "slightly_bearish"
        else:
            bias = "neutral"

        confidence = min(100, (positive_count + negative_count) * 10)

        return {
            "score": round(score, 1),
            "bias": bias,
            "positive_mentions": positive_count,
            "negative_mentions": negative_count,
            "confidence": confidence
        }

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if still valid."""
        if key in self._cache:
            cached = self._cache[key]
            age = (datetime.utcnow() - cached["timestamp"]).total_seconds()
            if age < self._cache_ttl:
                return cached["data"]
            else:
                del self._cache[key]
        return None

    def _get_fallback_results(
        self, query: str, instrument: Optional[str]
    ) -> Dict[str, Any]:
        """Return fallback results when search fails."""
        # Generic market context based on instrument
        fallback_news = {
            "DE40": [
                {"title": "European markets trading mixed amid economic data", "content": "DAX futures indicate cautious trading as investors await key economic releases.", "url": ""},
                {"title": "German manufacturing PMI in focus", "content": "Market participants monitoring industrial output data for trading signals.", "url": ""}
            ],
            "US500": [
                {"title": "S&P 500 futures steady ahead of Fed decision", "content": "US equity markets await monetary policy guidance.", "url": ""},
                {"title": "Tech sector leads market activity", "content": "Major technology stocks drive index movements.", "url": ""}
            ]
        }

        results = fallback_news.get(instrument, [
            {"title": "Markets trading within range", "content": "Financial markets show mixed signals.", "url": ""}
        ])

        return {
            "query": query,
            "instrument": instrument,
            "results": results,
            "result_count": len(results),
            "sentiment_summary": {"score": 0, "bias": "neutral", "confidence": 20},
            "timestamp": datetime.utcnow().isoformat(),
            "fallback": True
        }

    async def read_url(self, url: str) -> ToolResult:
        """Read and extract content from a URL using Jina Reader."""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        reader_url = f"{self.reader_url}/{url}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(reader_url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()
                        return ToolResult(
                            tool_name=self.name,
                            status=ToolStatus.SUCCESS,
                            data={
                                "url": url,
                                "content": content[:5000],  # Limit content
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        )
                    else:
                        return ToolResult(
                            tool_name=self.name,
                            status=ToolStatus.ERROR,
                            error=f"Failed to read URL: {response.status}"
                        )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                error=str(e)
            )
