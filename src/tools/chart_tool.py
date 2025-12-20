"""
Chart Tool - Analyze chart images using vision AI
"""
from typing import Any, Dict, Optional
import base64
import os
from pathlib import Path
from loguru import logger

from .base_tool import BaseTool, ToolResult, ToolStatus


class ChartTool(BaseTool):
    """
    Tool for analyzing trading chart images using Claude's vision capabilities.
    Provides AI-powered technical analysis from chart screenshots.
    """

    def __init__(self, llm_client: Any = None):
        super().__init__(
            name="analyze_chart",
            description="Analyze a trading chart image to identify patterns, trends, and trading signals"
        )
        self.llm_client = llm_client

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the chart image file"
                },
                "image_base64": {
                    "type": "string",
                    "description": "Base64 encoded image data"
                },
                "instrument": {
                    "type": "string",
                    "description": "Instrument being analyzed (e.g., DE40, EURUSD)"
                },
                "timeframe": {
                    "type": "string",
                    "description": "Chart timeframe (e.g., M5, H1, D1)"
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["full", "patterns", "levels", "trend", "signals"],
                    "default": "full",
                    "description": "Type of analysis to perform"
                }
            }
        }

    async def execute(self, **kwargs) -> ToolResult:
        image_path = kwargs.get("image_path")
        image_base64 = kwargs.get("image_base64")
        instrument = kwargs.get("instrument", "Unknown")
        timeframe = kwargs.get("timeframe", "Unknown")
        analysis_type = kwargs.get("analysis_type", "full")

        logger.info(f"Chart tool: analyzing {instrument} {timeframe}")

        # Get image data
        if image_path:
            image_data = self._load_image(image_path)
            if not image_data:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolStatus.ERROR,
                    error=f"Could not load image: {image_path}"
                )
        elif image_base64:
            image_data = image_base64
        else:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                error="Either image_path or image_base64 required"
            )

        if not self.llm_client:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                error="LLM client required for chart analysis"
            )

        try:
            # Analyze chart with vision
            analysis = await self._analyze_chart_image(
                image_data, instrument, timeframe, analysis_type
            )

            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                data={
                    "instrument": instrument,
                    "timeframe": timeframe,
                    "analysis_type": analysis_type,
                    "analysis": analysis
                }
            )

        except Exception as e:
            logger.error(f"Chart analysis error: {e}")
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                error=str(e)
            )

    def _load_image(self, path: str) -> Optional[str]:
        """Load and encode image to base64."""
        try:
            file_path = Path(path)
            if not file_path.exists():
                return None

            with open(file_path, "rb") as f:
                image_bytes = f.read()

            return base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None

    def _get_media_type(self, path: str) -> str:
        """Get media type from file extension."""
        ext = Path(path).suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        return media_types.get(ext, "image/png")

    async def _analyze_chart_image(
        self,
        image_data: str,
        instrument: str,
        timeframe: str,
        analysis_type: str
    ) -> Dict[str, Any]:
        """Analyze chart using Claude vision."""

        system_prompt = """You are an expert technical analyst. Analyze trading charts with precision.

You MUST respond with valid JSON only. No markdown, no extra text.

For each analysis, identify:
1. Current trend (uptrend/downtrend/sideways) and strength
2. Key support and resistance levels
3. Chart patterns (triangles, channels, head & shoulders, etc.)
4. Candlestick patterns
5. Potential entry, stop loss, and take profit levels
6. Overall trading signal (BUY/SELL/HOLD) with confidence"""

        analysis_prompts = {
            "full": f"""Analyze this {instrument} {timeframe} chart completely.

Return JSON:
{{
    "trend": {{"direction": "uptrend|downtrend|sideways", "strength": "strong|moderate|weak"}},
    "support_levels": [price1, price2],
    "resistance_levels": [price1, price2],
    "patterns": [{{"name": "pattern_name", "implication": "bullish|bearish"}}],
    "candlestick_patterns": ["pattern1", "pattern2"],
    "signal": {{"direction": "BUY|SELL|HOLD", "confidence": 0.0-1.0}},
    "entry_price": price,
    "stop_loss": price,
    "take_profit": price,
    "key_observations": ["observation1", "observation2"],
    "risk_level": "low|medium|high"
}}""",

            "patterns": f"""Identify chart patterns in this {instrument} {timeframe} chart.

Return JSON:
{{
    "chart_patterns": [{{"name": "", "status": "forming|complete", "target": price, "implication": "bullish|bearish"}}],
    "candlestick_patterns": [{{"name": "", "location": "support|resistance|middle", "significance": "strong|moderate|weak"}}]
}}""",

            "levels": f"""Identify key price levels in this {instrument} {timeframe} chart.

Return JSON:
{{
    "support_levels": [{{"price": 0, "strength": "strong|moderate|weak", "touches": 0}}],
    "resistance_levels": [{{"price": 0, "strength": "strong|moderate|weak", "touches": 0}}],
    "pivot_point": price,
    "current_position": "near_support|near_resistance|middle"
}}""",

            "trend": f"""Analyze the trend in this {instrument} {timeframe} chart.

Return JSON:
{{
    "primary_trend": {{"direction": "", "strength": "", "duration": "short|medium|long"}},
    "secondary_trend": {{"direction": "", "strength": ""}},
    "trend_line_broken": true|false,
    "momentum": "increasing|decreasing|neutral",
    "trend_exhaustion_signs": true|false
}}""",

            "signals": f"""Provide trading signals from this {instrument} {timeframe} chart.

Return JSON:
{{
    "signal": {{"direction": "BUY|SELL|HOLD", "confidence": 0.0-1.0, "reasoning": ""}},
    "entry": {{"price": 0, "type": "market|limit|stop"}},
    "stop_loss": {{"price": 0, "reasoning": ""}},
    "take_profit": [{{"price": 0, "reasoning": ""}}],
    "risk_reward_ratio": 0.0,
    "trade_quality": "A|B|C"
}}"""
        }

        prompt = analysis_prompts.get(analysis_type, analysis_prompts["full"])

        try:
            # Call LLM with vision
            response = await self.llm_client.analyze_with_image(
                system_prompt=system_prompt,
                user_prompt=prompt,
                image_data=image_data,
                media_type="image/png"
            )

            # Parse JSON response
            import json

            # Try to extract JSON from response
            response_text = response.strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            analysis = json.loads(response_text)
            return analysis

        except json.JSONDecodeError:
            # If JSON parsing fails, return structured error with raw response
            return {
                "error": "Could not parse analysis as JSON",
                "raw_response": response[:500] if response else None,
                "signal": {"direction": "HOLD", "confidence": 0.0}
            }
        except Exception as e:
            return {
                "error": str(e),
                "signal": {"direction": "HOLD", "confidence": 0.0}
            }
