"""
Autonomous Trading Agent - Pure tool-driven trading agent
Inspired by AI-Trader's zero-preset-strategy approach
"""
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
import json
from loguru import logger

from ..core.base_agent import BaseAgent, AgentRole, TradingSignal
from ..tools.toolchain import TradingToolchain
from ..tools.base_tool import ToolResult


@dataclass
class AgentThought:
    """Represents a step in the agent's reasoning chain."""
    step: int
    thought: str
    action: Optional[str] = None
    tool: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class TradingPlan:
    """Agent's trading plan derived from analysis."""
    instrument: str
    direction: str  # BUY, SELL, HOLD
    confidence: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    reasoning: str = ""
    risk_reward: Optional[float] = None


class AutonomousAgent(BaseAgent):
    """
    Autonomous trading agent that operates purely through tool calls.

    Key principles (from AI-Trader):
    - No preset trading strategies or algorithmic rules
    - Complete reliance on inherent AI reasoning capabilities
    - All operations executed through standardized tool calls
    - Full trading rationale documentation
    """

    def __init__(
        self,
        name: str = "AutonomousTrader",
        llm_client: Any = None,
        toolchain: Optional[TradingToolchain] = None,
        model_name: str = "claude"
    ):
        super().__init__(
            name=name,
            role=AgentRole.TRADER,
            description="Autonomous AI trading agent with pure tool-driven decision making",
            llm_client=llm_client
        )
        self.toolchain = toolchain
        self.model_name = model_name
        self.thought_chain: List[AgentThought] = []
        self.trading_history: List[Dict[str, Any]] = []
        self.max_reasoning_steps = 10

    def get_system_prompt(self) -> str:
        tools_desc = self.toolchain.get_tools_description() if self.toolchain else "No tools available"

        return f"""You are an autonomous AI trading agent for DAX40 (DE40) and other CFD instruments.

## Your Mission
Make profitable trading decisions by independently analyzing markets and executing trades.
You have complete autonomy - NO preset strategies, NO hard-coded rules.
Your decisions are based purely on your own analysis and reasoning.

## Available Tools
{tools_desc}

## Trading Guidelines
1. Always gather data before making decisions (use get_price, search tools)
2. Consider multiple timeframes (M5, H1, D1)
3. Calculate proper position sizing based on risk (use calculate tool)
4. Set stop loss and take profit for every trade
5. Document your reasoning for every decision

## Risk Management Rules
- Maximum risk per trade: 1-2% of account
- Minimum risk/reward ratio: 1:1.5
- Always use stop loss orders
- Consider current market volatility

## Response Format
You must respond with a JSON object containing your analysis and actions.

For analysis phase:
{{
    "phase": "analysis",
    "thoughts": ["thought1", "thought2"],
    "tool_calls": [
        {{"tool": "tool_name", "params": {{...}}}}
    ]
}}

For decision phase:
{{
    "phase": "decision",
    "analysis_summary": "...",
    "market_view": "bullish|bearish|neutral",
    "trade_plan": {{
        "action": "BUY|SELL|HOLD",
        "instrument": "DE40",
        "confidence": 0.0-1.0,
        "entry_price": null or price,
        "stop_loss": price,
        "take_profit": price,
        "position_size": lots,
        "reasoning": "..."
    }}
}}

For execution phase:
{{
    "phase": "execution",
    "tool_calls": [
        {{"tool": "trade", "params": {{...}}}}
    ]
}}

Be decisive but prudent. Quality trades over quantity."""

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run autonomous analysis and generate trading decision.

        This is the main entry point for the agent's decision loop.
        """
        instrument = data.get("instrument", "DE40")
        timeframe = data.get("timeframe", "M5")
        account_balance = data.get("account_balance", 10000)

        logger.info(f"Autonomous agent starting analysis for {instrument}")

        self.thought_chain = []
        analysis_results = {}

        try:
            # Phase 1: Data Gathering
            logger.info("Phase 1: Gathering market data")
            market_data = await self._gather_market_data(instrument, timeframe)
            analysis_results["market_data"] = market_data

            # Phase 2: Information Search
            logger.info("Phase 2: Searching for market news")
            news_data = await self._search_market_news(instrument)
            analysis_results["news"] = news_data

            # Phase 3: LLM Analysis
            logger.info("Phase 3: AI analysis and reasoning")
            ai_analysis = await self._run_ai_analysis(
                instrument, timeframe, market_data, news_data, account_balance
            )
            analysis_results["ai_analysis"] = ai_analysis

            # Phase 4: Generate Trading Plan
            logger.info("Phase 4: Generating trading plan")
            trade_plan = self._extract_trade_plan(ai_analysis, instrument)
            analysis_results["trade_plan"] = trade_plan

            # Generate signal
            signal = self._create_signal(trade_plan, instrument)

            return {
                "agent": self.name,
                "model": self.model_name,
                "instrument": instrument,
                "timeframe": timeframe,
                "analysis": analysis_results,
                "signal": signal,
                "thought_chain": [t.__dict__ for t in self.thought_chain],
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Autonomous agent error: {e}")
            return {
                "agent": self.name,
                "error": str(e),
                "signal": {"direction": "HOLD", "confidence": 0.0}
            }

    async def _gather_market_data(
        self, instrument: str, timeframe: str
    ) -> Dict[str, Any]:
        """Gather price data and indicators using tools."""
        self._add_thought("Gathering market data and technical indicators")

        if not self.toolchain:
            return {"error": "No toolchain available"}

        # Get current price
        quote_result = await self.toolchain.execute(
            "get_price",
            instrument=instrument,
            data_type="quote"
        )

        # Get OHLCV with indicators
        indicators_result = await self.toolchain.execute(
            "get_price",
            instrument=instrument,
            data_type="indicators",
            timeframe=timeframe,
            periods=100,
            indicators=["sma_20", "sma_50", "rsi", "macd", "atr", "bbands", "adx"]
        )

        # Get higher timeframe for context
        htf = "H1" if timeframe in ["M1", "M5", "M15"] else "D1"
        htf_result = await self.toolchain.execute(
            "get_price",
            instrument=instrument,
            data_type="indicators",
            timeframe=htf,
            periods=50,
            indicators=["sma_20", "sma_50", "rsi", "macd"]
        )

        self._add_thought(
            f"Retrieved quote, {timeframe} data with indicators, and {htf} context",
            observation=f"Current price: {quote_result.data.get('bid', 'N/A')}"
        )

        return {
            "quote": quote_result.data if quote_result.is_success else {},
            "primary_tf": indicators_result.data if indicators_result.is_success else {},
            "higher_tf": htf_result.data if htf_result.is_success else {}
        }

    async def _search_market_news(self, instrument: str) -> Dict[str, Any]:
        """Search for relevant market news."""
        self._add_thought(f"Searching for {instrument} market news and sentiment")

        if not self.toolchain:
            return {"error": "No toolchain available"}

        # Search for news
        news_result = await self.toolchain.execute(
            "search",
            query=f"{instrument} market news trading",
            search_type="news",
            instrument=instrument,
            time_range="24h",
            max_results=5
        )

        # Search for analysis
        analysis_result = await self.toolchain.execute(
            "search",
            query=f"{instrument} technical analysis forecast",
            search_type="analysis",
            instrument=instrument,
            max_results=3
        )

        sentiment = news_result.data.get("sentiment_summary", {}) if news_result.is_success else {}
        self._add_thought(
            "Analyzed market news and sentiment",
            observation=f"Sentiment: {sentiment.get('bias', 'unknown')}"
        )

        return {
            "news": news_result.data if news_result.is_success else {},
            "analysis": analysis_result.data if analysis_result.is_success else {},
            "combined_sentiment": sentiment
        }

    async def _run_ai_analysis(
        self,
        instrument: str,
        timeframe: str,
        market_data: Dict[str, Any],
        news_data: Dict[str, Any],
        account_balance: float
    ) -> Dict[str, Any]:
        """Run AI-powered analysis using LLM."""
        if not self.llm_client:
            return self._fallback_analysis(market_data, news_data)

        # Prepare context for LLM
        context = self._prepare_analysis_context(
            instrument, timeframe, market_data, news_data, account_balance
        )

        prompt = f"""Analyze this market data and provide a trading decision.

## Market Context
{context}

## Your Task
1. Analyze the technical indicators
2. Consider the news sentiment
3. Evaluate risk/reward
4. Make a trading decision

Respond with JSON containing your analysis and trade_plan."""

        try:
            response = await self.llm_client.analyze(
                system_prompt=self.get_system_prompt(),
                user_prompt=prompt
            )

            # Parse LLM response
            return self._parse_llm_response(response)

        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return self._fallback_analysis(market_data, news_data)

    def _prepare_analysis_context(
        self,
        instrument: str,
        timeframe: str,
        market_data: Dict[str, Any],
        news_data: Dict[str, Any],
        account_balance: float
    ) -> str:
        """Prepare context string for LLM analysis."""
        quote = market_data.get("quote", {})
        primary = market_data.get("primary_tf", {})
        higher = market_data.get("higher_tf", {})
        indicators = primary.get("indicators", {})
        htf_indicators = higher.get("indicators", {})

        context = f"""
Instrument: {instrument}
Timeframe: {timeframe}
Account Balance: ${account_balance:,.2f}

## Current Price
Bid: {quote.get('bid', 'N/A')}
Ask: {quote.get('ask', 'N/A')}

## Technical Indicators ({timeframe})
- SMA 20: {indicators.get('sma_20', 'N/A')}
- SMA 50: {indicators.get('sma_50', 'N/A')}
- RSI: {indicators.get('rsi', 'N/A')}
- MACD: {indicators.get('macd', {})}
- ATR: {indicators.get('atr', 'N/A')}
- Bollinger Bands: {indicators.get('bbands', {})}
- ADX: {indicators.get('adx', 'N/A')}

## Higher Timeframe Context
- SMA 20: {htf_indicators.get('sma_20', 'N/A')}
- SMA 50: {htf_indicators.get('sma_50', 'N/A')}
- RSI: {htf_indicators.get('rsi', 'N/A')}
- MACD: {htf_indicators.get('macd', {})}

## News Sentiment
{json.dumps(news_data.get('combined_sentiment', {}), indent=2)}

## Recent Headlines
"""
        news_items = news_data.get("news", {}).get("results", [])[:3]
        for item in news_items:
            context += f"- {item.get('title', 'N/A')}\n"

        return context

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract analysis."""
        try:
            # Try to extract JSON from response
            response = response.strip()

            # Remove markdown code blocks
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            # Find JSON object
            start = response.find("{")
            end = response.rfind("}") + 1

            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)

        except json.JSONDecodeError:
            pass

        # Return raw response if parsing fails
        return {
            "raw_response": response[:1000],
            "parse_error": True
        }

    def _fallback_analysis(
        self,
        market_data: Dict[str, Any],
        news_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate analysis without LLM using rule-based approach."""
        self._add_thought("Using fallback rule-based analysis")

        indicators = market_data.get("primary_tf", {}).get("indicators", {})
        sentiment = news_data.get("combined_sentiment", {})

        # Simple technical scoring
        score = 0

        # RSI analysis
        rsi = indicators.get("rsi")
        if rsi:
            if rsi < 30:
                score += 2  # Oversold - bullish
            elif rsi > 70:
                score -= 2  # Overbought - bearish
            elif rsi < 50:
                score -= 0.5
            else:
                score += 0.5

        # MACD analysis
        macd = indicators.get("macd", {})
        if macd:
            hist = macd.get("histogram", 0)
            if hist and hist > 0:
                score += 1
            elif hist and hist < 0:
                score -= 1

        # Moving average analysis
        sma20 = indicators.get("sma_20")
        sma50 = indicators.get("sma_50")
        if sma20 and sma50:
            if sma20 > sma50:
                score += 1  # Bullish crossover
            else:
                score -= 1

        # Sentiment adjustment
        sent_score = sentiment.get("score", 0)
        score += sent_score / 50  # Normalize sentiment contribution

        # Determine direction
        if score > 1.5:
            direction = "BUY"
            confidence = min(0.75, 0.5 + score * 0.05)
        elif score < -1.5:
            direction = "SELL"
            confidence = min(0.75, 0.5 + abs(score) * 0.05)
        else:
            direction = "HOLD"
            confidence = 0.5

        return {
            "analysis_type": "fallback_rules",
            "technical_score": score,
            "trade_plan": {
                "action": direction,
                "confidence": confidence,
                "reasoning": f"Technical score: {score:.2f}, Sentiment: {sent_score:.1f}"
            }
        }

    def _extract_trade_plan(
        self, ai_analysis: Dict[str, Any], instrument: str
    ) -> Optional[TradingPlan]:
        """Extract trading plan from AI analysis."""
        plan_data = ai_analysis.get("trade_plan", {})

        if not plan_data:
            return None

        return TradingPlan(
            instrument=instrument,
            direction=plan_data.get("action", "HOLD"),
            confidence=plan_data.get("confidence", 0.5),
            entry_price=plan_data.get("entry_price"),
            stop_loss=plan_data.get("stop_loss"),
            take_profit=plan_data.get("take_profit"),
            position_size=plan_data.get("position_size"),
            reasoning=plan_data.get("reasoning", ""),
            risk_reward=plan_data.get("risk_reward")
        )

    def _create_signal(
        self, trade_plan: Optional[TradingPlan], instrument: str
    ) -> Dict[str, Any]:
        """Create trading signal from plan."""
        if not trade_plan:
            return {
                "direction": "HOLD",
                "confidence": 0.0,
                "reason": "No valid trading plan generated"
            }

        return {
            "direction": trade_plan.direction,
            "confidence": trade_plan.confidence,
            "entry_price": trade_plan.entry_price,
            "stop_loss": trade_plan.stop_loss,
            "take_profit": trade_plan.take_profit,
            "position_size": trade_plan.position_size,
            "reasoning": trade_plan.reasoning,
            "risk_reward": trade_plan.risk_reward
        }

    def _add_thought(
        self,
        thought: str,
        action: Optional[str] = None,
        tool: Optional[str] = None,
        tool_params: Optional[Dict[str, Any]] = None,
        observation: Optional[str] = None
    ):
        """Add a thought to the reasoning chain."""
        step = len(self.thought_chain) + 1
        self.thought_chain.append(AgentThought(
            step=step,
            thought=thought,
            action=action,
            tool=tool,
            tool_params=tool_params,
            observation=observation
        ))

    async def execute_trade(self, trade_plan: TradingPlan) -> ToolResult:
        """Execute a trade based on the plan."""
        if not self.toolchain:
            from ..tools.base_tool import ToolResult, ToolStatus
            return ToolResult(
                tool_name="trade",
                status=ToolStatus.ERROR,
                error="No toolchain available"
            )

        if trade_plan.direction == "HOLD":
            from ..tools.base_tool import ToolResult, ToolStatus
            return ToolResult(
                tool_name="trade",
                status=ToolStatus.SUCCESS,
                data={"action": "HOLD", "message": "No trade executed"}
            )

        result = await self.toolchain.execute(
            "trade",
            action=trade_plan.direction.lower(),
            instrument=trade_plan.instrument,
            volume=trade_plan.position_size or 0.1,
            order_type="market",
            stop_loss=trade_plan.stop_loss,
            take_profit=trade_plan.take_profit,
            reason=trade_plan.reasoning
        )

        # Record trade
        self.trading_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "plan": trade_plan.__dict__,
            "result": result.to_dict()
        })

        return result

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get agent performance summary."""
        if not self.trading_history:
            return {"trades": 0, "message": "No trades executed yet"}

        trades = len(self.trading_history)
        successful = sum(
            1 for t in self.trading_history
            if t["result"]["status"] == "success"
        )

        return {
            "total_trades": trades,
            "successful_executions": successful,
            "execution_rate": successful / trades * 100 if trades > 0 else 0,
            "model": self.model_name,
            "last_trade": self.trading_history[-1] if self.trading_history else None
        }
