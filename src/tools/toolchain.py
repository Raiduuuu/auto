"""
Trading Toolchain - Unified interface for all MCP-style trading tools
Inspired by AI-Trader's pure tool-driven architecture
"""
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from loguru import logger

from .base_tool import BaseTool, ToolResult, ToolStatus
from .trade_tool import TradeTool
from .price_tool import PriceTool
from .search_tool import SearchTool
from .math_tool import MathTool
from .chart_tool import ChartTool


class TradingToolchain:
    """
    Unified toolchain providing standardized access to all trading tools.
    Enables autonomous agent operations through a pure tool-driven interface.
    """

    def __init__(
        self,
        ctrader_client: Any = None,
        market_data_provider: Any = None,
        llm_client: Any = None,
        jina_api_key: Optional[str] = None
    ):
        self.tools: Dict[str, BaseTool] = {}
        self.execution_history: List[Dict[str, Any]] = []

        # Initialize all tools
        self._init_tools(ctrader_client, market_data_provider, llm_client, jina_api_key)

        logger.info(f"TradingToolchain initialized with {len(self.tools)} tools")

    def _init_tools(
        self,
        ctrader_client: Any,
        market_data_provider: Any,
        llm_client: Any,
        jina_api_key: Optional[str]
    ):
        """Initialize all available tools."""
        self.tools = {
            "trade": TradeTool(ctrader_client),
            "get_price": PriceTool(market_data_provider),
            "search": SearchTool(jina_api_key),
            "calculate": MathTool(),
            "analyze_chart": ChartTool(llm_client),
        }

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a specific tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all available tool names."""
        return list(self.tools.keys())

    def get_tool_specs(self) -> List[Dict[str, Any]]:
        """Get all tool specifications for LLM function calling."""
        return [tool.get_tool_spec() for tool in self.tools.values()]

    def get_tools_description(self) -> str:
        """Get human-readable description of all tools."""
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- **{name}**: {tool.description}")
        return "\n".join(descriptions)

    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name with given parameters.

        This is the primary interface for autonomous agents to interact
        with the trading system.
        """
        tool = self.tools.get(tool_name)

        if not tool:
            logger.error(f"Unknown tool: {tool_name}")
            return ToolResult(
                tool_name=tool_name,
                status=ToolStatus.ERROR,
                error=f"Unknown tool: {tool_name}. Available: {self.list_tools()}"
            )

        logger.info(f"Executing tool: {tool_name}")

        try:
            result = await tool(**kwargs)

            # Record execution
            self.execution_history.append({
                "tool": tool_name,
                "params": kwargs,
                "status": result.status.value,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return result

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return ToolResult(
                tool_name=tool_name,
                status=ToolStatus.ERROR,
                error=str(e)
            )

    async def execute_sequence(
        self,
        operations: List[Dict[str, Any]]
    ) -> List[ToolResult]:
        """
        Execute a sequence of tool operations.

        Each operation should have:
            - tool: Tool name
            - params: Tool parameters

        Operations are executed in order, with each result available
        for the next operation.
        """
        results = []
        context = {}  # Shared context between operations

        for op in operations:
            tool_name = op.get("tool")
            params = op.get("params", {})

            # Allow params to reference previous results
            resolved_params = self._resolve_params(params, context)

            result = await self.execute(tool_name, **resolved_params)
            results.append(result)

            # Add result to context for next operations
            context[f"result_{len(results)-1}"] = result.data
            context["last_result"] = result.data

            # Stop on error if specified
            if not result.is_success and op.get("stop_on_error", True):
                logger.warning(f"Stopping sequence due to error in {tool_name}")
                break

        return results

    def _resolve_params(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve parameter references from context."""
        resolved = {}

        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                # Reference to context value
                ref = value[1:]  # Remove $
                parts = ref.split(".")

                ctx_value = context
                for part in parts:
                    if isinstance(ctx_value, dict):
                        ctx_value = ctx_value.get(part)
                    else:
                        ctx_value = None
                        break

                resolved[key] = ctx_value
            else:
                resolved[key] = value

        return resolved

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of tool executions."""
        if not self.execution_history:
            return {"total": 0, "by_tool": {}, "success_rate": 0}

        by_tool = {}
        success_count = 0

        for execution in self.execution_history:
            tool = execution["tool"]
            if tool not in by_tool:
                by_tool[tool] = {"count": 0, "success": 0}
            by_tool[tool]["count"] += 1
            if execution["status"] == "success":
                by_tool[tool]["success"] += 1
                success_count += 1

        return {
            "total": len(self.execution_history),
            "by_tool": by_tool,
            "success_rate": success_count / len(self.execution_history) * 100,
            "last_execution": self.execution_history[-1] if self.execution_history else None
        }

    def reset_history(self):
        """Clear execution history."""
        self.execution_history = []


# Convenience function for quick toolchain setup
def create_toolchain(
    ctrader_client: Any = None,
    market_data_provider: Any = None,
    llm_client: Any = None
) -> TradingToolchain:
    """Create a trading toolchain with optional dependencies."""
    import os
    jina_key = os.getenv("JINA_API_KEY")
    return TradingToolchain(
        ctrader_client=ctrader_client,
        market_data_provider=market_data_provider,
        llm_client=llm_client,
        jina_api_key=jina_key
    )
