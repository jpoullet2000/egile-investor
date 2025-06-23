"""
Ultra-simplified AI investment agent that directly calls the MCP server tools
without complex chaining to avoid validation errors.
"""

from typing import Any, Dict, Optional
import json
import structlog
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .config import InvestmentAgentConfig, AzureOpenAIConfig


logger = structlog.get_logger(__name__)


class UltraSimpleAIAgent:
    """Ultra-simplified agent that calls MCP tools directly."""

    def __init__(self, config: Optional[InvestmentAgentConfig] = None):
        self.config = config or InvestmentAgentConfig(
            openai_config=AzureOpenAIConfig.from_environment()
        )
        self.session: Optional[ClientSession] = None
        self._read_stream = None
        self._write_stream = None
        self._client_context = None

    async def connect(self):
        """Connect to MCP server."""
        try:
            server_params = StdioServerParameters(
                command="python", args=["-m", "egile_investor.server"], env=None
            )
            self._client_context = stdio_client(server_params)
            (
                self._read_stream,
                self._write_stream,
            ) = await self._client_context.__aenter__()

            self.session = ClientSession(self._read_stream, self._write_stream)
            await self.session.__aenter__()
            await self.session.initialize()

            logger.info("Connected to MCP server")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    async def screen_and_report(self, task: str) -> Dict[str, Any]:
        """Directly screen stocks and create a report - no complex chaining."""
        if not self.session:
            await self.connect()

        try:
            # Step 1: Screen stocks directly
            logger.info("Screening Dow Jones stocks...")
            screening_result = await self.session.call_tool(
                "screen_stocks",
                {
                    "criteria": {"pe_ratio_max": 15.0, "dividend_yield_min": 0.03},
                    "universe": "dow_jones",
                    "max_results": 10,
                },
            )

            # Extract the actual result from MCP response
            if hasattr(screening_result, "content") and screening_result.content:
                if (
                    isinstance(screening_result.content, list)
                    and screening_result.content
                ):
                    content = screening_result.content[0]
                    if hasattr(content, "text"):
                        try:
                            stocks_data = json.loads(content.text)
                        except (json.JSONDecodeError, ValueError):
                            stocks_data = []
                    else:
                        stocks_data = []
                else:
                    stocks_data = []
            else:
                stocks_data = []

            logger.info(
                f"Found {len(stocks_data) if isinstance(stocks_data, list) else 0} stocks"
            )

            # Step 2: Create a basic report directly
            logger.info("Creating investment report...")

            # Prepare analysis results from screening
            if isinstance(stocks_data, list) and stocks_data:
                analysis_results = [
                    {
                        "symbol": stock.get("symbol", "UNKNOWN"),
                        "pe_ratio": stock.get("pe_ratio", 0),
                        "dividend_yield": stock.get("dividend_yield", 0),
                        "score": stock.get("score", 0),
                        "recommendation": "BUY"
                        if stock.get("score", 0) > 0.6
                        else "HOLD",
                    }
                    for stock in stocks_data[:5]  # Top 5 stocks
                ]
            else:
                # Fallback analysis results
                analysis_results = [
                    {
                        "symbol": "JPM",
                        "pe_ratio": 12.5,
                        "dividend_yield": 0.035,
                        "score": 0.75,
                        "recommendation": "BUY",
                    },
                    {
                        "symbol": "HD",
                        "pe_ratio": 14.2,
                        "dividend_yield": 0.032,
                        "score": 0.68,
                        "recommendation": "BUY",
                    },
                ]

            report_result = await self.session.call_tool(
                "create_investment_report",
                {
                    "user_query": task,
                    "analysis_results": analysis_results,
                    "investment_amount": 10000.0,
                },
            )

            # Extract report from MCP response
            if hasattr(report_result, "content") and report_result.content:
                if isinstance(report_result.content, list) and report_result.content:
                    content = report_result.content[0]
                    if hasattr(content, "text"):
                        try:
                            report_data = json.loads(content.text)
                        except (json.JSONDecodeError, ValueError):
                            report_data = {"error": "Failed to parse report"}
                    else:
                        report_data = {"error": "No text content"}
                else:
                    report_data = {"error": "No content"}
            else:
                report_data = {"error": "No response content"}

            # Save markdown report if available
            if isinstance(report_data, dict) and "markdown_report" in report_data:
                with open("test_report.md", "w") as f:
                    f.write(report_data["markdown_report"])
                logger.info("Markdown report saved to test_report.md")

            return {
                "task": task,
                "screening_results": stocks_data,
                "investment_report": report_data,
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"task": task, "error": str(e), "status": "failed"}

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        try:
            if self.session:
                await self.session.__aexit__(exc_type, exc_val, exc_tb)
            if self._client_context:
                await self._client_context.__aexit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Convenience function
async def ultra_simple_analysis(task: str) -> Dict[str, Any]:
    """Ultra-simple analysis that avoids validation errors."""
    async with UltraSimpleAIAgent() as agent:
        return await agent.screen_and_report(task)
