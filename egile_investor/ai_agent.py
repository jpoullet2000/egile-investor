"""
AI-Powered Investment Agent that uses LLM reasoning to plan and execute MCP tool usage.

This agent uses an LLM to intelligently plan which tools to use and in what sequence
to accomplish complex investment analysis tasks.
"""

from typing import Any, Dict, List, Optional

import structlog
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Tool

from .config import InvestmentAgentConfig, AzureOpenAIConfig
from .client import AzureOpenAIClient


logger = structlog.get_logger(__name__)


class AIInvestmentAgent:
    """
    An AI-powered investment agent that uses LLM reasoning to plan and execute MCP tool usage.
    """

    def __init__(
        self,
        config: Optional[InvestmentAgentConfig] = None,
        server_command: Optional[str] = None,
    ):
        """
        Initialize the AI investment agent.

        Args:
            config: Investment agent configuration
            server_command: Command to start the MCP server
        """
        self.config = config or InvestmentAgentConfig(
            openai_config=AzureOpenAIConfig.from_environment()
        )
        self.server_command = server_command or "python -m egile_investor.server"
        self.session: Optional[ClientSession] = None
        self.available_tools: Dict[str, Tool] = {}
        self._read_stream = None
        self._write_stream = None
        self._client_context = None

        # Initialize OpenAI client using the same approach as the main agent
        self.openai_client = AzureOpenAIClient(self.config.openai_config)

    async def connect(self):
        """Connect to the MCP server and discover available tools."""
        try:
            # Parse the server command into command and args
            command_parts = self.server_command.split()
            command = command_parts[0]
            args = command_parts[1:] if len(command_parts) > 1 else []

            # Create server parameters
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=None,
            )

            # Connect to server and keep the connection alive
            self._client_context = stdio_client(server_params)
            (
                self._read_stream,
                self._write_stream,
            ) = await self._client_context.__aenter__()

            # Create session and keep it alive
            self.session = ClientSession(self._read_stream, self._write_stream)
            await self.session.__aenter__()

            # Initialize the session
            await self.session.initialize()

            # Discover available tools
            tools_response = await self.session.list_tools()
            self.available_tools = {tool.name: tool for tool in tools_response.tools}

            logger.info(
                f"Connected to MCP server with {len(self.available_tools)} tools"
            )
            for tool_name in self.available_tools.keys():
                logger.info(f"Available tool: {tool_name}")

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a specific MCP tool.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        if tool_name not in self.available_tools:
            raise ValueError(f"Tool '{tool_name}' not available")

        try:
            response = await self.session.call_tool(tool_name, arguments)

            logger.info(f"Called tool '{tool_name}' with arguments: {arguments}")

            # Handle MCP response format - extract text from content
            if hasattr(response, "content") and response.content:
                # MCP returns a list of content items
                if isinstance(response.content, list) and len(response.content) > 0:
                    # Get the first text content item
                    first_content = response.content[0]
                    if hasattr(first_content, "text"):
                        try:
                            # Try to parse JSON if it looks like JSON
                            import json

                            return json.loads(first_content.text)
                        except (json.JSONDecodeError, ValueError):
                            # Return as string if not JSON
                            return first_content.text
                    else:
                        return first_content
                else:
                    return response.content
            else:
                return response

        except Exception as e:
            logger.error(f"Failed to call tool '{tool_name}': {e}")
            raise

    def _format_tools_for_prompt(self) -> str:
        """Format available tools for the LLM prompt."""
        tool_descriptions = []
        for name, tool in self.available_tools.items():
            description = tool.description or "No description available"

            # Format input schema if available
            input_schema = ""
            if hasattr(tool, "inputSchema") and tool.inputSchema:
                properties = tool.inputSchema.get("properties", {})
                if properties:
                    params = []
                    for param_name, param_info in properties.items():
                        param_type = param_info.get("type", "any")
                        param_desc = param_info.get("description", "")
                        params.append(f"  - {param_name} ({param_type}): {param_desc}")
                    input_schema = "\n  Parameters:\n" + "\n".join(params)

            tool_descriptions.append(f"- {name}: {description}{input_schema}")

        return "\n".join(tool_descriptions)

    async def _create_ai_plan(self, task: str) -> List[Dict[str, Any]]:
        """
        Use LLM to create an intelligent execution plan.

        Args:
            task: The investment task to plan for

        Returns:
            List of planned steps with tool calls
        """
        tools_info = self._format_tools_for_prompt()

        system_prompt = f"""You are an intelligent investment planning assistant. Your job is to create a step-by-step plan to accomplish investment analysis tasks using available MCP tools.

Available tools:
{tools_info}

Rules:
1. Create a logical sequence of tool calls to accomplish the task
2. Each step should have: step_number, tool_name, description, and arguments
3. Consider dependencies between steps (e.g., you need to get stock data before analyzing it)
4. Be specific with arguments - extract relevant information from the task (stock symbols, time periods, etc.)
5. If the task is complex, break it into multiple steps
6. Focus on investment-specific analysis: stock analysis, portfolio optimization, risk assessment, market screening
7. Output valid JSON format

Example output format:
[
  {{
    "step": 1,
    "tool": "analyze_stock",
    "description": "Analyze the specified stock for investment potential",
    "arguments": {{"symbol": "AAPL", "analysis_type": "comprehensive", "include_technical": true, "include_fundamental": true}}
  }},
  {{
    "step": 2,
    "tool": "risk_assessment",
    "description": "Assess the investment risk of the analyzed stock",
    "arguments": {{"symbol": "AAPL", "risk_factors": ["volatility", "sector_risk", "market_risk"]}}
  }}
]"""

        user_prompt = f"""Create a step-by-step plan to accomplish this investment task: "{task}"

Consider what information an investor might want and plan accordingly. Think about:
- What stocks or investments to analyze
- Whether technical and/or fundamental analysis is needed
- If portfolio analysis or optimization would be valuable
- Risk assessment requirements
- Market screening or comparison needs
- The logical order of operations

Respond with a JSON array of steps."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await self.openai_client.chat_completion(
                messages=messages,
                model=self.config.openai_config.default_model,
                temperature=0.3,
                max_tokens=1000,
            )

            plan_text = response.choices[0].message.content.strip()

            # Parse the JSON response
            import json

            plan = json.loads(plan_text)

            logger.info(f"AI created plan with {len(plan)} steps for task: {task}")
            return plan

        except Exception as e:
            logger.error(f"Failed to create AI plan: {e}")
            # Fallback to simple heuristic planning
            return await self._fallback_plan(task)

    async def _fallback_plan(self, task: str) -> List[Dict[str, Any]]:
        """
        Fallback planning using simple heuristics.

        Args:
            task: The task to plan for

        Returns:
            Simple plan based on heuristics
        """
        plan = []
        task_lower = task.lower()

        # Extract potential stock symbol from task
        import re

        symbol_match = re.search(r"\b([A-Z]{1,5})\b", task.upper())
        symbol = symbol_match.group(1) if symbol_match else "AAPL"

        # Always start with stock analysis if a symbol is mentioned
        if any(
            word in task_lower
            for word in ["stock", "analyze", "investment", "buy", "sell"]
        ):
            plan.append(
                {
                    "step": 1,
                    "tool": "analyze_stock",
                    "description": "Analyze stock for investment potential",
                    "arguments": {"symbol": symbol, "analysis_type": "comprehensive"},
                }
            )

        # Add portfolio analysis if requested
        if "portfolio" in task_lower:
            plan.append(
                {
                    "step": len(plan) + 1,
                    "tool": "analyze_portfolio",
                    "description": "Analyze portfolio performance and optimization",
                    "arguments": {"analysis_type": "comprehensive"},
                }
            )

        # Add risk assessment if requested
        if "risk" in task_lower:
            plan.append(
                {
                    "step": len(plan) + 1,
                    "tool": "risk_assessment",
                    "description": "Assess investment risk",
                    "arguments": {"symbol": symbol if symbol != "AAPL" else None},
                }
            )

        # Add screening if requested
        if any(word in task_lower for word in ["screen", "find", "search", "best"]):
            plan.append(
                {
                    "step": len(plan) + 1,
                    "tool": "screen_stocks",
                    "description": "Screen stocks based on criteria",
                    "arguments": {"criteria": {"pe_ratio_max": 25, "roe_min": 0.15}},
                }
            )

        # Default to basic stock analysis if no specific plan
        if not plan:
            plan.append(
                {
                    "step": 1,
                    "tool": "analyze_stock",
                    "description": "Perform basic stock analysis",
                    "arguments": {"symbol": "AAPL", "analysis_type": "brief"},
                }
            )

        logger.info(f"Created fallback plan with {len(plan)} steps")
        return plan

    async def _execute_plan(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute the planned steps intelligently.

        Args:
            plan: List of planned steps

        Returns:
            Results from each step
        """
        results = []
        context = {}  # Shared context between steps

        for step in plan:
            step_num = step["step"]
            tool_name = step["tool"]
            description = step["description"]
            arguments = step.get("arguments", {})

            logger.info(f"Executing step {step_num}: {description}")

            try:
                # Enhance arguments with context from previous steps
                enhanced_arguments = await self._enhance_arguments_with_context(
                    tool_name, arguments, context
                )

                result = await self.call_tool(tool_name, enhanced_arguments)

                # Store useful information in context for next steps
                context[f"step_{step_num}_result"] = result
                if tool_name == "analyze_stock" and isinstance(result, dict):
                    context["stock_analysis"] = result
                elif tool_name == "screen_stocks" and isinstance(result, list):
                    context["screened_stocks"] = result

                step_result = {
                    "step": step_num,
                    "tool": tool_name,
                    "description": description,
                    "arguments": enhanced_arguments,
                    "result": result,
                    "success": True,
                }

                results.append(step_result)
                logger.info(f"Step {step_num} completed successfully")

            except Exception as e:
                step_result = {
                    "step": step_num,
                    "tool": tool_name,
                    "description": description,
                    "arguments": arguments,
                    "error": str(e),
                    "success": False,
                }

                results.append(step_result)
                logger.error(f"Step {step_num} failed: {e}")

        return results

    async def _enhance_arguments_with_context(
        self, tool_name: str, arguments: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance tool arguments with context from previous steps.

        Args:
            tool_name: Name of the tool being called
            arguments: Original arguments
            context: Context from previous steps

        Returns:
            Enhanced arguments
        """
        enhanced = arguments.copy()

        # If we need stock data and have analysis from previous steps
        if tool_name == "risk_assessment" and "symbol" not in enhanced:
            stock_analysis = context.get("stock_analysis", {})
            if stock_analysis and "symbol" in stock_analysis:
                enhanced["symbol"] = stock_analysis["symbol"]
                logger.info("Enhanced risk_assessment with symbol from stock analysis")

        # If we're doing portfolio analysis and have screened stocks
        if tool_name == "analyze_portfolio" and "stocks" not in enhanced:
            screened_stocks = context.get("screened_stocks", [])
            if screened_stocks:
                enhanced["stocks"] = [stock["symbol"] for stock in screened_stocks[:5]]
                logger.info("Enhanced portfolio analysis with screened stocks")

        return enhanced

    async def analyze(self, task: str) -> Dict[str, Any]:
        """
        Perform an intelligent investment analysis task.

        Args:
            task: Description of the investment analysis task

        Returns:
            Complete investment analysis results with plan and execution details
        """
        if not self.session:
            await self.connect()

        # Create AI-powered plan
        plan = await self._create_ai_plan(task)

        # Execute the plan
        results = await self._execute_plan(plan)

        return {
            "task": task,
            "plan": plan,
            "execution_results": results,
            "summary": self._create_summary(results),
        }

    def _create_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of the execution results."""
        successful_steps = [r for r in results if r.get("success", False)]
        failed_steps = [r for r in results if not r.get("success", False)]

        return {
            "total_steps": len(results),
            "successful_steps": len(successful_steps),
            "failed_steps": len(failed_steps),
            "tools_used": list(set(r["tool"] for r in successful_steps)),
            "has_errors": len(failed_steps) > 0,
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        try:
            if self.session:
                await self.session.__aexit__(exc_type, exc_val, exc_tb)
                self.session = None

            if self._client_context:
                await self._client_context.__aexit__(exc_type, exc_val, exc_tb)
                self._client_context = None

            self._read_stream = None
            self._write_stream = None
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Convenience function for quick AI-powered investment analysis
async def ai_investment_analysis(task: str) -> Dict[str, Any]:
    """
    Perform AI-powered investment analysis with automatic tool selection and planning.

    Args:
        task: Description of what you want to analyze

    Returns:
        Complete investment analysis results
    """
    async with AIInvestmentAgent() as agent:
        return await agent.analyze(task)
