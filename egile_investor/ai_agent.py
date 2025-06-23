"""
AI-Powered Investment Agent that uses LLM reasoning to plan and execute MCP tool usage.

This agent uses an LLM to intelligently plan which tools to use and in what sequence
to accomplish complex investment analysis tasks.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

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

                # Check if result contains errors and provide fallback if needed
                if tool_name == "analyze_multiple_stocks" and isinstance(result, dict):
                    # Check for individual symbol errors
                    error_symbols = []
                    for symbol, symbol_result in result.items():
                        if isinstance(symbol_result, dict) and "error" in symbol_result:
                            error_msg = str(symbol_result["error"])
                            if (
                                "FunctionTool" in error_msg
                                and "not callable" in error_msg
                            ):
                                error_symbols.append(symbol)
                                logger.debug(
                                    f"Detected FunctionTool error for {symbol}: {error_msg}"
                                )

                    if error_symbols:
                        logger.warning(
                            f"Found FunctionTool errors for symbols: {error_symbols} - providing fallback results"
                        )
                        # Provide fallback results for error symbols
                        for symbol in error_symbols:
                            original_error = result[symbol].get(
                                "error", "Unknown error"
                            )
                            result[symbol] = {
                                "symbol": symbol,
                                "analysis_type": "fallback",
                                "timestamp": datetime.now().isoformat(),
                                "basic_info": {
                                    "symbol": symbol,
                                    "company_name": f"{symbol} Corporation",
                                    "note": "Limited analysis due to technical issues",
                                },
                                "fallback_recommendation": {
                                    "status": "analysis_limited",
                                    "message": f"Full analysis for {symbol} could not be completed due to technical issues. Consider manual research for this symbol.",
                                    "confidence": "Low",
                                    "original_error": original_error,
                                },
                                "overall_assessment": {
                                    "technical_score": 0,
                                    "fundamental_score": 0,
                                    "overall_score": 0,
                                    "recommendation": "RESEARCH_REQUIRED",
                                    "confidence": "Low",
                                    "risk_level": "Unknown",
                                },
                            }
                        logger.info(
                            f"Provided fallback analysis for {len(error_symbols)} symbols with errors"
                        )
                    else:
                        logger.debug(
                            "No FunctionTool errors detected in analyze_multiple_stocks result"
                        )
                else:
                    logger.debug(f"Tool {tool_name} result type: {type(result)}")

                # Store useful information in context for next steps
                context[f"step_{step_num}_result"] = result
                if tool_name == "analyze_stock" and isinstance(result, dict):
                    context["stock_analysis"] = result
                elif tool_name == "screen_stocks" and isinstance(result, list):
                    context["screened_stocks"] = result
                    # Extract symbols from screening results
                    symbols = []
                    for stock in result:
                        if isinstance(stock, dict) and "symbol" in stock:
                            symbols.append(stock["symbol"])
                    if symbols:
                        context["stored_symbols"] = symbols
                        logger.info(
                            f"Stored {len(symbols)} screened symbols: {symbols[:3]}..."
                        )
                elif tool_name == "get_screening_symbols" and isinstance(result, list):
                    # Store symbols directly if they're returned as a list
                    context["stored_symbols"] = result
                    logger.info(
                        f"Stored {len(result)} screened symbols: {result[:3]}..."
                    )
                elif (
                    tool_name == "get_screening_symbols"
                    and isinstance(result, dict)
                    and "symbols" in result
                ):
                    # Store symbols if they're in a dict format
                    context["stored_symbols"] = result["symbols"]
                    logger.info(
                        f"Stored {len(result['symbols'])} screened symbols: {result['symbols'][:3]}..."
                    )

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
                error_msg = str(e)
                logger.error(f"Step {step_num} failed: {error_msg}")

                # Special handling for 'FunctionTool' object is not callable error
                if "'FunctionTool' object is not callable" in error_msg:
                    logger.warning(
                        f"MCP tool error for {tool_name}, attempting fallback..."
                    )

                    # Try to provide a fallback result based on the tool
                    fallback_result = await self._provide_fallback_result(
                        tool_name, enhanced_arguments, context
                    )
                    if fallback_result:
                        step_result = {
                            "step": step_num,
                            "tool": tool_name,
                            "description": description,
                            "arguments": enhanced_arguments,
                            "result": fallback_result,
                            "success": True,
                            "fallback_used": True,
                        }
                        logger.info(f"Step {step_num} completed using fallback")
                    else:
                        step_result = {
                            "step": step_num,
                            "tool": tool_name,
                            "description": description,
                            "arguments": arguments,
                            "error": error_msg,
                            "success": False,
                        }
                else:
                    step_result = {
                        "step": step_num,
                        "tool": tool_name,
                        "description": description,
                        "arguments": arguments,
                        "error": error_msg,
                        "success": False,
                    }

                results.append(step_result)

        return results

    async def _enhance_arguments_with_context(
        self, tool_name: str, arguments: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhanced argument processing with better context handling.
        """
        enhanced = arguments.copy()

        # Handle create_investment_report specifically
        if tool_name == "create_investment_report":
            # Filter out unsupported parameters - only keep supported ones
            supported_params = {"user_query", "analysis_results", "investment_amount"}
            enhanced = {k: v for k, v in enhanced.items() if k in supported_params}

            # Define the flattening function at the start so it can be reused
            def flatten_and_extract_dicts(items):
                """Recursively flatten lists and extract dictionary items."""
                result = []
                for item in items:
                    if isinstance(item, str) and "{result_of_step_" in item:
                        # Extract step number from placeholder
                        import re

                        match = re.search(r"step_(\d+)", item)
                        if match:
                            step_num = int(match.group(1))
                            step_result = context.get(f"step_{step_num}_result")
                            if step_result and isinstance(step_result, dict):
                                result.append(step_result)
                            elif step_result and isinstance(step_result, list):
                                # Recursively handle nested lists
                                result.extend(flatten_and_extract_dicts(step_result))
                            else:
                                logger.warning(
                                    f"No valid result found for step {step_num}"
                                )
                    elif isinstance(item, dict):
                        result.append(item)
                    elif isinstance(item, list):
                        # Recursively flatten nested lists
                        result.extend(flatten_and_extract_dicts(item))
                    else:
                        logger.warning(
                            f"Skipping invalid analysis result type: {type(item)}"
                        )
                return result

            # Fix analysis_results if it contains placeholder strings or invalid types
            if "analysis_results" in enhanced:
                analysis_results = enhanced["analysis_results"]
                if isinstance(analysis_results, list):
                    fixed_results = flatten_and_extract_dicts(analysis_results)
                    enhanced["analysis_results"] = fixed_results
                    logger.info(
                        f"Fixed analysis_results: {len(fixed_results)} valid dictionaries"
                    )

            # Also try to collect all analysis results from context if list is empty
            if (
                not enhanced.get("analysis_results")
                or len(enhanced["analysis_results"]) == 0
            ):
                collected_results = []
                for key, value in context.items():
                    if key.startswith("step_") and key.endswith("_result") and value:
                        collected_results.append(value)
                if collected_results:
                    # Apply the same flattening logic to collected results
                    flattened_collected = flatten_and_extract_dicts(collected_results)
                    enhanced["analysis_results"] = flattened_collected
                    logger.info(
                        f"Collected {len(collected_results)} analysis results from context"
                    )

        # Handle summarize_analysis_execution specifically
        elif tool_name == "summarize_analysis_execution":
            # Filter out unsupported parameters - only keep supported ones
            supported_params = {"user_query", "execution_results", "investment_amount"}
            enhanced = {k: v for k, v in enhanced.items() if k in supported_params}

        # Simple symbol extraction for single symbol tools
        if tool_name == "risk_assessment" and "symbol" not in enhanced:
            stock_analysis = context.get("stock_analysis", {})
            if stock_analysis and "symbol" in stock_analysis:
                enhanced["symbol"] = stock_analysis["symbol"]

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

        # Validate that results answer the user query and create investment report
        investment_report = await self._create_final_investment_report(task, results)

        # Create execution summary using the new summarization tool
        execution_summary = await self._create_execution_summary(task, results)

        return {
            "task": task,
            "plan": plan,
            "execution_results": results,
            "summary": self._create_summary(results),
            "investment_report": investment_report,
            "execution_summary": execution_summary,
        }

    async def _create_final_investment_report(
        self, task: str, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a final investment report that validates the analysis answers the user's query.

        Args:
            task: Original user task/query
            results: Execution results from the analysis

        Returns:
            Investment report or validation error
        """
        try:
            # Extract analysis results for the report
            analysis_data = []
            for result in results:
                if result.get("success") and result.get("result"):
                    analysis_data.append(result["result"])

            # Extract investment amount if mentioned in the task
            investment_amount = None
            import re

            amount_match = re.search(r"\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", task)
            if amount_match:
                investment_amount = float(amount_match.group(1).replace(",", ""))

            # Validate that we have sufficient data to answer the query
            validation_result = await self._validate_analysis_completeness(
                task, analysis_data
            )

            if validation_result["is_sufficient"]:
                # Use the new MCP tool to create investment report
                report_result = await self.call_tool(
                    "create_investment_report",
                    {
                        "user_query": task,
                        "analysis_results": analysis_data,
                        "investment_amount": investment_amount,
                    },
                )

                return {
                    "status": "success",
                    "validation": validation_result,
                    "report": report_result,
                }
            else:
                return {
                    "status": "insufficient_data",
                    "validation": validation_result,
                    "fallback_advice": await self._generate_fallback_advice(
                        task, investment_amount
                    ),
                }

        except Exception as e:
            logger.error(f"Error creating investment report: {e}")
            return {
                "status": "error",
                "error": str(e),
                "fallback_advice": await self._generate_fallback_advice(task, None),
            }

    async def _validate_analysis_completeness(
        self, task: str, analysis_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate that the analysis results are sufficient to answer the user's query.

        Args:
            task: Original user query
            analysis_data: Results from analysis steps

        Returns:
            Validation result with sufficiency assessment
        """
        validation = {
            "is_sufficient": False,
            "has_stock_recommendations": False,
            "has_risk_assessment": False,
            "has_financial_metrics": False,
            "missing_elements": [],
            "confidence_score": 0.0,
        }

        # Check for stock recommendations
        for data in analysis_data:
            if isinstance(data, dict):
                if "symbol" in data or "symbols" in data:
                    validation["has_stock_recommendations"] = True
                if any(key in data for key in ["risk_level", "volatility", "beta"]):
                    validation["has_risk_assessment"] = True
                if any(key in data for key in ["pe_ratio", "roe", "dividend_yield"]):
                    validation["has_financial_metrics"] = True

        # Check for list of stocks from screening
        for data in analysis_data:
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict) and "symbol" in data[0]:
                    validation["has_stock_recommendations"] = True

        # Determine missing elements
        if not validation["has_stock_recommendations"]:
            validation["missing_elements"].append("stock_recommendations")
        if not validation["has_risk_assessment"]:
            validation["missing_elements"].append("risk_assessment")
        if not validation["has_financial_metrics"]:
            validation["missing_elements"].append("financial_metrics")

        # Calculate confidence score
        score = 0
        if validation["has_stock_recommendations"]:
            score += 0.5
        if validation["has_risk_assessment"]:
            score += 0.3
        if validation["has_financial_metrics"]:
            score += 0.2

        validation["confidence_score"] = score
        validation["is_sufficient"] = (
            score >= 0.5
        )  # Need at least stock recommendations

        return validation

    async def _generate_fallback_advice(
        self, task: str, investment_amount: Optional[float]
    ) -> Dict[str, Any]:
        """
        Generate general investment advice when analysis is insufficient.

        Args:
            task: Original user query
            investment_amount: Investment amount if specified

        Returns:
            General investment advice
        """
        amount = investment_amount or 10000

        advice = {
            "general_recommendations": [
                "Consider diversified index funds (S&P 500, Total Stock Market)",
                "Look for blue-chip stocks with consistent dividend history",
                "Diversify across sectors (technology, healthcare, consumer goods)",
                "Use dollar-cost averaging over 3-6 months",
            ],
            "allocation_suggestion": {
                "index_funds": f"${amount * 0.6:,.0f} (60%)",
                "individual_stocks": f"${amount * 0.3:,.0f} (30%)",
                "cash_reserves": f"${amount * 0.1:,.0f} (10%)",
            },
            "risk_management": [
                "Start with low-cost index funds for broad market exposure",
                "Gradually add individual stocks as you gain experience",
                "Keep 3-6 months of expenses in emergency fund",
                "Review and rebalance quarterly",
            ],
            "next_steps": [
                "Research specific sectors that interest you",
                "Consider consulting with a financial advisor",
                "Start with small positions to gain experience",
                "Continue learning about fundamental analysis",
            ],
        }

        return advice

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

    async def close(self):
        """Close the agent and clean up resources."""
        await self.__aexit__(None, None, None)

    async def _provide_fallback_result(
        self, tool_name: str, arguments: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Provide fallback results when MCP tools fail.

        Args:
            tool_name: Name of the failed tool
            arguments: Arguments that were passed to the tool
            context: Execution context

        Returns:
            Fallback result or None if no fallback available
        """
        try:
            if tool_name == "analyze_multiple_stocks":
                # Provide a basic fallback analysis result
                symbols = arguments.get("symbols", ["AAPL", "MSFT", "GOOGL"])
                if isinstance(symbols, str):
                    symbols = [symbols]

                fallback_results = {}
                for symbol in symbols[:3]:  # Limit to 3 symbols for fallback
                    fallback_results[symbol] = {
                        "symbol": symbol,
                        "analysis_type": "fallback",
                        "basic_info": {
                            "symbol": symbol,
                            "company_name": f"{symbol} Corporation",
                            "note": "Limited analysis due to technical issues",
                        },
                        "fallback_recommendation": {
                            "status": "analysis_limited",
                            "message": f"Full analysis for {symbol} could not be completed due to technical issues. Consider manual research for this symbol.",
                            "confidence": "Low",
                        },
                    }

                logger.info(
                    f"Provided fallback analysis for {len(fallback_results)} symbols"
                )
                return fallback_results

            elif tool_name == "screen_stocks":
                # Provide basic screening fallback
                return [
                    {
                        "symbol": "Example",
                        "score": 0.0,
                        "company_name": "Screening unavailable",
                        "note": "Stock screening could not be completed due to technical issues",
                    }
                ]

            # Add more fallback cases as needed
            return None

        except Exception as e:
            logger.error(f"Fallback result generation failed for {tool_name}: {e}")
            return None

    async def _create_execution_summary(
        self, task: str, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a comprehensive summary of the analysis execution using the MCP tool.

        Args:
            task: Original user task/query
            results: Execution results from the analysis

        Returns:
            Execution summary with step analysis and conclusions
        """
        try:
            # Extract investment amount if mentioned in the task
            investment_amount = None
            import re

            amount_match = re.search(r"\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", task)
            if amount_match:
                investment_amount = float(amount_match.group(1).replace(",", ""))

            # Call the new summarization tool
            summary_result = await self.call_tool(
                "summarize_analysis_execution",
                {
                    "user_query": task,
                    "execution_results": results,
                    "investment_amount": investment_amount,
                },
            )

            logger.info("Created comprehensive execution summary")
            return {
                "status": "success",
                "summary": summary_result,
            }

        except Exception as e:
            logger.error(f"Error creating execution summary: {e}")
            # Provide a fallback summary
            success_count = sum(1 for result in results if result.get("success", False))
            return {
                "status": "fallback",
                "error": str(e),
                "basic_summary": {
                    "total_steps": len(results),
                    "successful_steps": success_count,
                    "success_rate": f"{(success_count / len(results) * 100):.1f}%"
                    if results
                    else "0%",
                    "note": "Detailed execution summary could not be generated due to technical issues",
                },
            }


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
