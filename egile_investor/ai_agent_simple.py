"""
Simplified AI-Powered Investment Agent that mimics MCP server tool patterns.

This simplified agent focuses on clean data flow and minimal validation to avoid
the complex validation errors present in the original implementation.
"""

from typing import Any, Dict, List, Optional
import json
import re

import structlog
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Tool

from .config import InvestmentAgentConfig, AzureOpenAIConfig
from .client import AzureOpenAIClient


logger = structlog.get_logger(__name__)


class SimpleAIInvestmentAgent:
    """
    A simplified AI investment agent that mimics MCP server tool patterns.
    """

    def __init__(
        self,
        config: Optional[InvestmentAgentConfig] = None,
        server_command: Optional[str] = None,
    ):
        """Initialize the simplified AI investment agent."""
        self.config = config or InvestmentAgentConfig(
            openai_config=AzureOpenAIConfig.from_environment()
        )
        self.server_command = server_command or "python -m egile_investor.server"
        self.session: Optional[ClientSession] = None
        self.available_tools: Dict[str, Tool] = {}
        self._read_stream = None
        self._write_stream = None
        self._client_context = None
        self.openai_client = AzureOpenAIClient(self.config.openai_config)

    async def connect(self):
        """Connect to the MCP server and discover available tools."""
        try:
            command_parts = self.server_command.split()
            command = command_parts[0]
            args = command_parts[1:] if len(command_parts) > 1 else []

            server_params = StdioServerParameters(command=command, args=args, env=None)
            self._client_context = stdio_client(server_params)
            (
                self._read_stream,
                self._write_stream,
            ) = await self._client_context.__aenter__()

            self.session = ClientSession(self._read_stream, self._write_stream)
            await self.session.__aenter__()
            await self.session.initialize()

            tools_response = await self.session.list_tools()
            self.available_tools = {tool.name: tool for tool in tools_response.tools}

            logger.info(
                f"Connected to MCP server with {len(self.available_tools)} tools"
            )

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a specific MCP tool with clean result extraction."""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        if tool_name not in self.available_tools:
            raise ValueError(f"Tool '{tool_name}' not available")

        try:
            response = await self.session.call_tool(tool_name, arguments)
            logger.info(f"Called tool '{tool_name}' successfully")

            # Proper result extraction from MCP response
            if hasattr(response, "content") and response.content:
                if isinstance(response.content, list) and len(response.content) > 0:
                    first_content = response.content[0]
                    if hasattr(first_content, "text"):
                        try:
                            # Try to parse JSON if it looks like JSON
                            return json.loads(first_content.text)
                        except (json.JSONDecodeError, ValueError):
                            # Return as string if not JSON
                            return first_content.text
                    else:
                        return first_content
                else:
                    return response.content
            else:
                # If no content, return the response itself (which might be the actual result)
                return response

        except Exception as e:
            logger.error(f"Failed to call tool '{tool_name}': {e}")
            # Re-raise the exception so it can be handled by the caller
            raise

    def _create_simple_plan(self, task: str) -> List[Dict[str, Any]]:
        """Create a simple plan based on task keywords - no complex AI planning."""
        plan = []
        task_lower = task.lower()

        # Extract investment amount if mentioned
        investment_amount = None
        amount_match = re.search(r"\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", task)
        if amount_match:
            investment_amount = float(amount_match.group(1).replace(",", ""))

        # Simple heuristic planning based on keywords
        step_num = 1

        # Step 1: Always start with screening if criteria are mentioned
        if any(
            word in task_lower
            for word in [
                "find",
                "screen",
                "pe ratio",
                "dividend yield",
                "dow jones",
                "value stocks",
            ]
        ):
            criteria = {}

            # Extract PE ratio criteria
            pe_match = re.search(r"pe.*?ratio.*?under.*?(\d+)", task_lower)
            if pe_match:
                criteria["pe_ratio_max"] = float(pe_match.group(1))

            # Extract dividend yield criteria
            div_match = re.search(r"dividend.*?yield.*?above.*?(\d+\.?\d*)", task_lower)
            if div_match:
                criteria["dividend_yield_min"] = float(div_match.group(1)) / 100

            # Set universe
            universe = "dow_jones" if "dow jones" in task_lower else "sp500"

            plan.append(
                {
                    "step": step_num,
                    "tool": "screen_stocks",
                    "description": f"Screen {universe} stocks based on specified criteria",
                    "arguments": {
                        "criteria": criteria,
                        "universe": universe,
                        "max_results": 10,
                    },
                }
            )
            step_num += 1

        # Step 2: Get symbols from screening
        if len(plan) > 0:
            plan.append(
                {
                    "step": step_num,
                    "tool": "get_screening_symbols",
                    "description": "Extract stock symbols from screening results",
                    "arguments": {"screening_results": "{result_of_step_1}"},
                }
            )
            step_num += 1

        # Step 3: Analyze the screened stocks
        if len(plan) > 0:
            plan.append(
                {
                    "step": step_num,
                    "tool": "analyze_multiple_stocks",
                    "description": "Perform comprehensive analysis on screened stocks",
                    "arguments": {
                        "symbols": "{result_of_step_2}",
                        "analysis_type": "comprehensive",
                        "include_technical": True,
                        "include_fundamental": True,
                    },
                }
            )
            step_num += 1

        # Step 4: Create investment report
        plan.append(
            {
                "step": step_num,
                "tool": "create_investment_report",
                "description": "Create comprehensive investment report",
                "arguments": {
                    "user_query": task,
                    "analysis_results": [
                        f"{{result_of_step_{i}}}" for i in range(1, step_num)
                    ],
                    "investment_amount": investment_amount,
                },
            }
        )
        step_num += 1

        # Step 5: Create execution summary
        plan.append(
            {
                "step": step_num,
                "tool": "summarize_analysis_execution",
                "description": "Create execution summary",
                "arguments": {
                    "user_query": task,
                    "execution_results": "all_steps_results",
                    "investment_amount": investment_amount,
                },
            }
        )

        logger.info(f"Created simple plan with {len(plan)} steps")
        return plan

    def _resolve_arguments(
        self, arguments: Dict[str, Any], results: Dict[int, Any]
    ) -> Dict[str, Any]:
        """Resolve argument placeholders with actual results, with proper error handling."""
        resolved = {}

        for key, value in arguments.items():
            if isinstance(value, str) and value.startswith("{result_of_step_"):
                # Extract step number
                step_match = re.search(r"step_(\d+)", value)
                if step_match:
                    step_num = int(step_match.group(1))
                    if step_num in results:
                        result = results[step_num]
                        # Only pass valid dictionary results, skip error strings
                        if isinstance(result, dict):
                            resolved[key] = result
                        elif isinstance(result, list):
                            resolved[key] = result
                        else:
                            logger.warning(
                                f"Skipping invalid result type for step {step_num}: {type(result)}"
                            )
                            resolved[key] = None
                    else:
                        logger.warning(f"No result found for step {step_num}")
                        resolved[key] = None
                else:
                    resolved[key] = value
            elif isinstance(value, list):
                # Handle list of placeholders
                resolved_list = []
                for item in value:
                    if isinstance(item, str) and item.startswith("{result_of_step_"):
                        step_match = re.search(r"step_(\d+)", item)
                        if step_match:
                            step_num = int(step_match.group(1))
                            if step_num in results:
                                result = results[step_num]
                                # Only include valid results
                                if isinstance(result, (dict, list)):
                                    resolved_list.append(result)
                                else:
                                    logger.warning(
                                        f"Skipping invalid result for step {step_num}"
                                    )
                        else:
                            if isinstance(item, (dict, list)):
                                resolved_list.append(item)
                    else:
                        if isinstance(item, (dict, list)):
                            resolved_list.append(item)
                resolved[key] = resolved_list
            elif value == "all_steps_results":
                # Special case for execution summary - only include successful results
                valid_results = []
                for step_num, result in results.items():
                    if isinstance(result, (dict, list)):
                        valid_results.append(
                            {"step": step_num, "result": result, "success": True}
                        )
                    else:
                        valid_results.append(
                            {"step": step_num, "error": str(result), "success": False}
                        )
                resolved[key] = valid_results
            else:
                resolved[key] = value

        return resolved

    async def _execute_simple_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute the plan with simple error handling and data validation."""
        results = {}
        step_results = []

        for step in plan:
            step_num = step["step"]
            tool_name = step["tool"]
            description = step["description"]
            arguments = step["arguments"]

            logger.info(f"Executing step {step_num}: {description}")

            try:
                # Resolve argument placeholders
                resolved_args = self._resolve_arguments(arguments, results)

                # Skip tools that need data from failed previous steps
                if tool_name in [
                    "get_screening_symbols",
                    "create_investment_report",
                    "summarize_analysis_execution",
                ]:
                    # Check if required data is available and valid
                    if tool_name == "get_screening_symbols":
                        screening_results = resolved_args.get("screening_results")
                        if not screening_results or not isinstance(
                            screening_results, (dict, list)
                        ):
                            logger.warning(
                                f"Skipping {tool_name} - no valid screening results available"
                            )
                            step_result = {
                                "step": step_num,
                                "tool": tool_name,
                                "description": description,
                                "success": False,
                                "error": "No valid screening results from previous step",
                                "skipped": True,
                            }
                            step_results.append(step_result)
                            continue

                    elif tool_name == "create_investment_report":
                        analysis_results = resolved_args.get("analysis_results", [])
                        # Filter out invalid results
                        valid_results = [
                            r for r in analysis_results if isinstance(r, dict)
                        ]
                        if not valid_results:
                            logger.warning(
                                f"Skipping {tool_name} - no valid analysis results available"
                            )
                            step_result = {
                                "step": step_num,
                                "tool": tool_name,
                                "description": description,
                                "success": False,
                                "error": "No valid analysis results from previous steps",
                                "skipped": True,
                            }
                            step_results.append(step_result)
                            continue
                        else:
                            # Update arguments with only valid results
                            resolved_args["analysis_results"] = valid_results

                # Call the tool
                result = await self.call_tool(tool_name, resolved_args)

                # Store the result
                results[step_num] = result

                step_result = {
                    "step": step_num,
                    "tool": tool_name,
                    "description": description,
                    "success": True,
                    "result": result,
                }

                logger.info(f"Step {step_num} completed successfully")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Step {step_num} failed: {error_msg}")

                # Store the error as a string (not in results dict to avoid passing it to next steps)
                step_result = {
                    "step": step_num,
                    "tool": tool_name,
                    "description": description,
                    "success": False,
                    "error": error_msg,
                }

            step_results.append(step_result)

        return {"step_results": step_results, "final_results": results}

    async def analyze(self, task: str) -> Dict[str, Any]:
        """
        Perform simplified investment analysis - mimics MCP server tool patterns.
        """
        if not self.session:
            await self.connect()

        try:
            # Create simple plan
            plan = self._create_simple_plan(task)

            # Execute plan
            execution = await self._execute_simple_plan(plan)

            # Extract key results
            final_results = execution["final_results"]
            step_results = execution["step_results"]

            # Get the investment report (usually the second-to-last step)
            investment_report = None
            execution_summary = None

            for step_num, result in final_results.items():
                if isinstance(result, dict):
                    # Check if this looks like an investment report
                    if any(
                        key in result
                        for key in [
                            "recommendations",
                            "markdown_report",
                            "query_analysis",
                        ]
                    ):
                        investment_report = result
                    # Check if this looks like an execution summary
                    elif any(
                        key in result
                        for key in ["step_analysis", "final_conclusion", "success_rate"]
                    ):
                        execution_summary = result

            # Build clean response similar to MCP server tools
            response = {
                "task": task,
                "plan": plan,
                "execution_results": step_results,
                "summary": {
                    "total_steps": len(step_results),
                    "successful_steps": sum(
                        1 for r in step_results if r.get("success", False)
                    ),
                    "errors": [
                        r.get("error")
                        for r in step_results
                        if not r.get("success", False) and r.get("error")
                    ],
                },
            }

            if investment_report:
                response["investment_report"] = investment_report

            if execution_summary:
                response["execution_summary"] = execution_summary

            logger.info("Analysis completed successfully")
            return response

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
                self.session = None

            if self._client_context:
                await self._client_context.__aexit__(exc_type, exc_val, exc_tb)
                self._client_context = None

            self._read_stream = None
            self._write_stream = None
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Convenience function for simplified AI investment analysis
async def simple_ai_investment_analysis(task: str) -> Dict[str, Any]:
    """
    Perform simplified AI investment analysis with clean data flow.
    """
    async with SimpleAIInvestmentAgent() as agent:
        return await agent.analyze(task)
