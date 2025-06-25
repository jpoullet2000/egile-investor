"""
Generic AI Agent that works with any MCP server, similar to how Copilot works.

This agent uses LLM reasoning to understand available tools and create execution plans
that work with any MCP server, not just investment-specific ones.
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


class GenericMCPAgent:
    """
    A generic AI agent that can work with any MCP server by using LLM reasoning
    to understand available tools and create appropriate execution plans.
    """

    def __init__(
        self,
        config: Optional[InvestmentAgentConfig] = None,
        server_command: Optional[str] = None,
    ):
        """
        Initialize the generic MCP agent.

        Args:
            config: Configuration for the OpenAI client
            server_command: Command to start the MCP server (e.g., "python -m my_server")
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
            for tool_name in self.available_tools.keys():
                logger.debug(f"Available tool: {tool_name}")

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a specific MCP tool and return the clean result."""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        if tool_name not in self.available_tools:
            raise ValueError(f"Tool '{tool_name}' not available")

        try:
            response = await self.session.call_tool(tool_name, arguments)
            logger.debug(f"Called tool '{tool_name}' successfully")

            # Extract result from MCP response format
            if hasattr(response, "content"):
                if isinstance(response.content, list) and len(response.content) > 0:
                    first_content = response.content[0]
                    if hasattr(first_content, "text"):
                        try:
                            # Try to parse as JSON first
                            return json.loads(first_content.text)
                        except (json.JSONDecodeError, ValueError):
                            # Return as string if not JSON
                            return first_content.text
                    else:
                        return first_content
                elif isinstance(response.content, list):
                    # Empty content list - return empty list
                    return []
                else:
                    return response.content
            else:
                # No content attribute - return empty list for consistency
                return []

        except Exception as e:
            logger.error(f"Failed to call tool '{tool_name}': {e}")
            raise

    def _format_tools_for_llm(self) -> str:
        """Format available tools for LLM understanding."""
        tool_descriptions = []

        for name, tool in self.available_tools.items():
            description = tool.description or "No description available"

            # Format input schema
            params_info = ""
            if hasattr(tool, "inputSchema") and tool.inputSchema:
                properties = tool.inputSchema.get("properties", {})
                required = tool.inputSchema.get("required", [])

                if properties:
                    param_lines = []
                    for param_name, param_info in properties.items():
                        param_type = param_info.get("type", "any")
                        param_desc = param_info.get("description", "")
                        is_required = (
                            "(required)" if param_name in required else "(optional)"
                        )
                        param_lines.append(
                            f"    - {param_name} ({param_type}) {is_required}: {param_desc}"
                        )

                    params_info = "\n  Parameters:\n" + "\n".join(param_lines)

            tool_descriptions.append(f"- **{name}**: {description}{params_info}")

        return "\n".join(tool_descriptions)

    async def _create_execution_plan(self, user_request: str) -> List[Dict[str, Any]]:
        """
        Use LLM to create a generic execution plan based on available tools.
        """
        tools_info = self._format_tools_for_llm()

        system_prompt = f"""You are an intelligent AI assistant that can work with any MCP (Model Context Protocol) server. Your job is to create a step-by-step execution plan to fulfill user requests using the available MCP tools.

Available MCP Tools:
{tools_info}

Rules for creating execution plans:
1. Analyze the user request to understand what they want to accomplish
2. Create a logical sequence of tool calls using the available tools
3. Each step should have: step_number, tool_name, description, and arguments
4. Consider dependencies between steps (some tools may need results from previous steps)
5. Use realistic arguments based on the user request and tool schemas
6. Be creative but practical - work with what's available
7. Output valid JSON format

Example format:
[
  {{
    "step": 1,
    "tool": "tool_name_here",
    "description": "What this step accomplishes",
    "arguments": {{"param1": "value1", "param2": "value2"}}
  }},
  {{
    "step": 2,
    "tool": "another_tool",
    "description": "Next step description", 
    "arguments": {{"param": "{{result_from_step_1}}"}}
  }}
]

If you need to reference results from previous steps, use the format: "{{result_from_step_N}}" where N is the step number.
"""

        user_prompt = f"""Create a step-by-step execution plan to fulfill this request: "{user_request}"

Analyze what the user wants and determine:
1. Which available tools can help accomplish this task
2. What order to call them in
3. What arguments each tool needs
4. How to chain results between steps if needed

Respond with a JSON array of execution steps."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await self.openai_client.chat_completion(
                messages=messages,
                model=self.config.openai_config.default_model,
                temperature=0.2,
                max_tokens=1500,
            )

            plan_text = response.choices[0].message.content.strip()

            # Extract JSON from response (handle cases where LLM adds extra text)
            json_match = re.search(r"\[.*\]", plan_text, re.DOTALL)
            if json_match:
                plan_text = json_match.group(0)

            plan = json.loads(plan_text)

            logger.info(f"Created execution plan with {len(plan)} steps")
            return plan

        except Exception as e:
            logger.error(f"Failed to create execution plan: {e}")
            # Return a simple fallback plan
            return [
                {
                    "step": 1,
                    "tool": list(self.available_tools.keys())[0],
                    "description": "Execute available tool with user request",
                    "arguments": {"query": user_request},
                }
            ]

    def _resolve_step_references(
        self, arguments: Dict[str, Any], step_results: Dict[int, Any]
    ) -> Dict[str, Any]:
        """
        Resolve references to previous step results in arguments with type validation.
        """
        resolved = {}

        for key, value in arguments.items():
            if isinstance(value, str) and "{result_from_step_" in value:
                # Extract step number
                step_match = re.search(r"result_from_step_(\d+)", value)
                if step_match:
                    step_num = int(step_match.group(1))
                    if step_num in step_results:
                        result = step_results[step_num]
                        # Handle MCP server validation - ensure we pass the right type
                        if isinstance(result, list) and key == "screening_results":
                            # Only screening_results needs JSON serialization
                            try:
                                resolved[key] = json.dumps(result)
                            except (TypeError, ValueError):
                                resolved[key] = str(result)
                        else:
                            # For other parameters, pass the actual data type
                            resolved[key] = result
                    else:
                        logger.warning(
                            f"Referenced step {step_num} not found in results"
                        )
                        resolved[key] = None
                else:
                    resolved[key] = value
            elif isinstance(value, dict):
                # Recursively resolve nested dictionaries
                resolved[key] = self._resolve_step_references(value, step_results)
            elif isinstance(value, list):
                # Handle lists that might contain references
                resolved_list = []
                for item in value:
                    if isinstance(item, str) and "{result_from_step_" in item:
                        step_match = re.search(r"result_from_step_(\d+)", item)
                        if step_match:
                            step_num = int(step_match.group(1))
                            if step_num in step_results:
                                resolved_list.append(step_results[step_num])
                        else:
                            resolved_list.append(item)
                    else:
                        resolved_list.append(item)
                resolved[key] = resolved_list
            else:
                resolved[key] = value

        return resolved

    async def _execute_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute the plan step by step with proper error handling.
        """
        step_results = {}
        execution_log = []

        for step in plan:
            step_num = step["step"]
            tool_name = step["tool"]
            description = step["description"]
            arguments = step.get("arguments", {})

            logger.info(f"Executing step {step_num}: {description}")

            try:
                # Resolve any references to previous step results
                resolved_args = self._resolve_step_references(arguments, step_results)

                # Validate that the tool exists
                if tool_name not in self.available_tools:
                    raise ValueError(f"Tool '{tool_name}' not available")

                # Call the tool
                result = await self.call_tool(tool_name, resolved_args)

                # Store the result
                step_results[step_num] = result

                execution_log.append(
                    {
                        "step": step_num,
                        "tool": tool_name,
                        "description": description,
                        "success": True,
                        "result_preview": str(result)[:200] + "..."
                        if len(str(result)) > 200
                        else str(result),
                    }
                )

                logger.info(f"Step {step_num} completed successfully")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Step {step_num} failed: {error_msg}")

                execution_log.append(
                    {
                        "step": step_num,
                        "tool": tool_name,
                        "description": description,
                        "success": False,
                        "error": error_msg,
                    }
                )

                # Continue execution even if one step fails
                step_results[step_num] = {"error": error_msg}

        return {
            "step_results": step_results,
            "execution_log": execution_log,
            "success_count": len([log for log in execution_log if log["success"]]),
            "total_steps": len(execution_log),
        }

    async def execute_request(self, user_request: str) -> Dict[str, Any]:
        """
        Main method to execute a user request using available MCP tools.

        Args:
            user_request: The user's request in natural language

        Returns:
            Complete execution results with plan, steps, and final results
        """
        if not self.session:
            await self.connect()

        try:
            # Create execution plan
            logger.info("Creating execution plan...")
            plan = await self._create_execution_plan(user_request)

            # Execute the plan
            logger.info("Executing plan...")
            execution = await self._execute_plan(plan)

            # Validate execution results
            logger.info("Validating execution results...")
            validation = await self._validate_execution_results(user_request, execution)

            if not validation["adequately_answered"]:
                logger.info(
                    "Execution results inadequate, taking corrective actions..."
                )
                # Take corrective actions based on validation feedback
                if validation["suggested_additional_steps"]:
                    # Execute additional steps if suggested
                    additional_execution = await self._execute_additional_steps(
                        user_request,
                        execution,
                        validation["suggested_additional_steps"],
                    )
                    execution = additional_execution

            # Compile final response
            main_response = self._extract_main_response(execution)

            response = {
                "user_request": user_request,
                "response": main_response,  # Main response content (markdown, etc.)
                "execution_plan": plan,
                "execution_results": execution,
                "final_results": execution["step_results"],
                "success_rate": f"{execution['success_count']}/{execution['total_steps']}",
                "validation": validation,
                "status": "completed"
                if validation["adequately_answered"]
                else "completed_with_concerns",
            }

            logger.info(
                f"Request execution completed with {execution['success_count']}/{execution['total_steps']} successful steps"
            )

            return response

        except Exception as e:
            logger.error(f"Request execution failed: {e}")
            return {"user_request": user_request, "error": str(e), "status": "failed"}

    async def _validate_execution_results(
        self, user_request: str, execution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that the execution results meaningfully address the user's request.

        Args:
            user_request: The original user request
            execution: The execution results from _execute_plan

        Returns:
            Validation assessment with suggestions for improvement
        """
        try:
            # Use LLM to evaluate if the results answer the user's query
            validation_prompt = f"""You are evaluating whether the execution results adequately answer a user's request.

USER REQUEST: "{user_request}"

EXECUTION SUMMARY:
- Total steps: {execution["total_steps"]}
- Successful steps: {execution["success_count"]}
- Success rate: {execution["success_count"]}/{execution["total_steps"]}

EXECUTION LOG:
{json.dumps(execution["execution_log"], indent=2)}

STEP RESULTS SUMMARY:
{self._summarize_step_results(execution["step_results"])}

Please evaluate:
1. Do the execution results contain meaningful data that addresses the user's request?
2. Are there obvious gaps or missing information?
3. If the results are inadequate, what additional steps could improve them?

Respond with JSON in this format:
{{
    "adequately_answered": true/false,
    "confidence": "high"/"medium"/"low",
    "assessment": "Brief explanation of why results are adequate or inadequate",
    "missing_elements": ["list", "of", "missing", "elements"],
    "suggested_additional_steps": [
        {{"step": "step_description", "tool": "tool_name", "reasoning": "why this would help"}}
    ]
}}"""

            messages = [{"role": "user", "content": validation_prompt}]

            response = await self.openai_client.chat_completion(
                messages=messages,
                model=self.config.openai_config.default_model,
                temperature=0.2,
                max_tokens=800,
            )

            validation_text = response.choices[0].message.content.strip()

            # Extract JSON from response
            json_match = re.search(r"\{.*\}", validation_text, re.DOTALL)
            if json_match:
                validation_result = json.loads(json_match.group(0))
            else:
                # Fallback validation
                validation_result = {
                    "adequately_answered": execution["success_count"] > 0,
                    "confidence": "low",
                    "assessment": "Unable to parse validation response",
                    "missing_elements": [],
                    "suggested_additional_steps": [],
                }

            logger.info(
                f"Result validation: {validation_result['adequately_answered']} (confidence: {validation_result['confidence']})"
            )
            return validation_result

        except Exception as e:
            logger.error(f"Error validating execution results: {e}")
            # Default to accepting results if validation fails
            return {
                "adequately_answered": True,
                "confidence": "low",
                "assessment": f"Validation error: {str(e)}",
                "missing_elements": [],
                "suggested_additional_steps": [],
            }

    def _summarize_step_results(self, step_results: Dict[int, Any]) -> str:
        """Create a brief summary of step results for validation."""
        summary_lines = []
        for step_num, result in step_results.items():
            if isinstance(result, dict):
                if "error" in result:
                    summary_lines.append(f"Step {step_num}: ERROR - {result['error']}")
                else:
                    # Try to extract meaningful info
                    key_info = []
                    for key in [
                        "recommendations",
                        "symbols",
                        "stocks",
                        "analysis",
                        "report",
                    ]:
                        if key in result:
                            value = result[key]
                            if isinstance(value, list):
                                key_info.append(f"{key}: {len(value)} items")
                            elif isinstance(value, dict):
                                key_info.append(f"{key}: dict with {len(value)} keys")
                            else:
                                key_info.append(f"{key}: {type(value).__name__}")

                    if key_info:
                        summary_lines.append(f"Step {step_num}: {', '.join(key_info)}")
                    else:
                        summary_lines.append(
                            f"Step {step_num}: {type(result).__name__} result"
                        )
            else:
                summary_lines.append(
                    f"Step {step_num}: {type(result).__name__} - {str(result)[:50]}..."
                )

        return "\n".join(summary_lines)

    async def _execute_additional_steps(
        self,
        user_request: str,
        current_execution: Dict[str, Any],
        suggested_steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Execute additional steps to improve the results.

        Args:
            user_request: Original user request
            current_execution: Current execution state
            suggested_steps: List of additional steps to execute

        Returns:
            Updated execution results
        """
        logger.info(
            f"Executing {len(suggested_steps)} additional steps to improve results"
        )

        # Continue from where we left off
        step_results = current_execution["step_results"].copy()
        execution_log = current_execution["execution_log"].copy()
        current_step_num = max(step_results.keys()) if step_results else 0

        for i, suggested_step in enumerate(
            suggested_steps[:3]
        ):  # Limit to 3 additional steps
            current_step_num += 1
            tool_name = suggested_step.get("tool", "unknown")
            description = suggested_step.get("step", f"Additional step {i + 1}")
            reasoning = suggested_step.get("reasoning", "Improving results")

            logger.info(f"Executing additional step {current_step_num}: {description}")

            try:
                # For suggested steps, we need to infer arguments based on available tools and context
                arguments = await self._infer_tool_arguments(
                    tool_name, user_request, step_results
                )

                if tool_name in self.available_tools:
                    result = await self.call_tool(tool_name, arguments)
                    step_results[current_step_num] = result

                    execution_log.append(
                        {
                            "step": current_step_num,
                            "tool": tool_name,
                            "description": description,
                            "reasoning": reasoning,
                            "success": True,
                            "result_preview": str(result)[:200] + "..."
                            if len(str(result)) > 200
                            else str(result),
                            "additional_step": True,
                        }
                    )

                    logger.info(
                        f"Additional step {current_step_num} completed successfully"
                    )
                else:
                    logger.warning(f"Suggested tool '{tool_name}' not available")
                    execution_log.append(
                        {
                            "step": current_step_num,
                            "tool": tool_name,
                            "description": description,
                            "reasoning": reasoning,
                            "success": False,
                            "error": f"Tool '{tool_name}' not available",
                            "additional_step": True,
                        }
                    )

            except Exception as e:
                logger.error(f"Additional step {current_step_num} failed: {e}")
                execution_log.append(
                    {
                        "step": current_step_num,
                        "tool": tool_name,
                        "description": description,
                        "reasoning": reasoning,
                        "success": False,
                        "error": str(e),
                        "additional_step": True,
                    }
                )

        # Update execution results
        success_count = len([log for log in execution_log if log.get("success", False)])

        return {
            "step_results": step_results,
            "execution_log": execution_log,
            "success_count": success_count,
            "total_steps": len(execution_log),
        }

    async def _infer_tool_arguments(
        self, tool_name: str, user_request: str, step_results: Dict[int, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM to infer appropriate arguments for a tool based on context.

        Args:
            tool_name: Name of the tool to call
            user_request: Original user request
            step_results: Results from previous steps

        Returns:
            Dictionary of arguments for the tool
        """
        if tool_name not in self.available_tools:
            logger.warning(f"Tool '{tool_name}' not available for argument inference")
            return {}

        try:
            # Get tool schema for the LLM
            tool_schema = self.available_tools[tool_name]

            # Prepare context for LLM
            context_summary = self._summarize_step_results(step_results)

            prompt = f"""You are helping to infer arguments for a tool call based on context.

TOOL TO CALL: {tool_name}

TOOL SCHEMA:
{tool_schema.description}

Parameters:
{self._format_tool_parameters(tool_schema)}

CONTEXT:
User Request: {user_request}
Previous Step Results: {context_summary}

TASK: Generate appropriate arguments for calling '{tool_name}' based on the context above.

RULES:
1. Only include parameters that are defined in the tool schema
2. Use data from previous step results when appropriate (e.g., symbols from screening)
3. Provide reasonable defaults for optional parameters
4. If previous steps contain lists of stocks/symbols, extract just the symbol strings
5. Return arguments as a JSON object

Example format:
{{"parameter_name": "value", "another_param": ["list", "of", "values"]}}

Arguments:"""

            response = await self.openai_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=500,
            )

            arguments_text = response.strip()

            # Try to parse as JSON
            try:
                import json

                arguments = json.loads(arguments_text)
                logger.debug(f"LLM inferred arguments for {tool_name}: {arguments}")
                return arguments
            except json.JSONDecodeError:
                logger.warning(
                    f"Could not parse LLM response as JSON: {arguments_text}"
                )
                return self._fallback_argument_inference(tool_name, step_results)

        except Exception as e:
            logger.error(f"LLM argument inference failed for {tool_name}: {e}")
            return self._fallback_argument_inference(tool_name, step_results)

    def _format_tool_parameters(self, tool_schema) -> str:
        """Format tool parameters for LLM understanding."""
        try:
            if hasattr(tool_schema, "inputSchema") and tool_schema.inputSchema:
                schema = tool_schema.inputSchema
                if isinstance(schema, dict) and "properties" in schema:
                    formatted = []
                    for param_name, param_info in schema["properties"].items():
                        param_type = param_info.get("type", "unknown")
                        param_desc = param_info.get("description", "No description")
                        required = param_name in schema.get("required", [])
                        req_marker = " (required)" if required else " (optional)"
                        formatted.append(
                            f"- {param_name}: {param_type}{req_marker} - {param_desc}"
                        )
                    return "\n".join(formatted)
            return "Parameters not available"
        except Exception as e:
            logger.warning(f"Could not format tool parameters: {e}")
            return "Parameters not available"

    def _summarize_step_results(self, step_results: Dict[int, Any]) -> str:
        """Create a concise summary of step results for LLM context."""
        if not step_results:
            return "No previous steps"

        summary_parts = []
        for step_num, result in step_results.items():
            if isinstance(result, list):
                if result and isinstance(result[0], dict) and "symbol" in result[0]:
                    symbols = [item.get("symbol", "unknown") for item in result[:5]]
                    summary_parts.append(
                        f"Step {step_num}: Stock screening results with symbols: {symbols}"
                    )
                elif result:
                    summary_parts.append(
                        f"Step {step_num}: List with {len(result)} items"
                    )
                else:
                    summary_parts.append(f"Step {step_num}: Empty list")
            elif isinstance(result, dict):
                if "symbol" in result:
                    summary_parts.append(
                        f"Step {step_num}: Single stock data for {result['symbol']}"
                    )
                elif result.get("error"):
                    summary_parts.append(f"Step {step_num}: Error - {result['error']}")
                else:
                    summary_parts.append(f"Step {step_num}: Dictionary result")
            elif isinstance(result, str):
                preview = result[:100] + "..." if len(result) > 100 else result
                summary_parts.append(f"Step {step_num}: Text result - {preview}")
            else:
                summary_parts.append(f"Step {step_num}: {type(result).__name__} result")

        return "\n".join(summary_parts)

    def _fallback_argument_inference(
        self, tool_name: str, step_results: Dict[int, Any]
    ) -> Dict[str, Any]:
        """Simple fallback argument inference when LLM fails."""
        # Basic patterns for common tools
        if "symbols" in str(self.available_tools.get(tool_name, {})):
            # Tool likely needs symbols - extract from step results
            symbols = []
            for result in step_results.values():
                if isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict) and "symbol" in item:
                            symbols.append(item["symbol"])
                        elif isinstance(item, str):
                            symbols.append(item)

            if symbols:
                return {"symbols": symbols[:10]}  # Limit to 10
            else:
                return {"symbols": ["AAPL", "MSFT", "GOOGL"]}  # Fallback

        return {}

    def _extract_main_response(self, execution: Dict[str, Any]) -> str:
        """
        Extract the main response content (like markdown reports) from execution results.

        Args:
            execution: Execution results from _execute_plan

        Returns:
            Main response content as string
        """
        step_results = execution.get("step_results", {})

        # Look for report/content generation tools in reverse order (most recent first)
        for step_num in sorted(step_results.keys(), reverse=True):
            result = step_results[step_num]

            if isinstance(result, dict):
                # Check for markdown content
                if "markdown" in result:
                    return result["markdown"]
                elif "content" in result and isinstance(result["content"], str):
                    return result["content"]
                elif "report" in result and isinstance(result["report"], str):
                    return result["report"]
                elif "text" in result and isinstance(result["text"], str):
                    return result["text"]

            elif isinstance(result, str):
                # Direct string result might be the content
                if len(result) > 100:  # Assume substantial content
                    return result

        # If no substantial content found, summarize the results
        summary_parts = []
        for step_num, result in step_results.items():
            if isinstance(result, dict) and not result.get("error"):
                summary_parts.append(f"Step {step_num}: Completed successfully")
            elif isinstance(result, str):
                preview = result[:200] + "..." if len(result) > 200 else result
                summary_parts.append(f"Step {step_num}: {preview}")

        return (
            "\n".join(summary_parts)
            if summary_parts
            else "Execution completed but no content generated."
        )

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


# Convenience function for generic MCP agent usage
async def generic_mcp_request(
    user_request: str,
    server_command: Optional[str] = None,
    config: Optional[InvestmentAgentConfig] = None,
) -> Dict[str, Any]:
    """
    Execute a generic request against any MCP server.

    Args:
        user_request: The user's request in natural language
        server_command: Command to start the MCP server (optional)
        config: Configuration for the OpenAI client (optional)

    Returns:
        Complete execution results
    """
    async with GenericMCPAgent(config=config, server_command=server_command) as agent:
        return await agent.execute_request(user_request)
