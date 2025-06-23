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
            (self._read_stream, self._write_stream) = await self._client_context.__aenter__()

            self.session = ClientSession(self._read_stream, self._write_stream)
            await self.session.__aenter__()
            await self.session.initialize()

            tools_response = await self.session.list_tools()
            self.available_tools = {tool.name: tool for tool in tools_response.tools}

            logger.info(f"Connected to MCP server with {len(self.available_tools)} tools")
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
            if hasattr(response, "content") and response.content:
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
                else:
                    return response.content
            else:
                return response

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
                        is_required = "(required)" if param_name in required else "(optional)"
                        param_lines.append(f"    - {param_name} ({param_type}) {is_required}: {param_desc}")
                    
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
            json_match = re.search(r'\[.*\]', plan_text, re.DOTALL)
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
                    "arguments": {"query": user_request}
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
                        if isinstance(result, list) and key in ["screening_results", "symbols"]:
                            # For tools that expect strings but get lists, try to serialize
                            try:
                                resolved[key] = json.dumps(result)
                            except (TypeError, ValueError):
                                resolved[key] = str(result)
                        else:
                            resolved[key] = result
                    else:
                        logger.warning(f"Referenced step {step_num} not found in results")
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
                
                execution_log.append({
                    "step": step_num,
                    "tool": tool_name,
                    "description": description,
                    "success": True,
                    "result_preview": str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                })
                
                logger.info(f"Step {step_num} completed successfully")
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Step {step_num} failed: {error_msg}")
                
                execution_log.append({
                    "step": step_num,
                    "tool": tool_name,
                    "description": description,
                    "success": False,
                    "error": error_msg
                })
                
                # Continue execution even if one step fails
                step_results[step_num] = {"error": error_msg}
        
        return {
            "step_results": step_results,
            "execution_log": execution_log,
            "success_count": len([log for log in execution_log if log["success"]]),
            "total_steps": len(execution_log)
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
            
            # Compile final response
            response = {
                "user_request": user_request,
                "execution_plan": plan,
                "execution_results": execution,
                "final_results": execution["step_results"],
                "success_rate": f"{execution['success_count']}/{execution['total_steps']}",
                "status": "completed"
            }
            
            logger.info(f"Request execution completed with {execution['success_count']}/{execution['total_steps']} successful steps")
            
            return response
            
        except Exception as e:
            logger.error(f"Request execution failed: {e}")
            return {
                "user_request": user_request,
                "error": str(e),
                "status": "failed"
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
            if self._client_context:
                await self._client_context.__aexit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Convenience function for generic MCP agent usage
async def generic_mcp_request(
    user_request: str, 
    server_command: Optional[str] = None,
    config: Optional[InvestmentAgentConfig] = None
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
