"""
GitHub Copilot-style AI Agent that works with any MCP server.

This agent provides a conversational interface similar to GitHub Copilot,
maintaining context across interactions and providing helpful suggestions.
"""

from typing import Any, Dict, List, Optional, Union
import json
import re
import structlog
from datetime import datetime
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Tool

from .config import InvestmentAgentConfig, AzureOpenAIConfig
from .client import AzureOpenAIClient


logger = structlog.get_logger(__name__)


class CopilotMCPAgent:
    """
    A GitHub Copilot-style AI agent that provides conversational interactions
    with any MCP server, maintaining context and providing intelligent suggestions.
    """

    def __init__(
        self,
        config: Optional[InvestmentAgentConfig] = None,
        server_command: Optional[str] = None,
        agent_name: str = "Copilot",
    ):
        """
        Initialize the Copilot-style MCP agent.

        Args:
            config: Configuration for the OpenAI client
            server_command: Command to start the MCP server
            agent_name: Name for the agent (default: "Copilot")
        """
        self.config = config or InvestmentAgentConfig(
            openai_config=AzureOpenAIConfig.from_environment()
        )
        self.server_command = server_command or "python -m egile_investor.server"
        self.agent_name = agent_name
        self.session: Optional[ClientSession] = None
        self.available_tools: Dict[str, Tool] = {}
        self._read_stream = None
        self._write_stream = None
        self._client_context = None
        self.openai_client = AzureOpenAIClient(self.config.openai_config)

        # Conversation state
        self.conversation_history: List[Dict[str, Any]] = []
        self.context_memory: Dict[str, Any] = {}
        self.last_execution_results: Optional[Dict[str, Any]] = None
        self.user_preferences: Dict[str, Any] = {}

    async def connect(self):
        """Connect to the MCP server and discover available tools."""
        try:
            # Clean up any existing connections first
            await self.disconnect()
            
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
                f"🤖 {self.agent_name} connected with {len(self.available_tools)} tools available"
            )

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            await self.disconnect()  # Clean up on failure
            raise

    async def disconnect(self):
        """Disconnect from the MCP server and clean up resources."""
        try:
            if self.session:
                try:
                    await self.session.__aexit__(None, None, None)
                except Exception as e:
                    logger.debug(f"Session cleanup error (non-critical): {e}")
                finally:
                    self.session = None

            if self._client_context:
                try:
                    await self._client_context.__aexit__(None, None, None)
                except Exception as e:
                    logger.debug(f"Client context cleanup error (non-critical): {e}")
                finally:
                    self._client_context = None

            # Reset streams
            self._read_stream = None
            self._write_stream = None
            
            logger.debug(f"🤖 {self.agent_name} disconnected")

        except Exception as e:
            logger.error(f"Disconnect error: {e}")
            # Force cleanup even if there were errors
            self.session = None
            self._client_context = None
            self._read_stream = None
            self._write_stream = None

    async def shutdown(self):
        """Gracefully shutdown the agent and clean up all resources."""
        logger.info(f"🤖 {self.agent_name} shutting down...")
        await self.disconnect()
        logger.info(f"🤖 {self.agent_name} shutdown complete")

    async def chat(self, user_input: str, stream: bool = False) -> Union[str, Any]:
        """
        Main chat interface - responds to user input in a conversational manner.

        Args:
            user_input: User's message/question
            stream: Whether to stream the response (for future implementation)

        Returns:
            Conversational response string
        """
        if not self.session:
            try:
                await self.connect()
            except Exception as e:
                return f"I'm having trouble connecting to my tools. Error: {str(e)}. Please try again in a moment."

        # Add user input to conversation history
        self.conversation_history.append(
            {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat(),
            }
        )

        try:
            # Determine the type of response needed
            response_strategy = await self._determine_response_strategy(user_input)

            if response_strategy["type"] == "direct_answer":
                # Answer directly without tool execution
                response = await self._generate_direct_response(user_input)

            elif response_strategy["type"] == "tool_execution":
                # Execute tools and provide results
                response = await self._execute_and_respond(
                    user_input, response_strategy
                )

            elif response_strategy["type"] == "clarification":
                # Ask for clarification or suggest options
                response = await self._generate_clarification_response(
                    user_input, response_strategy
                )

            elif response_strategy["type"] == "follow_up":
                # Handle follow-up questions based on previous context
                response = await self._handle_follow_up(user_input)

            else:
                # Default response
                response = await self._generate_helpful_response(user_input)

            # Add response to conversation history
            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat(),
                    "strategy": response_strategy["type"],
                }
            )

            # Limit conversation history to last 20 exchanges
            if len(self.conversation_history) > 40:
                self.conversation_history = self.conversation_history[-40:]

            return response

        except Exception as e:
            error_response = f"I encountered an error: {str(e)}. Let me try a different approach or ask me to clarify what you need."
            logger.error(f"Chat error: {e}")
            return error_response

    async def _determine_response_strategy(self, user_input: str) -> Dict[str, Any]:
        """
        Determine the best strategy for responding to user input.
        Uses LLM to classify the intent and suggest approach.
        """
        # Get recent conversation context
        recent_context = self._get_recent_context()

        # Check if this is a follow-up question
        is_follow_up = len(self.conversation_history) > 0 and any(
            keyword in user_input.lower()
            for keyword in [
                "that",
                "this",
                "them",
                "it",
                "those",
                "these",
                "also",
                "too",
                "more",
            ]
        )

        system_prompt = f"""You are {self.agent_name}, a helpful AI assistant with access to various tools. 
Analyze the user's input and determine the best response strategy.

Available tools: {", ".join(self.available_tools.keys())}

Recent conversation context:
{recent_context}

User has access to tools for: investment analysis, stock screening, market data, financial analysis, etc.

Response strategies:
1. "direct_answer" - Answer immediately without tools (for general questions, explanations, definitions)
2. "tool_execution" - Execute tools to gather data and provide comprehensive response
3. "clarification" - Ask for clarification or suggest specific options
4. "follow_up" - Handle follow-up questions using previous context
5. "helpful_suggestion" - Provide helpful suggestions or alternatives

Consider:
- Is this a follow-up question referring to previous results?
- Does this require real data/analysis or just explanation?
- Is the request clear enough to execute?
- What would be most helpful to the user?

Respond with JSON:
{{
    "type": "strategy_name",
    "confidence": "high/medium/low", 
    "reasoning": "brief explanation",
    "suggested_tools": ["tool1", "tool2"],
    "needs_clarification": ["what specific info might be needed"]
}}"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User input: '{user_input}'"},
            ]

            response = await self.openai_client.chat_completion(
                messages=messages,
                model=self.config.openai_config.default_model,
                temperature=0.3,
                max_tokens=500,
            )

            strategy_text = response.choices[0].message.content.strip()

            # Extract JSON
            json_match = re.search(r"\{.*\}", strategy_text, re.DOTALL)
            if json_match:
                strategy = json.loads(json_match.group(0))
            else:
                # Fallback strategy
                strategy = {
                    "type": "tool_execution" if is_follow_up else "direct_answer",
                    "confidence": "medium",
                    "reasoning": "Fallback strategy",
                    "suggested_tools": [],
                    "needs_clarification": [],
                }

            logger.debug(
                f"Response strategy: {strategy['type']} (confidence: {strategy['confidence']})"
            )
            return strategy

        except Exception as e:
            logger.error(f"Strategy determination failed: {e}")
            return {
                "type": "helpful_suggestion",
                "confidence": "low",
                "reasoning": f"Error in strategy determination: {e}",
                "suggested_tools": [],
                "needs_clarification": [],
            }

    async def _generate_direct_response(self, user_input: str) -> str:
        """Generate a direct response without tool execution."""
        recent_context = self._get_recent_context()

        system_prompt = f"""You are {self.agent_name}, a knowledgeable AI assistant specializing in investment and financial analysis.

Provide a helpful, conversational response to the user's question. Be:
- Friendly and approachable (like GitHub Copilot)
- Informative but concise
- Encouraging about what you can help with
- Specific about your capabilities

Recent conversation:
{recent_context}

Available capabilities: stock analysis, market data, portfolio analysis, screening, sentiment analysis, risk assessment, financial ratios, technical analysis, and report generation.

If the user might benefit from data analysis, gently suggest specific things you can help with."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        response = await self.openai_client.chat_completion(
            messages=messages,
            model=self.config.openai_config.default_model,
            temperature=0.7,
            max_tokens=800,
        )

        return response.choices[0].message.content.strip()

    async def _execute_and_respond(
        self, user_input: str, strategy: Dict[str, Any]
    ) -> str:
        """Execute tools and provide a comprehensive response."""
        # Create execution plan
        plan = await self._create_smart_execution_plan(user_input, strategy)

        # Execute the plan
        execution = await self._execute_plan(plan)
        self.last_execution_results = execution

        # Store relevant context
        self._update_context_memory(user_input, execution)

        # Generate conversational response with results
        response = await self._generate_conversational_response(user_input, execution)

        return response

    async def _create_smart_execution_plan(
        self, user_input: str, strategy: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create an intelligent execution plan based on user input and strategy."""
        tools_info = self._format_tools_for_llm()
        recent_context = self._get_recent_context()

        system_prompt = f"""You are {self.agent_name}, creating an execution plan for the user's request.

Available tools:
{tools_info}

Recent conversation context:
{recent_context}

Strategy analysis: {strategy.get("reasoning", "No specific strategy")}
Suggested tools: {strategy.get("suggested_tools", [])}

Create a logical, efficient plan that:
1. Addresses the user's specific needs
2. Uses the most appropriate tools
3. Builds on previous conversation context when relevant
4. Provides comprehensive but not overwhelming information
5. Follows logical dependencies between steps

Output format:
[
  {{
    "step": 1,
    "tool": "tool_name",
    "description": "Human-readable description",
    "arguments": {{"param": "value"}},
    "reasoning": "Why this step is needed"
  }}
]

Keep plans concise but thorough - typically 1-4 steps."""

        user_prompt = f"""Create an execution plan for: "{user_input}"

Consider:
- What specific information would be most valuable?
- How can I build on previous conversation context?
- What's the most efficient way to get comprehensive results?

Execution plan:"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await self.openai_client.chat_completion(
                messages=messages,
                model=self.config.openai_config.default_model,
                temperature=0.4,
                max_tokens=1200,
            )

            plan_text = response.choices[0].message.content.strip()

            # Extract JSON
            json_match = re.search(r"\[.*\]", plan_text, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group(0))
            else:
                # Fallback to simple plan
                plan = self._create_fallback_plan(user_input, strategy)

            logger.info(f"Created execution plan with {len(plan)} steps")
            return plan

        except Exception as e:
            logger.error(f"Plan creation failed: {e}")
            return self._create_fallback_plan(user_input, strategy)

    def _create_fallback_plan(
        self, user_input: str, strategy: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create a simple fallback plan when smart planning fails."""
        suggested_tools = strategy.get("suggested_tools", [])

        if suggested_tools:
            return [
                {
                    "step": 1,
                    "tool": suggested_tools[0],
                    "description": f"Execute {suggested_tools[0]} for user request",
                    "arguments": {"query": user_input},
                    "reasoning": "Fallback execution",
                }
            ]
        elif self.available_tools:
            first_tool = list(self.available_tools.keys())[0]
            return [
                {
                    "step": 1,
                    "tool": first_tool,
                    "description": f"Execute {first_tool} for user request",
                    "arguments": {"query": user_input},
                    "reasoning": "Fallback to first available tool",
                }
            ]
        else:
            return []

    async def _generate_conversational_response(
        self, user_input: str, execution: Dict[str, Any]
    ) -> str:
        """Generate a conversational response incorporating execution results."""
        # Extract key insights from execution results
        insights = self._extract_insights(execution)
        recent_context = self._get_recent_context()

        system_prompt = f"""You are {self.agent_name}, providing a helpful response to the user based on analysis results.

Be conversational, insightful, and actionable. Structure your response like GitHub Copilot:
- Start with a direct answer/summary
- Provide key insights and findings  
- Offer specific, actionable recommendations
- Suggest follow-up questions or next steps
- Use clear formatting with bullets, numbers, or sections when helpful

Recent conversation:
{recent_context}

Execution summary:
- Success rate: {execution.get("success_count", 0)}/{execution.get("total_steps", 0)}
- Steps completed: {len(execution.get("execution_log", []))}

Key insights from analysis:
{insights}

Make the response valuable and encourage further interaction."""

        user_prompt = f"""User asked: "{user_input}"

Based on the analysis results, provide a comprehensive but conversational response that:
1. Directly addresses their question
2. Highlights the most important findings
3. Provides actionable insights
4. Suggests logical next steps

Response:"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await self.openai_client.chat_completion(
                messages=messages,
                model=self.config.openai_config.default_model,
                temperature=0.6,
                max_tokens=1500,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"I completed the analysis with {execution.get('success_count', 0)} successful steps. Here are the key findings:\n\n{insights}\n\nWould you like me to elaborate on any specific aspect?"

    def _extract_insights(self, execution: Dict[str, Any]) -> str:
        """Extract key insights from execution results for response generation."""
        insights = []
        step_results = execution.get("step_results", {})

        for step_num, result in step_results.items():
            if isinstance(result, dict):
                # Look for key information patterns
                if "symbol" in result:
                    insights.append(f"• Analyzed {result['symbol']}")
                elif "symbols" in result:
                    symbols = result["symbols"]
                    if isinstance(symbols, list):
                        insights.append(
                            f"• Found {len(symbols)} symbols: {', '.join(symbols[:3])}{'...' if len(symbols) > 3 else ''}"
                        )
                elif "recommendation" in result:
                    insights.append(f"• Recommendation: {result['recommendation']}")
                elif "score" in result:
                    insights.append(f"• Score: {result['score']}")
                elif "error" in result:
                    insights.append(f"• Step {step_num} encountered an issue")
                elif len(str(result)) > 100:
                    insights.append(f"• Step {step_num}: Generated detailed analysis")

            elif isinstance(result, list) and result:
                if len(result) > 0 and isinstance(result[0], dict):
                    if "symbol" in result[0]:
                        symbols = [item.get("symbol", "Unknown") for item in result[:5]]
                        insights.append(
                            f"• Found {len(result)} stocks: {', '.join(symbols)}"
                        )
                    else:
                        insights.append(f"• Generated {len(result)} analysis results")
                else:
                    insights.append(f"• Generated {len(result)} results")
            elif isinstance(result, str) and len(result) > 50:
                insights.append("• Generated detailed report/analysis")

        return "\n".join(insights) if insights else "Analysis completed successfully"

    async def _generate_clarification_response(
        self, user_input: str, strategy: Dict[str, Any]
    ) -> str:
        """Generate a response asking for clarification or suggesting options."""
        needs_clarification = strategy.get("needs_clarification", [])
        available_capabilities = list(self.available_tools.keys())

        clarification_prompt = f"""The user asked: "{user_input}"

I can help with: {", ".join(available_capabilities)}

Areas that need clarification: {", ".join(needs_clarification)}

Generate a helpful response that:
1. Acknowledges their request
2. Explains what specific information would help
3. Suggests 2-3 concrete options they could choose from
4. Maintains an encouraging, helpful tone

Be specific about what I can do for them."""

        try:
            response = await self.openai_client.chat_completion(
                messages=[{"role": "user", "content": clarification_prompt}],
                model=self.config.openai_config.default_model,
                temperature=0.7,
                max_tokens=600,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Clarification response failed: {e}")
            return f"I'd be happy to help with that! To give you the best response, could you clarify:\n\n{chr(10).join(f'• {item}' for item in needs_clarification)}\n\nI can help with {', '.join(available_capabilities[:3])} and more."

    async def _handle_follow_up(self, user_input: str) -> str:
        """Handle follow-up questions using previous context."""
        if not self.last_execution_results:
            return "I don't have previous results to reference. Could you ask your question with more context?"

        # Use previous results to answer follow-up
        previous_insights = self._extract_insights(self.last_execution_results)
        recent_context = self._get_recent_context()

        follow_up_prompt = f"""The user is asking a follow-up question: "{user_input}"

Previous analysis insights:
{previous_insights}

Recent conversation:
{recent_context}

Previous execution had {self.last_execution_results.get("success_count", 0)} successful steps.

Provide a helpful response that:
1. References the previous analysis appropriately
2. Answers their follow-up question
3. Offers additional insights if relevant
4. Suggests next steps if appropriate

Be conversational and build on what we've already discussed."""

        try:
            response = await self.openai_client.chat_completion(
                messages=[{"role": "user", "content": follow_up_prompt}],
                model=self.config.openai_config.default_model,
                temperature=0.6,
                max_tokens=800,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Follow-up handling failed: {e}")
            return f"Based on the previous analysis:\n\n{previous_insights}\n\nCould you clarify what specific aspect you'd like me to elaborate on?"

    async def _generate_helpful_response(self, user_input: str) -> str:
        """Generate a helpful response when other strategies don't apply."""
        capabilities = list(self.available_tools.keys())

        helpful_prompt = f"""The user said: "{user_input}"

I'm {self.agent_name}, an AI assistant that can help with: {", ".join(capabilities[:5])}{"..." if len(capabilities) > 5 else ""}

Generate a helpful, encouraging response that:
1. Acknowledges what they said
2. Explains how I might be able to help
3. Gives 2-3 specific examples of what I can do
4. Asks an engaging follow-up question

Be friendly and specific about my capabilities."""

        try:
            response = await self.openai_client.chat_completion(
                messages=[{"role": "user", "content": helpful_prompt}],
                model=self.config.openai_config.default_model,
                temperature=0.7,
                max_tokens=600,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Helpful response generation failed: {e}")
            return f"I'm here to help! I can assist with {', '.join(capabilities[:3])} and more. What would you like to explore?"

    def _get_recent_context(self, max_exchanges: int = 3) -> str:
        """Get recent conversation context for LLM prompts."""
        if not self.conversation_history:
            return "No previous conversation"

        recent = self.conversation_history[-(max_exchanges * 2) :]
        context_lines = []

        for msg in recent:
            role = "User" if msg["role"] == "user" else self.agent_name
            content = (
                msg["content"][:200] + "..."
                if len(msg["content"]) > 200
                else msg["content"]
            )
            context_lines.append(f"{role}: {content}")

        return "\n".join(context_lines)

    def _update_context_memory(self, user_input: str, execution: Dict[str, Any]):
        """Update context memory with relevant information from the interaction."""
        # Store key information that might be referenced later
        if execution.get("step_results"):
            for step_num, result in execution["step_results"].items():
                if isinstance(result, list) and result:
                    # Store symbol lists
                    if isinstance(result[0], dict) and "symbol" in result[0]:
                        symbols = [item.get("symbol") for item in result[:10]]
                        self.context_memory["recent_symbols"] = symbols

                elif isinstance(result, dict):
                    if "symbol" in result:
                        self.context_memory["last_analyzed_symbol"] = result["symbol"]

        # Store user intent/topic
        self.context_memory["last_topic"] = user_input.lower()
        self.context_memory["last_execution_time"] = datetime.now().isoformat()

    # Include tool execution methods from GenericMCPAgent
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
                            return json.loads(first_content.text)
                        except (json.JSONDecodeError, ValueError):
                            return first_content.text
                    else:
                        return first_content
                elif isinstance(response.content, list):
                    return []
                else:
                    return response.content
            else:
                return []

        except Exception as e:
            logger.error(f"Failed to call tool '{tool_name}': {e}")
            raise

    def _format_tools_for_llm(self) -> str:
        """Format available tools for LLM understanding."""
        tool_descriptions = []

        for name, tool in self.available_tools.items():
            description = tool.description or "No description available"

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

    async def _execute_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute the plan step by step with proper error handling."""
        step_results = {}
        execution_log = []

        for step in plan:
            step_num = step["step"]
            tool_name = step["tool"]
            description = step["description"]
            arguments = step.get("arguments", {})

            logger.info(f"Executing step {step_num}: {description}")

            try:
                resolved_args = self._resolve_step_references(arguments, step_results)

                if tool_name not in self.available_tools:
                    raise ValueError(f"Tool '{tool_name}' not available")

                result = await self.call_tool(tool_name, resolved_args)
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

                step_results[step_num] = {"error": error_msg}

        return {
            "step_results": step_results,
            "execution_log": execution_log,
            "success_count": len([log for log in execution_log if log["success"]]),
            "total_steps": len(execution_log),
        }

    def _resolve_step_references(
        self, arguments: Dict[str, Any], step_results: Dict[int, Any]
    ) -> Dict[str, Any]:
        """Resolve references to previous step results in arguments."""
        resolved = {}

        for key, value in arguments.items():
            if isinstance(value, str) and "{result_from_step_" in value:
                step_match = re.search(r"result_from_step_(\d+)", value)
                if step_match:
                    step_num = int(step_match.group(1))
                    if step_num in step_results:
                        result = step_results[step_num]
                        if isinstance(result, list) and key == "screening_results":
                            try:
                                resolved[key] = json.dumps(result)
                            except (TypeError, ValueError):
                                resolved[key] = str(result)
                        else:
                            resolved[key] = result
                    else:
                        logger.warning(
                            f"Referenced step {step_num} not found in results"
                        )
                        resolved[key] = None
                else:
                    resolved[key] = value
            elif isinstance(value, dict):
                resolved[key] = self._resolve_step_references(value, step_results)
            elif isinstance(value, list):
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

    # Convenience methods for easier interaction
    async def ask(self, question: str) -> str:
        """Alias for chat method."""
        return await self.chat(question)

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history."""
        return self.conversation_history.copy()

    def clear_conversation(self):
        """Clear the conversation history and context."""
        self.conversation_history.clear()
        self.context_memory.clear()
        self.last_execution_results = None
        logger.info("Conversation history cleared")

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.available_tools.keys())

    async def suggest_next_steps(self) -> str:
        """Suggest logical next steps based on recent conversation."""
        if not self.conversation_history:
            return "Start by asking me about stocks, market analysis, or investment research!"

        recent_context = self._get_recent_context()

        suggestion_prompt = f"""Based on this recent conversation with the user:

{recent_context}

Suggest 2-3 logical next steps or follow-up questions that would be valuable for the user.
Be specific and actionable. Format as a brief, encouraging message with bullet points."""

        try:
            response = await self.openai_client.chat_completion(
                messages=[{"role": "user", "content": suggestion_prompt}],
                temperature=0.7,
                max_tokens=400,
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Suggestion generation failed: {e}")
            return "Feel free to ask me about any investment analysis, stock research, or market insights you'd like to explore!"

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with improved cleanup."""
        await self.disconnect()


# Convenience function for creating a copilot session
async def create_copilot_session(
    server_command: Optional[str] = None,
    config: Optional[InvestmentAgentConfig] = None,
    agent_name: str = "Copilot",
) -> CopilotMCPAgent:
    """
    Create and connect a Copilot MCP Agent session.

    Args:
        server_command: Command to start the MCP server
        config: Configuration for the OpenAI client
        agent_name: Name for the agent

    Returns:
        Connected CopilotMCPAgent instance
    """
    agent = CopilotMCPAgent(
        config=config, server_command=server_command, agent_name=agent_name
    )
    await agent.connect()
    return agent


# Convenience function for one-off Copilot interactions
async def copilot_chat(
    user_input: str,
    server_command: Optional[str] = None,
    config: Optional[InvestmentAgentConfig] = None,
    agent_name: str = "Copilot",
) -> str:
    """
    Have a one-off chat interaction with the Copilot agent.
    
    Args:
        user_input: The user's message/question
        server_command: Command to start the MCP server
        config: Configuration for the OpenAI client
        agent_name: Name for the agent
        
    Returns:
        Response from the agent
    """
    async with CopilotMCPAgent(
        config=config, server_command=server_command, agent_name=agent_name
    ) as agent:
        return await agent.chat(user_input)
