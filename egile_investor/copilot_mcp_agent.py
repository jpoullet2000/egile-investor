"""
GitHub Copilot-style AI Agent that works with any MCP server.

This agent provides a conversational interface similar to GitHub Copilot,
maintaining context across interactions and providing helpful suggestions.
"""

from typing import Any, Dict, List, Optional, Union
import json
import re
import structlog
from datetime import datetime, date
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
        self.server_command = (
            server_command or "python -m egile_investor.server_standard"
        )
        self.agent_name = agent_name
        self.session: Optional[ClientSession] = None
        self.available_tools: Dict[str, Tool] = {}
        self._read_stream = None
        self._write_stream = None
        self._client_context = None
        self.openai_client = AzureOpenAIClient(self.config.openai_config)

        # Date context for analysis
        self.current_date = date(2025, 6, 26)  # June 26, 2025
        self.current_year = 2025
        self.current_quarter = "Q2 2025"

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
1. ALWAYS validate stock symbols first if the query contains symbols (use validate_symbol tool)
2. If a symbol is invalid, suggest corrections and ask user to confirm
3. Use the most appropriate tools for valid symbols
4. Build on previous conversation context when relevant
5. Provide comprehensive but not overwhelming information
6. Follow logical dependencies between steps

IMPORTANT: If you detect common symbol mistakes (AMZ→AMZN, APPL→AAPL, etc.), include a validation step first.

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

        system_prompt = f"""You are {self.agent_name}, a knowledgeable financial AI assistant providing data-driven investment insights.

IMPORTANT: The analysis results contain REAL financial data from live market sources (Yahoo Finance, etc.). 
Always use the specific numbers, prices, and metrics provided in the insights below. Never say you can't access real data.

Be conversational, insightful, and actionable. Structure your response like GitHub Copilot:
- Start with a direct answer using REAL data from the insights
- Provide key insights with actual numbers (prices, ratios, percentages)
- Offer specific, actionable recommendations based on the data
- Suggest follow-up questions or next steps
- Use clear formatting with bullets, numbers, or sections when helpful

Recent conversation:
{recent_context}

Execution summary:
- Success rate: {execution.get("success_count", 0)}/{execution.get("total_steps", 0)}
- Steps completed: {len(execution.get("execution_log", []))}

REAL MARKET DATA AND INSIGHTS:
{insights}

The data above contains real, current market information. Use it to provide specific, data-driven responses.
Make the response valuable and encourage further interaction."""

        user_prompt = f"""User asked: "{user_input}"

Based on the REAL market data analysis results above, provide a comprehensive but conversational response that:
1. Directly addresses their question using specific data points (prices, ratios, etc.)
2. Highlights the most important findings with actual numbers
3. Provides actionable insights based on the real financial metrics
4. Suggests logical next steps

Remember: Use the specific financial data provided - current prices, P/E ratios, market caps, recommendations, etc. 
This is real, live market data, not simulated information.

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

            generated_response = response.choices[0].message.content.strip()

            # Fallback check: If the AI still claims it can't access real data despite having it,
            # append the insights directly to ensure data is shared
            if (
                any(
                    phrase in generated_response.lower()
                    for phrase in [
                        "can't pull live data",
                        "can't access real",
                        "check a reliable financial platform",
                        "please check yahoo finance",
                        "since i can't pull",
                        "i don't have access to real-time",
                    ]
                )
                and insights
                and "Analysis completed successfully" not in insights
            ):
                fallback_data = f"\n\n**Real Market Data Retrieved:**\n{insights}\n\nThe data above was retrieved from live financial sources (Yahoo Finance). Please use this current information for your analysis."
                generated_response = generated_response + fallback_data

            return generated_response

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"I completed the analysis with {execution.get('success_count', 0)} successful steps. Here are the key findings:\n\n{insights}\n\nWould you like me to elaborate on any specific aspect?"

    def _extract_insights(self, execution: Dict[str, Any]) -> str:
        """Extract key insights from execution results for response generation."""
        insights = []
        step_results = execution.get("step_results", {})

        for step_num, result in step_results.items():
            if isinstance(result, dict):
                # Extract stock analysis insights
                if "symbol" in result and "stock_data" in result:
                    symbol = result["symbol"]
                    stock_data = result.get("stock_data", {})
                    info = stock_data.get("info", {})

                    # Extract current price
                    current_price = info.get("currentPrice") or info.get(
                        "regularMarketPrice"
                    )
                    if current_price:
                        insights.append(
                            f"• {symbol} Current Price: ${current_price:.2f}"
                        )

                    # Extract company name
                    company_name = info.get("longName") or info.get("shortName")
                    if company_name:
                        insights.append(f"• Company: {company_name}")

                    # Extract market cap
                    market_cap = info.get("marketCap")
                    if market_cap:
                        market_cap_b = market_cap / 1e9
                        insights.append(f"• Market Cap: ${market_cap_b:.1f}B")

                    # Extract P/E ratio
                    pe_ratio = info.get("trailingPE") or info.get("forwardPE")
                    if pe_ratio:
                        insights.append(f"• P/E Ratio: {pe_ratio:.2f}")

                    # Extract recommendation from analysis
                    overall_assessment = result.get("overall_assessment", {})
                    if overall_assessment:
                        recommendation = overall_assessment.get("recommendation")
                        confidence = overall_assessment.get("confidence")
                        risk_level = overall_assessment.get("risk_level")

                        if recommendation:
                            insights.append(f"• Recommendation: {recommendation}")
                        if confidence:
                            insights.append(f"• Confidence: {confidence}")
                        if risk_level:
                            insights.append(f"• Risk Level: {risk_level}")

                    # Extract technical analysis
                    tech_analysis = result.get("technical_analysis", {})
                    if tech_analysis:
                        rsi = tech_analysis.get("rsi")
                        if rsi:
                            insights.append(f"• RSI: {rsi:.1f}")

                        sma = tech_analysis.get("sma_50")
                        if sma:
                            insights.append(f"• 50-day SMA: ${sma:.2f}")

                    # Extract fundamental analysis
                    fund_analysis = result.get("fundamental_analysis", {})
                    if fund_analysis:
                        roe = fund_analysis.get("roe")
                        if roe:
                            insights.append(f"• ROE: {roe:.2%}")

                        debt_ratio = fund_analysis.get("debt_to_equity")
                        if debt_ratio:
                            insights.append(f"• Debt/Equity: {debt_ratio:.2f}")

                # Extract market data insights with better price handling
                elif "symbol" in result and (
                    "historical_data" in result or "info" in result
                ):
                    symbol = result["symbol"]
                    info = result.get("info", {})

                    # Try multiple price sources for current price
                    current_price = None
                    price_source = None

                    # Method 1: Try info fields
                    if info.get("currentPrice"):
                        current_price = info["currentPrice"]
                        price_source = "current"
                    elif info.get("regularMarketPrice"):
                        current_price = info["regularMarketPrice"]
                        price_source = "regular_market"
                    elif info.get("previousClose"):
                        current_price = info["previousClose"]
                        price_source = "previous_close"

                    # Method 2: Try historical data if info failed
                    if not current_price and "historical_data" in result:
                        hist_data = result["historical_data"]
                        close_prices = hist_data.get("Close", {})
                        if close_prices:
                            latest_date = max(close_prices.keys())
                            current_price = close_prices[latest_date]
                            price_source = f"historical ({latest_date[:10]})"

                    if current_price:
                        insights.append(
                            f"• {symbol} Current Price: ${current_price:.2f} (source: {price_source})"
                        )

                    # Extract company name
                    company_name = info.get("longName") or info.get("shortName")
                    if company_name:
                        insights.append(f"• Company: {company_name}")

                    # Extract other key data
                    if info.get("marketCap"):
                        market_cap_b = info["marketCap"] / 1e9
                        insights.append(f"• Market Cap: ${market_cap_b:.1f}B")

                    # Extract change information
                    change = info.get("regularMarketChange")
                    change_pct = info.get("regularMarketChangePercent")
                    if change and change_pct:
                        insights.append(
                            f"• Daily Change: ${change:.2f} ({change_pct:.2%})"
                        )

                # Extract screening results
                elif isinstance(result, list) and result and "symbol" in result[0]:
                    symbols = []
                    scores = []
                    for item in result[:5]:
                        if isinstance(item, dict):
                            symbol = item.get("symbol")
                            score = item.get("score")
                            if symbol:
                                symbols.append(symbol)
                            if score is not None:
                                scores.append(f"{symbol}: {score:.2f}")

                    if symbols:
                        insights.append(
                            f"• Top Screening Results: {', '.join(symbols)}"
                        )
                    if scores:
                        insights.append(f"• Screening Scores: {', '.join(scores[:3])}")

                # Generic patterns
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
            # Validate and fix arguments based on tool schema
            fixed_arguments = self._validate_and_fix_arguments(tool_name, arguments)

            response = await self.session.call_tool(tool_name, fixed_arguments)
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

    def _validate_and_fix_arguments(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and fix arguments against tool schema to prevent type mismatches.

        Args:
            tool_name: Name of the tool
            arguments: Arguments to validate and fix

        Returns:
            Fixed arguments that match the tool schema
        """
        if tool_name not in self.available_tools:
            return arguments

        try:
            tool = self.available_tools[tool_name]
            if not hasattr(tool, "inputSchema") or not tool.inputSchema:
                return arguments

            schema = tool.inputSchema
            properties = schema.get("properties", {})
            fixed_args = {}

            for param_name, param_info in properties.items():
                # Handle complex schemas with anyOf, oneOf etc.
                param_type = self._extract_param_type(param_info)

                if param_name in arguments:
                    value = arguments[param_name]

                    # Check for empty critical parameters before type conversion
                    critical_params = ["symbol", "symbols", "ticker", "stock_symbol"]
                    if param_name.lower() in critical_params:
                        if (
                            (isinstance(value, list) and len(value) == 0)
                            or value == ""
                            or value is None
                        ):
                            logger.error(
                                f"Critical parameter '{param_name}' is empty for tool '{tool_name}'"
                            )
                            # Don't include this parameter - let validation fail properly
                            continue

                    # Fix common type mismatches
                    if param_type == "string" and isinstance(value, list):
                        # If expecting string but got list, take first item or join
                        if len(value) == 1:
                            fixed_args[param_name] = str(value[0])
                            logger.debug(
                                f"Fixed {param_name}: converted list {value} to string '{value[0]}'"
                            )
                        elif len(value) > 1:
                            # For multiple items, we might need to call tool multiple times
                            # For now, take the first one and log warning
                            fixed_args[param_name] = str(value[0])
                            logger.warning(
                                f"Tool '{tool_name}' expects single {param_type} for '{param_name}', but got list {value}. Using first item: {value[0]}"
                            )
                        else:
                            # Empty list - don't set to empty string for critical parameters
                            critical_params = [
                                "symbol",
                                "symbols",
                                "ticker",
                                "stock_symbol",
                            ]
                            if param_name.lower() in critical_params:
                                logger.error(
                                    f"Critical parameter '{param_name}' has empty list for tool '{tool_name}'"
                                )
                                # Don't set it - let validation fail properly
                                continue
                            else:
                                fixed_args[param_name] = ""

                    elif param_type == "array" and not isinstance(value, list):
                        # If expecting array but got single value, wrap in list
                        fixed_args[param_name] = [value]
                        logger.debug(f"Fixed {param_name}: wrapped '{value}' in list")

                    elif param_type == "number" and isinstance(value, str):
                        # Try to convert string to number
                        try:
                            if "." in value:
                                fixed_args[param_name] = float(value)
                            else:
                                fixed_args[param_name] = int(value)
                            logger.debug(
                                f"Fixed {param_name}: converted string '{value}' to number"
                            )
                        except ValueError:
                            fixed_args[param_name] = (
                                value  # Keep original if conversion fails
                            )

                    elif param_type == "boolean" and isinstance(value, str):
                        # Convert string to boolean
                        fixed_args[param_name] = value.lower() in [
                            "true",
                            "1",
                            "yes",
                            "on",
                        ]
                        logger.debug(
                            f"Fixed {param_name}: converted string '{value}' to boolean"
                        )

                    else:
                        # No conversion needed
                        fixed_args[param_name] = value
                else:
                    # Parameter not provided, check if it's required
                    required_params = schema.get("required", [])
                    if param_name in required_params:
                        # Don't provide defaults for critical parameters - let the tool validation fail
                        # This prevents issues like empty symbol names
                        critical_params = [
                            "symbol",
                            "symbols",
                            "ticker",
                            "stock_symbol",
                        ]
                        if param_name.lower() in critical_params:
                            logger.error(
                                f"Critical required parameter '{param_name}' missing for tool '{tool_name}'"
                            )
                            # Don't set a default - let it fail with proper error
                            continue

                        # Provide reasonable defaults only for non-critical parameters
                        if param_type == "string":
                            fixed_args[param_name] = ""
                        elif param_type == "array":
                            fixed_args[param_name] = []
                        elif param_type == "number":
                            fixed_args[param_name] = 0
                        elif param_type == "boolean":
                            fixed_args[param_name] = False
                        logger.warning(
                            f"Required parameter '{param_name}' missing for tool '{tool_name}', using default"
                        )

            # Only add arguments that are defined in the schema or are known to be valid
            # This prevents passing unexpected parameters that would cause validation errors
            for key, value in arguments.items():
                if key not in fixed_args and key in properties:
                    # Check for empty critical parameters again before adding
                    critical_params = ["symbol", "symbols", "ticker", "stock_symbol"]
                    if key.lower() in critical_params:
                        if (
                            (isinstance(value, list) and len(value) == 0)
                            or value == ""
                            or value is None
                        ):
                            logger.debug(
                                f"Skipping empty critical parameter '{key}' for tool '{tool_name}'"
                            )
                            continue

                    # Only pass through arguments that are actually defined in the schema
                    fixed_args[key] = value
                elif key not in fixed_args and key not in properties:
                    # Log when we're filtering out unexpected parameters
                    logger.debug(
                        f"Filtering out unexpected parameter '{key}' for tool '{tool_name}'"
                    )

            return fixed_args

        except Exception as e:
            logger.error(f"Error validating arguments for tool '{tool_name}': {e}")
            return arguments  # Return original if validation fails

    def _extract_param_type(self, param_info: Dict[str, Any]) -> str:
        """
        Extract the actual parameter type from complex schema definitions.

        Args:
            param_info: Parameter information from the schema

        Returns:
            The primary type expected for this parameter
        """
        # Direct type
        if "type" in param_info:
            return param_info["type"]

        # Handle anyOf, oneOf patterns
        if "anyOf" in param_info:
            # Look for the main type (non-null)
            for option in param_info["anyOf"]:
                if isinstance(option, dict) and "type" in option:
                    if option["type"] != "null":
                        return option["type"]

        if "oneOf" in param_info:
            # Look for the main type (non-null)
            for option in param_info["oneOf"]:
                if isinstance(option, dict) and "type" in option:
                    if option["type"] != "null":
                        return option["type"]

        # Default fallback
        return "string"

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

                # Check if we need to handle multiple items for single-item tools
                needs_multi_call = self._needs_multi_call(tool_name, resolved_args)

                if needs_multi_call:
                    # Handle tools that expect single items when we have multiple
                    results = await self._handle_multi_item_tools(
                        tool_name, resolved_args
                    )
                    result = results  # Store all results
                else:
                    # Validate and fix arguments based on tool schema
                    validated_args = self._validate_and_fix_arguments(
                        tool_name, resolved_args
                    )
                    result = await self.call_tool(tool_name, validated_args)
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
                    step_ref = int(step_match.group(1))
                    if step_ref in step_results:
                        # Replace the placeholder with actual result
                        step_result = step_results[step_ref]
                        if isinstance(step_result, dict) and "symbol" in step_result:
                            resolved[key] = step_result["symbol"]
                        else:
                            resolved[key] = str(step_result)
                    else:
                        logger.warning(f"Step {step_ref} not found for reference")
                        resolved[key] = value
                else:
                    resolved[key] = value
            else:
                resolved[key] = value

        return resolved

    async def _direct_query_with_tools(self, query: str) -> str:
        """Handle direct tool queries using Azure OpenAI."""
        try:
            tool_list = self._format_tools_for_llm()

            # Include date context in the system prompt
            date_context = f"Current date: {self.current_date.strftime('%B %d, %Y')} ({self.current_quarter})"

            system_prompt = f"""You are an investment analysis AI assistant with access to real-time financial tools.

{date_context}

Available tools:
{tool_list}

Your task is to answer the user's investment question using the available tools. 
Be conversational, helpful, and provide actionable insights.

When analyzing stocks or markets:
- Always consider the current date context for relevance
- Mention when data is current as of {self.current_date.strftime("%B %Y")}
- Provide context about the current market year ({self.current_year})

If you need to use a tool, describe what you're doing and why.
Provide clear, actionable recommendations based on the analysis results."""

            response = await self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=1000,
                temperature=0.7,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Direct query failed: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"

    async def _execute_with_tools(self, query: str) -> str:
        """Execute a query that requires tool usage with date awareness."""
        try:
            # Create the plan
            plan = await self._create_plan(query)

            if not plan or "steps" not in plan:
                return await self._direct_query_with_tools(query)

            # Execute the plan
            execution_results = await self._execute_plan(plan["steps"])

            # Generate final response with date context
            date_context = f"Analysis performed on {self.current_date.strftime('%B %d, %Y')} during {self.current_quarter}"

            response_prompt = f"""Based on the following execution results, provide a comprehensive answer to the user's question.

Current Context: {date_context}

User Question: {query}

Execution Results:
{json.dumps(execution_results, indent=2, default=str)}

Provide a clear, conversational response that:
1. Directly answers the user's question
2. Includes relevant findings from the tool executions
3. Mentions the current date context when relevant
4. Provides actionable investment insights
5. Acknowledges any limitations or errors that occurred

Be helpful and conversational, not just a data dump."""

            response = await self.client.chat_completion(
                messages=[{"role": "user", "content": response_prompt}],
                max_tokens=1500,
                temperature=0.7,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"I encountered an error while analyzing your request: {str(e)}"

    async def _handle_multi_item_tools(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> List[Any]:
        """
        Handle tools that expect single items when we have multiple items to process.

        Args:
            tool_name: Name of the tool
            arguments: Arguments that may contain lists

        Returns:
            List of results from multiple tool calls
        """
        if tool_name not in self.available_tools:
            return []

        tool = self.available_tools[tool_name]
        if not hasattr(tool, "inputSchema") or not tool.inputSchema:
            # No schema info, try single call
            try:
                result = await self.call_tool(tool_name, arguments)
                return [result]
            except Exception as e:
                logger.error(f"Error calling tool '{tool_name}': {e}")
                return []

        schema = tool.inputSchema
        properties = schema.get("properties", {})
        results = []

        # Find parameters that have lists but tool expects strings
        multi_params = {}
        single_params = {}

        for param_name, value in arguments.items():
            if param_name in properties:
                param_type = self._extract_param_type(properties[param_name])
                if param_type == "string" and isinstance(value, list):
                    multi_params[param_name] = value
                else:
                    single_params[param_name] = value
            else:
                single_params[param_name] = value

        if multi_params:
            # We have parameters with lists that tool expects as strings
            # Call tool multiple times, once for each combination

            # For simplicity, handle the most common case: one list parameter
            if len(multi_params) == 1:
                param_name, values = next(iter(multi_params.items()))

                for value in values[:5]:  # Limit to 5 calls to avoid overwhelming
                    call_args = single_params.copy()
                    call_args[param_name] = value

                    try:
                        result = await self.call_tool(tool_name, call_args)
                        results.append(result)
                        logger.debug(
                            f"Multi-call for {tool_name} with {param_name}={value} succeeded"
                        )
                    except Exception as e:
                        logger.error(
                            f"Multi-call for {tool_name} with {param_name}={value} failed: {e}"
                        )
                        results.append({"error": str(e), "symbol": value})
            else:
                # Multiple list parameters - more complex, for now just take first values
                call_args = single_params.copy()
                for param_name, values in multi_params.items():
                    call_args[param_name] = values[0] if values else ""

                try:
                    result = await self.call_tool(tool_name, call_args)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Multi-param call for {tool_name} failed: {e}")
                    results.append({"error": str(e)})
        else:
            # No multi-params, single call
            try:
                result = await self.call_tool(tool_name, arguments)
                results.append(result)
            except Exception as e:
                logger.error(f"Single call for {tool_name} failed: {e}")
                results.append({"error": str(e)})

        return results

    def _needs_multi_call(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """
        Check if a tool call needs to be split into multiple calls due to type mismatches.

        Args:
            tool_name: Name of the tool
            arguments: Arguments to check

        Returns:
            True if multiple calls are needed
        """
        if tool_name not in self.available_tools:
            return False

        tool = self.available_tools[tool_name]
        if not hasattr(tool, "inputSchema") or not tool.inputSchema:
            return False

        schema = tool.inputSchema
        properties = schema.get("properties", {})

        for param_name, value in arguments.items():
            if param_name in properties:
                param_type = self._extract_param_type(properties[param_name])
                if (
                    param_type == "string"
                    and isinstance(value, list)
                    and len(value) > 1
                ):
                    return True

        return False

    def get_date_context(self) -> Dict[str, Any]:
        """Get current date context for the agent."""
        return {
            "current_date": self.current_date.isoformat(),
            "current_year": self.current_year,
            "current_quarter": self.current_quarter,
            "analysis_context": f"Analysis performed on {self.current_date.strftime('%B %d, %Y')}",
            "market_year": self.current_year,
            "ytd_period": f"January 1, {self.current_year} to {self.current_date.strftime('%B %d, %Y')}",
        }

    async def __aenter__(self):
        """Async context manager entry - connect to MCP server."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - disconnect from MCP server."""
        await self.disconnect()
        return False  # Don't suppress exceptions
