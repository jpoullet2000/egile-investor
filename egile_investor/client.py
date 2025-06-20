"""
Azure OpenAI client for the Egile Investor package.

This module provides a client for interacting with Azure OpenAI services
for investment analysis and AI-powered decision making.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import structlog
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
from openai import AsyncAzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import AzureOpenAIConfig
from .exceptions import AzureOpenAIError


logger = structlog.get_logger(__name__)


class AzureOpenAIClient:
    """
    Client for Azure OpenAI services with investment analysis capabilities.

    Features:
    - Automatic credential management via Azure Key Vault
    - Retry logic with exponential backoff
    - Investment-specific prompt templates
    - Token usage tracking
    - Error handling and logging
    """

    def __init__(self, config: AzureOpenAIConfig):
        """
        Initialize the Azure OpenAI client.

        Args:
            config: Azure OpenAI configuration
        """
        self.config = config
        self.client: Optional[AsyncAzureOpenAI] = None
        self.api_key: Optional[str] = None

        # Token usage tracking
        self.total_tokens_used = 0
        self.total_cost = 0.0

        logger.info(
            "Initializing Azure OpenAI client",
            endpoint=config.endpoint,
            model=config.default_model,
            use_managed_identity=config.use_managed_identity,
        )

    async def _get_api_key(self) -> str:
        """Get API key from direct config, Key Vault, or environment."""
        if self.api_key:
            return self.api_key

        # First try direct API key from config
        if self.config.api_key:
            self.api_key = self.config.api_key
            logger.info("Using API key from direct configuration")
            return self.api_key

        # Fall back to Key Vault if configured
        if not self.config.key_vault_url or not self.config.api_key_secret_name:
            raise AzureOpenAIError(
                "API key must be provided either directly (AZURE_OPENAI_API_KEY) "
                "or through Key Vault (AZURE_KEY_VAULT_URL and AZURE_OPENAI_API_KEY_SECRET_NAME)"
            )

        try:
            if self.config.use_managed_identity:
                credential = ManagedIdentityCredential()
            else:
                credential = DefaultAzureCredential()

            secret_client = SecretClient(
                vault_url=self.config.key_vault_url, credential=credential
            )

            secret = secret_client.get_secret(self.config.api_key_secret_name)
            self.api_key = secret.value

            logger.info("Successfully retrieved API key from Key Vault")
            return self.api_key

        except Exception as e:
            logger.error(f"Failed to retrieve API key from Key Vault: {e}")
            raise AzureOpenAIError(f"Failed to get API key: {e}")

    async def _initialize_client(self):
        """Initialize the OpenAI client."""
        if self.client:
            return

        try:
            api_key = await self._get_api_key()

            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=api_key,
                api_version=self.config.api_version,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )

            logger.info("Azure OpenAI client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise AzureOpenAIError(f"Client initialization failed: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs,
    ) -> Any:
        """
        Create a chat completion with retry logic.

        Args:
            messages: List of messages for the conversation
            model: Model to use (defaults to config default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            **kwargs: Additional arguments

        Returns:
            Chat completion response
        """
        await self._initialize_client()

        model = model or self.config.default_model

        try:
            logger.debug(
                "Making chat completion request",
                model=model,
                messages_count=len(messages),
                temperature=temperature,
            )

            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                **kwargs,
            )

            # Track token usage
            if hasattr(response, "usage") and response.usage:
                tokens_used = response.usage.total_tokens
                self.total_tokens_used += tokens_used

                # Rough cost estimation (this would need to be updated with actual pricing)
                estimated_cost = tokens_used * 0.00003  # $0.03 per 1K tokens estimate
                self.total_cost += estimated_cost

                logger.debug(
                    "Chat completion successful",
                    tokens_used=tokens_used,
                    total_tokens=self.total_tokens_used,
                    estimated_cost=estimated_cost,
                )

            return response

        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise AzureOpenAIError(f"Chat completion failed: {e}")

    async def investment_analysis(
        self,
        analysis_request: str,
        context_data: Optional[Dict[str, Any]] = None,
        analysis_type: str = "comprehensive",
    ) -> str:
        """
        Perform investment analysis using AI.

        Args:
            analysis_request: The investment analysis request
            context_data: Additional context data (market data, financials, etc.)
            analysis_type: Type of analysis (brief, comprehensive, detailed)

        Returns:
            Investment analysis response
        """
        system_prompt = self._get_investment_analysis_system_prompt(analysis_type)

        user_prompt = self._format_investment_analysis_prompt(
            analysis_request, context_data
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self.chat_completion(
            messages=messages,
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=2000 if analysis_type == "comprehensive" else 1000,
        )

        return response.choices[0].message.content

    def _get_investment_analysis_system_prompt(self, analysis_type: str) -> str:
        """Get system prompt for investment analysis."""
        base_prompt = """You are an expert investment analyst with deep knowledge of financial markets, 
        technical analysis, fundamental analysis, and investment strategies. You provide clear, 
        actionable investment insights based on data and established financial principles.

        Your analysis should be:
        - Data-driven and objective
        - Clear about assumptions and limitations
        - Include both opportunities and risks
        - Provide specific, actionable recommendations
        - Consider multiple timeframes when relevant
        - Use proper financial terminology
        """

        if analysis_type == "brief":
            return (
                base_prompt
                + "\n\nProvide a concise analysis focusing on key points and clear recommendations."
            )
        elif analysis_type == "detailed":
            return (
                base_prompt
                + "\n\nProvide a thorough, detailed analysis with comprehensive explanations of all factors considered."
            )
        else:  # comprehensive
            return (
                base_prompt
                + "\n\nProvide a balanced, comprehensive analysis covering all important aspects."
            )

    def _format_investment_analysis_prompt(
        self, request: str, context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format the investment analysis prompt with context data."""
        prompt = f"Investment Analysis Request: {request}\n\n"

        if context_data:
            prompt += "Context Data:\n"
            for key, value in context_data.items():
                if isinstance(value, (dict, list)):
                    prompt += f"{key}: {str(value)[:500]}...\n"  # Truncate large data
                else:
                    prompt += f"{key}: {value}\n"
            prompt += "\n"

        prompt += """Please provide a comprehensive investment analysis including:
        1. Executive Summary
        2. Current Market Context
        3. Technical Analysis (if applicable)
        4. Fundamental Analysis (if applicable)
        5. Risk Assessment
        6. Investment Recommendation
        7. Key Considerations and Limitations
        """

        return prompt

    async def generate_investment_report(
        self,
        title: str,
        data: Dict[str, Any],
        format_type: str = "markdown",
    ) -> str:
        """
        Generate a formatted investment report.

        Args:
            title: Report title
            data: Analysis data to include in report
            format_type: Output format (markdown, html, text)

        Returns:
            Formatted investment report
        """
        system_prompt = f"""You are an expert financial report writer. Create a professional 
        investment report in {format_type} format. The report should be well-structured, 
        clear, and include appropriate formatting for readability."""

        user_prompt = f"""Create an investment report with the following:
        
        Title: {title}
        
        Data to include:
        {str(data)[:2000]}...
        
        The report should include:
        - Executive Summary
        - Key Findings
        - Detailed Analysis
        - Recommendations
        - Risk Factors
        - Conclusion
        
        Format the report professionally in {format_type} format.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self.chat_completion(
            messages=messages,
            temperature=0.4,
            max_tokens=3000,
        )

        return response.choices[0].message.content

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get client usage statistics."""
        return {
            "total_tokens_used": self.total_tokens_used,
            "estimated_total_cost": self.total_cost,
            "model": self.config.default_model,
            "endpoint": self.config.endpoint,
        }

    async def close(self):
        """Close the client connection."""
        if self.client:
            await self.client.close()
            self.client = None
            logger.info("Azure OpenAI client closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
