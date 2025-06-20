"""
Custom exceptions for the Egile Investor package.

These exceptions provide specific error handling for investment analysis
and financial data operations.
"""


class InvestmentAgentError(Exception):
    """Base exception for investment agent operations."""

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        """
        Initialize the investment agent error.

        Args:
            message: Error message
            error_code: Optional error code for categorization
            details: Optional additional error details
        """
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class AzureOpenAIError(InvestmentAgentError):
    """Exception for Azure OpenAI operations."""

    pass


class MarketDataError(InvestmentAgentError):
    """Exception for market data operations."""

    pass


class PortfolioAnalysisError(InvestmentAgentError):
    """Exception for portfolio analysis operations."""

    pass


class StockAnalysisError(InvestmentAgentError):
    """Exception for stock analysis operations."""

    pass


class TechnicalAnalysisError(InvestmentAgentError):
    """Exception for technical analysis operations."""

    pass


class FundamentalAnalysisError(InvestmentAgentError):
    """Exception for fundamental analysis operations."""

    pass


class RiskAnalysisError(InvestmentAgentError):
    """Exception for risk analysis operations."""

    pass


class DataProviderError(InvestmentAgentError):
    """Exception for data provider operations."""

    def __init__(self, message: str, provider: str = None, **kwargs):
        """
        Initialize the data provider error.

        Args:
            message: Error message
            provider: Name of the data provider that failed
            **kwargs: Additional arguments for parent class
        """
        super().__init__(message, **kwargs)
        self.provider = provider


class APIRateLimitError(DataProviderError):
    """Exception for API rate limit exceeded."""

    pass


class APIKeyError(DataProviderError):
    """Exception for invalid or missing API key."""

    pass


class InvalidSymbolError(InvestmentAgentError):
    """Exception for invalid stock symbols."""

    def __init__(self, symbol: str, message: str = None, **kwargs):
        """
        Initialize the invalid symbol error.

        Args:
            symbol: The invalid symbol
            message: Optional custom message
            **kwargs: Additional arguments for parent class
        """
        message = message or f"Invalid or unknown symbol: {symbol}"
        super().__init__(message, **kwargs)
        self.symbol = symbol


class InsufficientDataError(InvestmentAgentError):
    """Exception for insufficient data for analysis."""

    pass


class CalculationError(InvestmentAgentError):
    """Exception for calculation errors in financial metrics."""

    pass


class ConfigurationError(InvestmentAgentError):
    """Exception for configuration-related errors."""

    pass


class MCPServerError(InvestmentAgentError):
    """Exception for MCP server operations."""

    pass


class ToolExecutionError(InvestmentAgentError):
    """Exception for tool execution failures."""

    def __init__(self, tool_name: str, message: str, **kwargs):
        """
        Initialize the tool execution error.

        Args:
            tool_name: Name of the tool that failed
            message: Error message
            **kwargs: Additional arguments for parent class
        """
        super().__init__(f"Tool '{tool_name}' failed: {message}", **kwargs)
        self.tool_name = tool_name
