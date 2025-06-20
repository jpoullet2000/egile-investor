"""
Configuration classes for the Egile Investor package.

These classes provide type-safe configuration management
following Azure best practices for investment analysis automation.
"""

import os
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv


@dataclass
class AzureOpenAIConfig:
    """Configuration for Azure OpenAI client."""

    endpoint: str
    api_version: str = "2024-12-01-preview"
    api_key: Optional[str] = None
    key_vault_url: Optional[str] = None
    api_key_secret_name: Optional[str] = None
    default_model: str = "gpt-4.1-mini"
    max_retries: int = 3
    timeout: int = 30
    use_managed_identity: bool = True

    @classmethod
    def from_environment(cls, env_file: Optional[str] = None) -> "AzureOpenAIConfig":
        """Load configuration from environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to load from .env file in current directory
            load_dotenv()

        return cls(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            key_vault_url=os.getenv("AZURE_KEY_VAULT_URL"),
            api_key_secret_name=os.getenv("AZURE_OPENAI_API_KEY_SECRET_NAME"),
            default_model=os.getenv("AZURE_OPENAI_DEFAULT_MODEL", "gpt-4"),
            max_retries=int(os.getenv("AZURE_OPENAI_MAX_RETRIES", "3")),
            timeout=int(os.getenv("AZURE_OPENAI_TIMEOUT", "30")),
            use_managed_identity=os.getenv("AZURE_USE_MANAGED_IDENTITY", "true").lower()
            == "true",
        )


class InvestmentAgentConfig(BaseModel):
    """Configuration for the Investment Agent."""

    # Agent identification
    name: str = "EgileInvestor"
    description: str = "AI-powered investment analysis and research automation"

    # Investment focus and preferences
    investment_focus: List[str] = Field(
        default=["stocks", "etfs", "mutual_funds"],
        description="Types of investments to focus on",
    )

    risk_tolerance: str = Field(
        default="moderate",
        description="Risk tolerance level: conservative, moderate, aggressive",
    )

    investment_horizon: str = Field(
        default="long_term",
        description="Investment horizon: short_term, medium_term, long_term",
    )

    max_positions: int = Field(
        default=20, description="Maximum number of positions in analysis"
    )

    # Data and analysis configuration
    default_analysis_period: int = Field(
        default=252,
        description="Default analysis period in trading days (252 = 1 year)",
    )

    max_stocks_per_analysis: int = Field(
        default=10,
        description="Maximum number of stocks to analyze in a single request",
    )

    cache_duration_minutes: int = Field(
        default=15, description="Duration to cache market data in minutes"
    )

    # Data sources configuration
    data_sources: List[str] = Field(
        default=["yahoo", "alpha_vantage", "finnhub"],
        description="Preferred data sources for market data",
    )

    enable_real_time_data: bool = Field(
        default=True, description="Whether to use real-time data when available"
    )

    # API Keys and credentials
    alpha_vantage_api_key: Optional[str] = Field(
        default=None, description="Alpha Vantage API key"
    )

    finnhub_api_key: Optional[str] = Field(default=None, description="Finnhub API key")

    quandl_api_key: Optional[str] = Field(default=None, description="Quandl API key")

    # Analysis preferences
    technical_indicators: List[str] = Field(
        default=["SMA", "EMA", "RSI", "MACD", "BB"],
        description="Technical indicators to include in analysis",
    )

    fundamental_metrics: List[str] = Field(
        default=["PE", "PB", "ROE", "ROA", "EPS_growth", "Revenue_growth"],
        description="Fundamental metrics to analyze",
    )

    # Output and reporting
    include_charts: bool = Field(
        default=True, description="Whether to include charts in analysis reports"
    )

    report_format: str = Field(
        default="comprehensive",
        description="Report format: brief, comprehensive, detailed",
    )

    max_report_length: int = Field(
        default=5000, description="Maximum length of generated reports in characters"
    )

    # OpenAI configuration
    openai_config: AzureOpenAIConfig

    @classmethod
    def from_environment(
        cls, env_file: Optional[str] = None
    ) -> "InvestmentAgentConfig":
        """Load configuration from environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        return cls(
            name=os.getenv("INVESTMENT_AGENT_NAME", "EgileInvestor"),
            investment_focus=os.getenv(
                "INVESTMENT_FOCUS", "stocks,etfs,mutual_funds"
            ).split(","),
            risk_tolerance=os.getenv("RISK_TOLERANCE", "moderate"),
            investment_horizon=os.getenv("INVESTMENT_HORIZON", "long_term"),
            max_positions=int(os.getenv("MAX_POSITIONS", "20")),
            default_analysis_period=int(os.getenv("DEFAULT_ANALYSIS_PERIOD", "252")),
            max_stocks_per_analysis=int(os.getenv("MAX_STOCKS_PER_ANALYSIS", "10")),
            cache_duration_minutes=int(os.getenv("CACHE_DURATION_MINUTES", "15")),
            data_sources=os.getenv("DATA_SOURCES", "yahoo,alpha_vantage,finnhub").split(
                ","
            ),
            enable_real_time_data=os.getenv("ENABLE_REAL_TIME_DATA", "true").lower()
            == "true",
            alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
            finnhub_api_key=os.getenv("FINNHUB_API_KEY"),
            quandl_api_key=os.getenv("QUANDL_API_KEY"),
            technical_indicators=os.getenv(
                "TECHNICAL_INDICATORS", "SMA,EMA,RSI,MACD,BB"
            ).split(","),
            fundamental_metrics=os.getenv(
                "FUNDAMENTAL_METRICS", "PE,PB,ROE,ROA,EPS_growth,Revenue_growth"
            ).split(","),
            include_charts=os.getenv("INCLUDE_CHARTS", "true").lower() == "true",
            report_format=os.getenv("REPORT_FORMAT", "comprehensive"),
            max_report_length=int(os.getenv("MAX_REPORT_LENGTH", "5000")),
            openai_config=AzureOpenAIConfig.from_environment(env_file),
        )

    @validator("risk_tolerance")
    def validate_risk_tolerance(cls, v):
        """Validate risk tolerance values."""
        valid_values = ["conservative", "moderate", "aggressive"]
        if v not in valid_values:
            raise ValueError(f"Risk tolerance must be one of: {valid_values}")
        return v

    @validator("investment_horizon")
    def validate_investment_horizon(cls, v):
        """Validate investment horizon values."""
        valid_values = ["short_term", "medium_term", "long_term"]
        if v not in valid_values:
            raise ValueError(f"Investment horizon must be one of: {valid_values}")
        return v

    @validator("report_format")
    def validate_report_format(cls, v):
        """Validate report format values."""
        valid_values = ["brief", "comprehensive", "detailed"]
        if v not in valid_values:
            raise ValueError(f"Report format must be one of: {valid_values}")
        return v


class MarketDataConfig(BaseModel):
    """Configuration for market data providers."""

    default_provider: str = Field(
        default="yahoo", description="Default market data provider"
    )

    fallback_providers: List[str] = Field(
        default=["alpha_vantage", "finnhub"],
        description="Fallback providers if default fails",
    )

    rate_limit_per_minute: int = Field(
        default=60, description="Rate limit for API calls per minute"
    )

    timeout_seconds: int = Field(
        default=30, description="Timeout for API calls in seconds"
    )

    retry_attempts: int = Field(
        default=3, description="Number of retry attempts for failed requests"
    )


class PortfolioConfig(BaseModel):
    """Configuration for portfolio analysis."""

    benchmark_symbol: str = Field(
        default="SPY", description="Benchmark symbol for portfolio comparison"
    )

    risk_free_rate: float = Field(
        default=0.02, description="Risk-free rate for calculations (annual)"
    )

    rebalancing_frequency: str = Field(
        default="quarterly", description="Portfolio rebalancing frequency"
    )

    min_position_size: float = Field(
        default=0.01, description="Minimum position size as percentage of portfolio"
    )

    max_position_size: float = Field(
        default=0.10, description="Maximum position size as percentage of portfolio"
    )


class ScreeningConfig(BaseModel):
    """Configuration for stock screening."""

    universe: str = Field(
        default="SP500",
        description="Stock universe for screening (SP500, NASDAQ, NYSE, etc.)",
    )

    min_market_cap: float = Field(
        default=1000000000,  # 1B
        description="Minimum market capitalization",
    )

    min_daily_volume: int = Field(
        default=100000, description="Minimum average daily volume"
    )

    max_pe_ratio: float = Field(default=50.0, description="Maximum P/E ratio")

    min_roe: float = Field(default=0.10, description="Minimum Return on Equity")

    sectors_to_exclude: List[str] = Field(
        default=[], description="Sectors to exclude from screening"
    )
