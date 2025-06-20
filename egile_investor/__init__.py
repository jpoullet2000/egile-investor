"""
Egile Investor: AI-powered investment analysis and stock research automation.

This package provides intelligent investment research capabilities through an AI agent
that can analyze stocks, market trends, financial data, and provide investment recommendations.
"""

from .ai_agent import AIInvestmentAgent, ai_investment_analysis
from .agent import InvestmentAgent
from .config import InvestmentAgentConfig, AzureOpenAIConfig
from .client import AzureOpenAIClient
from .exceptions import InvestmentAgentError, AzureOpenAIError

__version__ = "0.1.0"
__author__ = "Jean-Baptiste Poullet"
__email__ = "jeanbaptistepoullet@gmail.com"

__all__ = [
    "AIInvestmentAgent",
    "ai_investment_analysis",
    "InvestmentAgent",
    "InvestmentAgentConfig",
    "AzureOpenAIConfig",
    "AzureOpenAIClient",
    "InvestmentAgentError",
    "AzureOpenAIError",
]
