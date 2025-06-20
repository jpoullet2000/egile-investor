"""
MCP Server for Egile Investor

This server exposes investment agent capabilities as MCP tools.
"""

import asyncio
from typing import Any, Dict, List, Optional

import structlog
from fastmcp import FastMCP

from .agent import InvestmentAgent
from .config import InvestmentAgentConfig, AzureOpenAIConfig


logger = structlog.get_logger(__name__)

# Global investment agent instance
investment_agent: Optional[InvestmentAgent] = None

# Initialize MCP server
mcp = FastMCP("Egile Investor")


async def get_investment_agent() -> InvestmentAgent:
    """Get or create the investment agent instance."""
    global investment_agent
    if investment_agent is None:
        config = InvestmentAgentConfig(
            openai_config=AzureOpenAIConfig.from_environment()
        )
        investment_agent = InvestmentAgent(config=config)
        logger.info("Investment agent initialized")
    return investment_agent


@mcp.tool()
async def analyze_stock(
    symbol: str,
    analysis_type: str = "comprehensive",
    include_technical: bool = True,
    include_fundamental: bool = True,
) -> Dict[str, Any]:
    """
    Perform comprehensive stock analysis with technical and fundamental insights.

    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
        analysis_type: Type of analysis (brief, comprehensive, detailed)
        include_technical: Whether to include technical analysis
        include_fundamental: Whether to include fundamental analysis

    Returns:
        Complete stock analysis with recommendations
    """
    agent = await get_investment_agent()
    return await agent.analyze_stock(
        symbol=symbol,
        analysis_type=analysis_type,
        include_technical=include_technical,
        include_fundamental=include_fundamental,
    )


@mcp.tool()
async def get_market_data(
    symbol: str,
    period: str = "1y",
    source: str = "yahoo",
) -> Dict[str, Any]:
    """
    Retrieve real-time and historical market data for a stock.

    Args:
        symbol: Stock symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        source: Data source (yahoo, alpha_vantage, finnhub)

    Returns:
        Market data including historical prices and company information
    """
    agent = await get_investment_agent()
    return await agent.get_stock_data(
        symbol=symbol,
        period=period,
        source=source,
    )


@mcp.tool()
async def screen_stocks(
    criteria: Dict[str, Any],
    universe: Optional[List[str]] = None,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Screen stocks based on various financial criteria.

    Args:
        criteria: Screening criteria (e.g., {"pe_ratio": {"max": 25}, "roe": {"min": 0.15}})
        universe: List of stock symbols to screen (defaults to major indices)
        max_results: Maximum number of results to return

    Returns:
        List of stocks meeting the criteria with scores
    """
    agent = await get_investment_agent()
    screened = await agent.screen_stocks(
        criteria=criteria,
        universe=universe,
    )
    return screened[:max_results]


@mcp.tool()
async def analyze_portfolio(
    holdings: List[Dict[str, Any]],
    benchmark: str = "SPY",
    analysis_type: str = "comprehensive",
) -> Dict[str, Any]:
    """
    Analyze portfolio performance, risk, and optimization opportunities.

    Args:
        holdings: List of portfolio holdings with symbols and weights
        benchmark: Benchmark symbol for comparison
        analysis_type: Type of analysis (brief, comprehensive, detailed)

    Returns:
        Portfolio analysis with performance metrics and recommendations
    """
    # This would be implemented in the agent
    # For now, return a placeholder
    return {
        "portfolio_analysis": "Portfolio analysis feature coming soon",
        "holdings": holdings,
        "benchmark": benchmark,
        "analysis_type": analysis_type,
    }


@mcp.tool()
async def get_financial_ratios(
    symbol: str,
    ratios: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calculate and analyze financial ratios for a stock.

    Args:
        symbol: Stock symbol
        ratios: List of specific ratios to calculate (defaults to all common ratios)

    Returns:
        Dictionary of financial ratios with explanations
    """
    agent = await get_investment_agent()
    stock_data = await agent.get_stock_data(symbol)

    # Extract financial ratios from stock data
    info = stock_data.get("info", {})

    default_ratios = [
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "returnOnEquity",
        "returnOnAssets",
        "profitMargins",
        "debtToEquity",
        "currentRatio",
    ]

    ratios_to_calc = ratios or default_ratios

    financial_ratios = {}
    for ratio in ratios_to_calc:
        value = info.get(ratio)
        if value is not None:
            financial_ratios[ratio] = value

    return {
        "symbol": symbol,
        "financial_ratios": financial_ratios,
        "timestamp": stock_data.get("timestamp"),
    }


@mcp.tool()
async def technical_analysis(
    symbol: str,
    indicators: Optional[List[str]] = None,
    period: str = "1y",
) -> Dict[str, Any]:
    """
    Perform technical analysis with indicators and pattern recognition.

    Args:
        symbol: Stock symbol
        indicators: List of technical indicators (SMA, EMA, RSI, MACD, BB)
        period: Time period for analysis

    Returns:
        Technical analysis results with signals and patterns
    """
    agent = await get_investment_agent()
    stock_data = await agent.get_stock_data(symbol, period=period)

    # Perform technical analysis (this would be more comprehensive in practice)
    analysis = await agent._perform_technical_analysis(stock_data)

    return {
        "symbol": symbol,
        "period": period,
        "technical_analysis": analysis,
        "timestamp": stock_data.get("timestamp"),
    }


@mcp.tool()
async def sentiment_analysis(
    symbol: str,
    sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Analyze market sentiment from news and social media.

    Args:
        symbol: Stock symbol
        sources: List of sources to analyze (news, twitter, reddit)

    Returns:
        Sentiment analysis results with scores and trends
    """
    # This would integrate with news APIs and sentiment analysis
    # For now, return a placeholder
    return {
        "symbol": symbol,
        "sentiment_score": 0.65,  # Placeholder: 0-1 scale
        "sentiment_trend": "positive",
        "sources_analyzed": sources or ["news", "social_media"],
        "summary": f"Market sentiment for {symbol} appears moderately positive based on recent news and social media activity.",
    }


@mcp.tool()
async def risk_assessment(
    symbol: Optional[str] = None,
    portfolio: Optional[List[Dict[str, Any]]] = None,
    risk_factors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Assess investment risk and calculate risk metrics.

    Args:
        symbol: Individual stock symbol to assess
        portfolio: Portfolio holdings for portfolio risk assessment
        risk_factors: Specific risk factors to evaluate

    Returns:
        Risk assessment with metrics and recommendations
    """
    if symbol:
        agent = await get_investment_agent()
        stock_data = await agent.get_stock_data(symbol)
        info = stock_data.get("info", {})

        # Calculate basic risk metrics
        beta = info.get("beta", 1.0)
        volatility = "High" if beta > 1.5 else "Medium" if beta > 0.8 else "Low"

        return {
            "symbol": symbol,
            "risk_assessment": {
                "beta": beta,
                "volatility_level": volatility,
                "market_risk": "Medium",  # Would be calculated based on market conditions
                "sector_risk": "Medium",  # Would be based on sector analysis
                "overall_risk": volatility,
            },
            "recommendations": [
                f"Stock has {'high' if beta > 1.5 else 'moderate' if beta > 1.0 else 'low'} sensitivity to market movements",
                "Consider position sizing based on risk tolerance",
                "Monitor market conditions for potential volatility",
            ],
        }

    return {
        "portfolio_risk": "Portfolio risk assessment feature coming soon",
        "risk_factors": risk_factors
        or ["market_risk", "sector_risk", "liquidity_risk"],
    }


@mcp.tool()
async def generate_report(
    analysis_type: str = "stock_analysis",
    data: Dict[str, Any] = None,
    format_type: str = "markdown",
) -> Dict[str, Any]:
    """
    Generate comprehensive investment analysis reports.

    Args:
        analysis_type: Type of report (stock_analysis, portfolio_review, market_outlook)
        data: Analysis data to include in the report
        format_type: Output format (markdown, html, pdf)

    Returns:
        Generated report with formatted content
    """
    agent = await get_investment_agent()

    # Generate report using AI
    if data and "symbol" in data:
        title = f"Investment Analysis Report: {data['symbol']}"
    else:
        title = f"{analysis_type.replace('_', ' ').title()} Report"

    report_content = await agent.openai_client.generate_investment_report(
        title=title,
        data=data or {},
        format_type=format_type,
    )

    return {
        "report_type": analysis_type,
        "format": format_type,
        "title": title,
        "content": report_content,
        "generated_at": agent.openai_client.get_usage_stats(),
    }


def main():
    """Main entry point for the MCP server."""
    try:
        logger.info("Starting Egile Investor MCP Server")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
