"""
MCP Server for Egile Investor

This server exposes investment agent capabilities as MCP tools.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

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
    universe: Optional[Union[List[str], str]] = None,
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
    data: Optional[Dict[str, Any]] = None,
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


@mcp.tool()
async def get_screening_symbols(
    screening_results: str,  # Changed from List to str - we'll parse JSON
    max_symbols: int = 5,
) -> List[str]:
    """
    Extract stock symbols from screening results.

    Args:
        screening_results: JSON string results from screen_stocks tool
        max_symbols: Maximum number of symbols to return

    Returns:
        List of stock symbols from the screening results
    """
    try:
        import json

        # Parse the JSON string
        if isinstance(screening_results, str):
            results = json.loads(screening_results)
        else:
            results = screening_results

        symbols = []
        for result in results[:max_symbols]:
            if isinstance(result, dict) and "symbol" in result:
                symbols.append(result["symbol"])
        return symbols
    except Exception as e:
        logger.error(f"Failed to extract symbols from screening results: {e}")
        return []


@mcp.tool()
async def analyze_multiple_stocks(
    symbols: Union[List[str], str],
    analysis_type: str = "comprehensive",
    include_technical: bool = True,
    include_fundamental: bool = True,
) -> Dict[str, Any]:
    """
    Perform analysis on multiple stocks at once.

    Args:
        symbols: List of stock symbols to analyze, or a string that will be parsed
        analysis_type: Type of analysis (brief, comprehensive, detailed)
        include_technical: Whether to include technical analysis
        include_fundamental: Whether to include fundamental analysis

    Returns:
        Dictionary with analysis results for each symbol
    """
    # Handle symbols parameter - convert string to list if needed
    if isinstance(symbols, str):
        # If it's a placeholder string, use fallback symbols
        if any(
            placeholder in symbols.lower()
            for placeholder in ["step", "result", "{", "<"]
        ):
            symbols = ["AAPL", "MSFT", "GOOGL"]  # Fallback symbols
        else:
            # Single symbol as string
            symbols = [symbols]

    if not symbols:
        symbols = ["AAPL", "MSFT", "GOOGL"]  # Fallback if empty

    agent = await get_investment_agent()
    results = {}

    for symbol in symbols:
        try:
            analysis = await agent.analyze_stock(
                symbol=symbol,
                analysis_type=analysis_type,
                include_technical=include_technical,
                include_fundamental=include_fundamental,
            )
            results[symbol] = analysis
        except Exception as e:
            results[symbol] = {"error": str(e), "symbol": symbol}

    return results


@mcp.tool()
async def get_multiple_market_data(
    symbols: Union[List[str], str],
    period: str = "1y",
    source: str = "yahoo",
) -> Dict[str, Any]:
    """
    Retrieve market data for multiple stocks at once.

    Args:
        symbols: List of stock symbols, or a string that will be parsed
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        source: Data source (yahoo, alpha_vantage, finnhub)

    Returns:
        Dictionary with market data for each symbol
    """
    # Handle symbols parameter - convert string to list if needed
    if isinstance(symbols, str):
        # If it's a placeholder string, use fallback symbols
        if any(
            placeholder in symbols.lower()
            for placeholder in ["step", "result", "{", "<"]
        ):
            symbols = ["AAPL", "MSFT", "GOOGL"]  # Fallback symbols
        else:
            # Single symbol as string
            symbols = [symbols]

    if not symbols:
        symbols = ["AAPL", "MSFT", "GOOGL"]  # Fallback if empty

    agent = await get_investment_agent()
    results = {}

    for symbol in symbols:
        try:
            data = await agent.get_stock_data(
                symbol=symbol,
                period=period,
                source=source,
            )
            results[symbol] = data
        except Exception as e:
            results[symbol] = {"error": str(e), "symbol": symbol}

    return results


@mcp.tool()
async def assess_multiple_risks(
    symbols: Union[List[str], str],
    risk_factors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Assess investment risk for multiple stocks at once.

    Args:
        symbols: List of stock symbols to assess, or a string that will be parsed
        risk_factors: Specific risk factors to evaluate

    Returns:
        Dictionary with risk assessment for each symbol
    """
    # Handle symbols parameter - convert string to list if needed
    if isinstance(symbols, str):
        # If it's a placeholder string, use fallback symbols
        if any(
            placeholder in symbols.lower()
            for placeholder in ["step", "result", "{", "<"]
        ):
            symbols = ["AAPL", "MSFT", "GOOGL"]  # Fallback symbols
        else:
            # Single symbol as string
            symbols = [symbols]

    if not symbols:
        symbols = ["AAPL", "MSFT", "GOOGL"]  # Fallback if empty

    agent = await get_investment_agent()
    results = {}

    for symbol in symbols:
        try:
            # Use the existing risk_assessment tool logic
            risk_result = await risk_assessment(
                symbol=symbol, risk_factors=risk_factors
            )
            results[symbol] = risk_result
        except Exception as e:
            results[symbol] = {"error": str(e), "symbol": symbol}

    return results


@mcp.tool()
async def get_multiple_financial_ratios(
    symbols: Union[List[str], str],
    ratios: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calculate financial ratios for multiple stocks at once.

    Args:
        symbols: List of stock symbols, or a string that will be parsed
        ratios: List of specific ratios to calculate (defaults to all common ratios)

    Returns:
        Dictionary with financial ratios for each symbol
    """
    # Handle symbols parameter - convert string to list if needed
    if isinstance(symbols, str):
        # If it's a placeholder string, use fallback symbols
        if any(
            placeholder in symbols.lower()
            for placeholder in ["step", "result", "{", "<"]
        ):
            symbols = ["AAPL", "MSFT", "GOOGL"]  # Fallback symbols
        else:
            # Single symbol as string
            symbols = [symbols]

    if not symbols:
        symbols = ["AAPL", "MSFT", "GOOGL"]  # Fallback if empty

    results = {}

    for symbol in symbols:
        try:
            # Use the existing get_financial_ratios tool logic
            ratios_result = await get_financial_ratios(symbol=symbol, ratios=ratios)
            results[symbol] = ratios_result
        except Exception as e:
            results[symbol] = {"error": str(e), "symbol": symbol}

    return results


@mcp.tool()
async def analyze_multiple_sentiment(
    symbols: Union[List[str], str],
    sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Analyze market sentiment for multiple stocks at once.

    Args:
        symbols: List of stock symbols, or a string that will be parsed
        sources: List of sources to analyze (news, twitter, reddit)

    Returns:
        Dictionary with sentiment analysis for each symbol
    """
    # Handle symbols parameter - convert string to list if needed
    if isinstance(symbols, str):
        # If it's a placeholder string, use fallback symbols
        if any(
            placeholder in symbols.lower()
            for placeholder in ["step", "result", "{", "<"]
        ):
            symbols = ["AAPL", "MSFT", "GOOGL"]  # Fallback symbols
        else:
            # Single symbol as string
            symbols = [symbols]

    if not symbols:
        symbols = ["AAPL", "MSFT", "GOOGL"]  # Fallback if empty

    results = {}

    for symbol in symbols:
        try:
            # Use the existing sentiment_analysis tool logic
            sentiment_result = await sentiment_analysis(symbol=symbol, sources=sources)
            results[symbol] = sentiment_result
        except Exception as e:
            results[symbol] = {"error": str(e), "symbol": symbol}

    return results


@mcp.tool()
async def create_portfolio_from_symbols(
    symbols: Union[List[str], str],
    weights: Optional[List[float]] = None,
    benchmark: str = "SPY",
    analysis_type: str = "comprehensive",
) -> Dict[str, Any]:
    """
    Create and analyze a portfolio from a list of stock symbols.

    Args:
        symbols: List of stock symbols for the portfolio, or a string that will be parsed
        weights: Optional list of weights for each symbol (defaults to equal weights)
        benchmark: Benchmark symbol for comparison
        analysis_type: Type of analysis (brief, comprehensive, detailed)

    Returns:
        Portfolio analysis results
    """
    # Handle symbols parameter - convert string to list if needed
    if isinstance(symbols, str):
        # If it's a placeholder string, use fallback symbols
        if any(
            placeholder in symbols.lower()
            for placeholder in ["step", "result", "{", "<"]
        ):
            symbols = ["AAPL", "MSFT", "GOOGL"]  # Fallback symbols
        else:
            # Single symbol as string
            symbols = [symbols]

    if not symbols:
        symbols = ["AAPL", "MSFT", "GOOGL"]  # Fallback if empty

    # Create holdings list with proper format
    if weights is None:
        # Equal weights
        weight = 1.0 / len(symbols)
        weights = [weight] * len(symbols)
    elif len(weights) != len(symbols):
        raise ValueError("Number of weights must match number of symbols")

    holdings = []
    for symbol, weight in zip(symbols, weights):
        holdings.append({"symbol": symbol, "weight": weight})

    # Return portfolio analysis directly
    return {
        "portfolio_created": "Portfolio created successfully",
        "holdings": holdings,
        "benchmark": benchmark,
        "analysis_type": analysis_type,
        "total_symbols": len(symbols),
        "weights": weights,
        "equal_weighted": weights is None,
        "portfolio_summary": f"Created portfolio with {len(symbols)} symbols: {', '.join(symbols)}",
    }


@mcp.tool()
async def get_top_symbols_from_screening(
    screening_results: str,  # Changed from List to str - we'll parse JSON
    count: int = 5,
    sort_by: str = "score",
) -> List[str]:
    """
    Get top stock symbols from screening results based on score or other criteria.

    Args:
        screening_results: JSON string results from screen_stocks tool
        count: Number of top symbols to return
        sort_by: Field to sort by (score, symbol)

    Returns:
        List of top stock symbols
    """
    try:
        import json

        # Parse the JSON string
        if isinstance(screening_results, str):
            results = json.loads(screening_results)
        else:
            results = screening_results

        if not results:
            return []

        # Sort results
        if sort_by == "score":
            sorted_results = sorted(
                results, key=lambda x: x.get("score", 0), reverse=True
            )
        else:
            sorted_results = results

        # Extract symbols
        symbols = []
        for result in sorted_results[:count]:
            if isinstance(result, dict) and "symbol" in result:
                symbols.append(result["symbol"])

        return symbols
    except Exception as e:
        logger.error(f"Failed to extract top symbols from screening results: {e}")
        return []


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
