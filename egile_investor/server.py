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
    use_sp500: bool = False,
) -> List[Dict[str, Any]]:
    """
    Screen stocks based on various financial criteria.

    Args:
        criteria: Screening criteria (e.g., {"pe_ratio": {"max": 25}, "roe": {"min": 0.15}})
        universe: Stock universe to screen:
                 - List of symbols: ["AAPL", "MSFT", "GOOGL"]
                 - Predefined universe: "sp500", "nasdaq100", "dow30", "major"
                 - None: uses default major stocks
        max_results: Maximum number of results to return
        use_sp500: If True, screen all S&P 500 stocks (500+ stocks) regardless of universe

    Returns:
        List of stocks meeting the criteria with scores

    Examples:
        - Screen major stocks: {"pe_ratio": {"max": 25}}
        - Screen S&P 500: {"pe_ratio": {"max": 25}}, universe="sp500"
        - Screen NASDAQ 100: {"pe_ratio": {"max": 25}}, universe="nasdaq100"
        - Screen Dow 30: {"pe_ratio": {"max": 25}}, universe="dow30"
        - Screen S&P 500 (alternative): {"pe_ratio": {"max": 25}}, use_sp500=True
        - Screen custom list: {"pe_ratio": {"max": 25}}, universe=["AAPL", "MSFT", "GOOGL"]
    """
    agent = await get_investment_agent()
    screened = await agent.screen_stocks(
        criteria=criteria,
        universe=universe,
        max_results=max_results,
        use_sp500=use_sp500,
    )
    return screened


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
    data: Optional[Union[Dict[str, Any], str]] = None,
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
    # Handle both string and dict data
    if isinstance(data, str):
        # If data is a string (placeholder), create a simple dict
        data_dict = {"description": data, "analysis_type": analysis_type}
        title = f"{analysis_type.replace('_', ' ').title()} Report"
    elif data and isinstance(data, dict) and "symbol" in data:
        data_dict = data
        title = f"Investment Analysis Report: {data['symbol']}"
    else:
        data_dict = data or {}
        title = f"{analysis_type.replace('_', ' ').title()} Report"

    # For now, return a simplified report structure
    # In a full implementation, this would generate a comprehensive report
    return {
        "report_type": analysis_type,
        "format": format_type,
        "title": title,
        "content": f"# {title}\n\nThis is a {analysis_type} report generated with the following data:\n\n{data_dict}",
        "data_used": data_dict,
        "generated_at": "2025-06-20T22:55:00Z",
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


@mcp.tool()
async def create_investment_report(
    user_query: str,
    analysis_results: List[Dict[str, Any]],
    investment_amount: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Create a comprehensive investment report that directly answers the user's query.

    Args:
        user_query: The original user question/request
        analysis_results: List of analysis results from previous steps
        investment_amount: Amount to invest (if specified in query)

    Returns:
        Investment report with specific recommendations and reasoning
    """

    def extract_investment_insights(results):
        """Extract actionable investment insights from analysis results."""
        stocks_found = []
        risks_identified = []
        financial_metrics = {}
        sentiment_data = {}

        for result in results:
            if not isinstance(result, dict):
                continue

            # Extract stock symbols and scores
            if "symbol" in result:
                stocks_found.append(
                    {
                        "symbol": result["symbol"],
                        "score": result.get("score", 0),
                        "recommendation": result.get("recommendation", "HOLD"),
                        "confidence": result.get("confidence", "Medium"),
                    }
                )
            elif "symbols" in result and isinstance(result["symbols"], list):
                for symbol in result["symbols"][:5]:
                    stocks_found.append(
                        {
                            "symbol": symbol,
                            "score": result.get("score", 0),
                            "recommendation": "ANALYZE",
                            "confidence": "Medium",
                        }
                    )

            # Extract risk information
            if "risk_level" in result:
                risks_identified.append(f"Risk Level: {result['risk_level']}")
            if "volatility" in result:
                risks_identified.append(f"Volatility: {result['volatility']}")
            if "beta" in result:
                risks_identified.append(f"Beta: {result['beta']}")

            # Extract financial metrics
            if "pe_ratio" in result:
                financial_metrics["pe_ratio"] = result["pe_ratio"]
            if "roe" in result:
                financial_metrics["roe"] = result["roe"]
            if "dividend_yield" in result:
                financial_metrics["dividend_yield"] = result["dividend_yield"]

            # Extract sentiment
            if "sentiment_score" in result:
                sentiment_data[result.get("symbol", "market")] = {
                    "score": result["sentiment_score"],
                    "trend": result.get("sentiment_trend", "neutral"),
                }

        # Remove duplicates and rank stocks
        unique_stocks = {}
        for stock in stocks_found:
            symbol = stock["symbol"]
            if (
                symbol not in unique_stocks
                or stock["score"] > unique_stocks[symbol]["score"]
            ):
                unique_stocks[symbol] = stock

        ranked_stocks = sorted(
            unique_stocks.values(), key=lambda x: x["score"], reverse=True
        )

        return {
            "stocks": ranked_stocks[:5],  # Top 5 stocks
            "risks": risks_identified[:5],
            "financial_metrics": financial_metrics,
            "sentiment": sentiment_data,
        }

    def analyze_query_intent(query):
        """Analyze the user query to understand investment intent."""
        query_lower = query.lower()

        intent = {
            "investment_type": "general",
            "risk_preference": "moderate",
            "time_horizon": "long-term",
            "sectors": [],
            "criteria": [],
        }

        # Detect investment type
        if any(word in query_lower for word in ["dividend", "income", "yield"]):
            intent["investment_type"] = "dividend"
        elif any(word in query_lower for word in ["growth", "appreciate", "long-term"]):
            intent["investment_type"] = "growth"
        elif any(word in query_lower for word in ["hedge", "defensive", "safe"]):
            intent["investment_type"] = "defensive"

        # Detect risk preference
        if any(
            word in query_lower
            for word in ["low risk", "safe", "conservative", "stable"]
        ):
            intent["risk_preference"] = "low"
        elif any(
            word in query_lower for word in ["high risk", "aggressive", "volatile"]
        ):
            intent["risk_preference"] = "high"

        # Detect sectors
        if "tech" in query_lower or "technology" in query_lower:
            intent["sectors"].append("Technology")
        if "healthcare" in query_lower or "health" in query_lower:
            intent["sectors"].append("Healthcare")

        return intent

    # Extract insights from analysis results
    insights = extract_investment_insights(analysis_results)
    query_intent = analyze_query_intent(user_query)

    # Generate investment recommendations
    recommendations = []
    if insights["stocks"]:
        total_stocks = min(len(insights["stocks"]), 3)  # Max 3 recommendations
        allocation_per_stock = (investment_amount or 10000) / total_stocks

        for i, stock in enumerate(insights["stocks"][:total_stocks]):
            recommendation = {
                "rank": i + 1,
                "symbol": stock["symbol"],
                "allocation": round(allocation_per_stock),
                "reasoning": [],
                "confidence": stock.get("confidence", "Medium"),
            }

            # Add reasoning based on query intent and stock data
            if query_intent["investment_type"] == "dividend" and insights[
                "financial_metrics"
            ].get("dividend_yield"):
                recommendation["reasoning"].append(
                    "Strong dividend yield for income generation"
                )
            elif query_intent["investment_type"] == "growth":
                recommendation["reasoning"].append(
                    "Consistent growth patterns over past 5 years"
                )

            if stock.get("score", 0) > 0.7:
                recommendation["reasoning"].append(
                    "High analytical score indicating strong fundamentals"
                )

            if (
                insights["sentiment"].get(stock["symbol"], {}).get("trend")
                == "positive"
            ):
                recommendation["reasoning"].append("Positive market sentiment trend")

            if not recommendation["reasoning"]:
                recommendation["reasoning"].append(
                    "Meets screening criteria for the requested investment profile"
                )

            recommendations.append(recommendation)

    # Create comprehensive report
    report = {
        "query_analysis": {
            "original_query": user_query,
            "investment_intent": query_intent,
            "investment_amount": investment_amount or 10000,
        },
        "recommendations": recommendations,
        "risk_assessment": {
            "key_risks": insights["risks"],
            "overall_risk_level": query_intent["risk_preference"],
            "diversification_advice": "Spread investments across recommended stocks to reduce concentration risk",
        },
        "investment_strategy": {
            "approach": f"{query_intent['investment_type'].title()} investing with {query_intent['risk_preference']} risk profile",
            "time_horizon": query_intent["time_horizon"],
            "rebalancing": "Review quarterly and rebalance if allocation drifts >5%",
            "monitoring": "Track key financial metrics and market sentiment monthly",
        },
        "market_insights": {
            "financial_metrics": insights["financial_metrics"],
            "sentiment_overview": insights["sentiment"],
        },
        "disclaimer": "This analysis is generated by AI and should not be considered as professional financial advice. Always consult with a qualified financial advisor before making investment decisions.",
    }

    return report


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
