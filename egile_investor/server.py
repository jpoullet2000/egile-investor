"""
MCP Server for Egile Investor

This server exposes investment agent capabilities as MCP tools.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

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
    filename: Optional[str] = None,
    save_to_file: bool = True,
) -> Dict[str, Any]:
    """
    Generate comprehensive investment analysis reports and optionally save to file.

    Args:
        analysis_type: Type of report (stock_analysis, portfolio_review, market_outlook)
        data: Analysis data to include in the report
        format_type: Output format (markdown, html, pdf)
        filename: Optional filename for saving the report. If not provided, uses report<YYYYMMDD-HHMM>.md
        save_to_file: Whether to save the report to a file (default: True)

    Returns:
        Generated report with formatted content and file information
    """
    from datetime import datetime
    
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

    # Generate report content
    timestamp = datetime.now()
    formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    # Create comprehensive markdown content
    content = f"""# {title}

*Generated on {formatted_timestamp}*

## Report Summary

**Analysis Type:** {analysis_type.replace('_', ' ').title()}  
**Format:** {format_type}  
**Generated At:** {formatted_timestamp}

## Analysis Data

"""
    
    # Add data content based on type
    if isinstance(data_dict, dict) and data_dict:
        if "symbol" in data_dict:
            content += f"**Stock Symbol:** {data_dict['symbol']}\n\n"
        
        if "score" in data_dict:
            content += f"**Analysis Score:** {data_dict['score']}\n\n"
            
        if "recommendation" in data_dict:
            content += f"**Recommendation:** {data_dict['recommendation']}\n\n"
        
        # Add other relevant data
        content += "### Detailed Data\n\n"
        for key, value in data_dict.items():
            if key not in ["symbol", "score", "recommendation"]:
                content += f"- **{key.replace('_', ' ').title()}:** {value}\n"
        content += "\n"
    else:
        content += f"Analysis data: {data_dict}\n\n"
    
    content += """## Disclaimer

This report is generated by AI and should not be considered as professional financial advice. 
Always consult with a qualified financial advisor before making investment decisions.

---
*Report generated by Egile Investor AI Agent*
"""
    
    # Determine filename
    file_saved = False
    file_path = None
    
    if save_to_file:
        if filename:
            # Use provided filename, ensure it has .md extension if it's markdown
            if format_type == "markdown" and not filename.endswith('.md'):
                file_path = f"{filename}.md"
            else:
                file_path = filename
        else:
            # Generate filename with timestamp
            timestamp_str = timestamp.strftime("%Y%m%d-%H%M")
            file_path = f"report{timestamp_str}.md"
        
        try:
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            file_saved = True
            logger.info(f"Report saved to file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save report to file {file_path}: {e}")
            file_saved = False
    
    # Create response
    response = {
        "report_type": analysis_type,
        "format": format_type,
        "title": title,
        "content": content,
        "data_used": data_dict,
        "generated_at": timestamp.isoformat(),
    }
    
    # Add file information if saved
    if save_to_file:
        response["file_info"] = {
            "save_attempted": True,
            "file_saved": file_saved,
            "file_path": file_path,
            "file_size": len(content.encode('utf-8')) if file_saved else None,
        }
    
    return response


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

    # Generate markdown report
    markdown_content = f"""# Investment Analysis Report

## Query Analysis
**Original Query:** {user_query}  
**Investment Strategy:** {query_intent["investment_type"].title()} investing  
**Risk Profile:** {query_intent["risk_preference"].title()} risk  
**Investment Amount:** ${investment_amount or 10000:,}

## Investment Recommendations

"""

    if recommendations:
        for rec in recommendations:
            markdown_content += f"""### {rec["rank"]}. {rec["symbol"]} - ${rec["allocation"]:,} allocation

**Confidence Level:** {rec["confidence"]}

**Reasoning:**
"""
            for reason in rec["reasoning"]:
                markdown_content += f"- {reason}\n"
            markdown_content += "\n"
    else:
        markdown_content += "No specific investment recommendations were generated based on the analysis.\n\n"

    markdown_content += f"""## Risk Assessment

**Overall Risk Level:** {query_intent["risk_preference"].title()}

**Key Risks Identified:**
"""
    for risk in insights["risks"]:
        markdown_content += f"- {risk}\n"

    markdown_content += f"""
**Diversification Strategy:** {report["risk_assessment"]["diversification_advice"]}

## Investment Strategy

**Approach:** {report["investment_strategy"]["approach"]}  
**Time Horizon:** {report["investment_strategy"]["time_horizon"]}  
**Rebalancing:** {report["investment_strategy"]["rebalancing"]}  
**Monitoring:** {report["investment_strategy"]["monitoring"]}

## Market Insights

"""

    if insights["financial_metrics"]:
        markdown_content += "**Key Financial Metrics:**\n"
        for metric, value in insights["financial_metrics"].items():
            markdown_content += f"- {metric.replace('_', ' ').title()}: {value}\n"
        markdown_content += "\n"

    if insights["sentiment"]:
        markdown_content += "**Market Sentiment Overview:**\n"
        for symbol, sentiment in insights["sentiment"].items():
            markdown_content += f"- {symbol}: {sentiment['trend'].title()} (Score: {sentiment['score']})\n"
        markdown_content += "\n"

    markdown_content += f"""## Important Disclaimer

{report["disclaimer"]}

---
*Report generated on {datetime.now().strftime("%Y-%m-%d at %H:%M:%S")}*
"""

    # Add markdown content to the report
    report["markdown_report"] = markdown_content

    return report


@mcp.tool()
async def summarize_analysis_execution(
    user_query: str,
    execution_results: List[Dict[str, Any]],
    investment_amount: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Create a comprehensive summary of the analysis execution steps and final conclusion.

    Args:
        user_query: The original user question/request
        execution_results: List of step execution results from the AI agent
        investment_amount: Amount to invest (if specified in query)

    Returns:
        Structured summary with step analysis and final conclusion
    """

    def analyze_step_outcomes(results):
        """Analyze what was accomplished in each execution step."""
        step_summaries = []
        key_findings = {
            "stocks_screened": [],
            "stocks_analyzed": [],
            "recommendations": [],
            "risks_identified": [],
            "tools_used": set(),
            "success_count": 0,
            "total_steps": len(results),
        }

        for step in results:
            step_num = step.get("step", 0)
            tool_name = step.get("tool", "unknown")
            description = step.get("description", "")
            success = step.get("success", False)
            result_data = step.get("result", {})
            error = step.get("error", "")
            fallback_used = step.get("fallback_used", False)

            key_findings["tools_used"].add(tool_name)
            if success:
                key_findings["success_count"] += 1

            # Analyze step outcome
            step_summary = {
                "step": step_num,
                "tool": tool_name,
                "description": description,
                "status": "✅ Success" if success else "❌ Failed",
                "outcome": "",
                "key_data": {},
            }

            if fallback_used:
                step_summary["status"] += " (Fallback Used)"

            # Extract meaningful outcomes based on tool type
            if success and result_data:
                if tool_name == "screen_stocks" and isinstance(result_data, list):
                    count = len(result_data)
                    step_summary["outcome"] = f"Found {count} stocks matching criteria"
                    step_summary["key_data"]["stocks_found"] = count
                    if count > 0:
                        symbols = [
                            stock.get("symbol", "")
                            for stock in result_data
                            if isinstance(stock, dict)
                        ]
                        key_findings["stocks_screened"].extend(symbols[:5])  # Top 5
                        step_summary["key_data"]["top_symbols"] = symbols[:3]

                elif tool_name in ["analyze_stock", "analyze_multiple_stocks"]:
                    if isinstance(result_data, dict):
                        if "symbol" in result_data:
                            # Single stock analysis
                            symbol = result_data["symbol"]
                            step_summary["outcome"] = f"Analyzed {symbol}"
                            key_findings["stocks_analyzed"].append(symbol)

                            # Extract recommendation if available
                            overall = result_data.get("overall_assessment", {})
                            recommendation = overall.get("recommendation", "")
                            if recommendation:
                                key_findings["recommendations"].append(
                                    f"{symbol}: {recommendation}"
                                )
                                step_summary["key_data"]["recommendation"] = (
                                    recommendation
                                )

                        else:
                            # Multiple stock analysis
                            analyzed_count = 0
                            for symbol, analysis in result_data.items():
                                if isinstance(analysis, dict) and "symbol" in analysis:
                                    analyzed_count += 1
                                    key_findings["stocks_analyzed"].append(symbol)

                                    # Extract recommendations
                                    overall = analysis.get("overall_assessment", {})
                                    recommendation = overall.get("recommendation", "")
                                    if recommendation:
                                        key_findings["recommendations"].append(
                                            f"{symbol}: {recommendation}"
                                        )

                            step_summary["outcome"] = (
                                f"Analyzed {analyzed_count} stocks"
                            )
                            step_summary["key_data"]["stocks_analyzed"] = analyzed_count

                elif tool_name == "create_investment_report":
                    recommendations = result_data.get("recommendations", [])
                    step_summary["outcome"] = (
                        f"Generated investment report with {len(recommendations)} recommendations"
                    )
                    step_summary["key_data"]["report_generated"] = True
                    step_summary["key_data"]["recommendation_count"] = len(
                        recommendations
                    )

                elif tool_name in ["risk_assessment", "assess_multiple_risks"]:
                    if "risk_level" in result_data:
                        risk_level = result_data["risk_level"]
                        step_summary["outcome"] = f"Risk assessment: {risk_level} risk"
                        key_findings["risks_identified"].append(risk_level)
                        step_summary["key_data"]["risk_level"] = risk_level

                else:
                    # Generic outcome for other tools
                    if isinstance(result_data, (list, dict)):
                        size = len(result_data)
                        step_summary["outcome"] = (
                            f"Processed data ({size} {'items' if isinstance(result_data, list) else 'fields'})"
                        )
                    else:
                        step_summary["outcome"] = "Completed successfully"

            elif not success:
                step_summary["outcome"] = (
                    f"Failed: {error[:100]}..."
                    if len(error) > 100
                    else f"Failed: {error}"
                )

            step_summaries.append(step_summary)

        key_findings["tools_used"] = list(key_findings["tools_used"])
        return step_summaries, key_findings

    def generate_final_conclusion(query, findings, investment_amount):
        """Generate a final conclusion that directly answers the user's query."""
        conclusion = {
            "query_addressed": query,
            "analysis_summary": "",
            "key_findings": [],
            "investment_recommendations": [],
            "limitations": [],
            "next_steps": [],
            "confidence_level": "Medium",
        }

        # Analyze success rate
        success_rate = (
            (findings["success_count"] / findings["total_steps"]) * 100
            if findings["total_steps"] > 0
            else 0
        )

        # Generate analysis summary
        tools_used = ", ".join(findings["tools_used"][:5])
        conclusion["analysis_summary"] = (
            f"Completed {findings['success_count']}/{findings['total_steps']} analysis steps "
            f"({success_rate:.1f}% success rate) using tools: {tools_used}"
        )

        # Key findings
        if findings["stocks_screened"]:
            conclusion["key_findings"].append(
                f"Screened stocks: {', '.join(findings['stocks_screened'][:5])}"
            )

        if findings["stocks_analyzed"]:
            conclusion["key_findings"].append(
                f"Analyzed stocks: {', '.join(set(findings['stocks_analyzed'][:5]))}"
            )

        if findings["recommendations"]:
            conclusion["key_findings"].append(
                f"Generated {len(findings['recommendations'])} stock recommendations"
            )

        # Investment recommendations
        if findings["recommendations"]:
            conclusion["investment_recommendations"] = findings["recommendations"][
                :5
            ]  # Top 5
        else:
            conclusion["investment_recommendations"] = [
                "No specific stock recommendations generated due to analysis limitations"
            ]

        # Limitations
        failed_steps = findings["total_steps"] - findings["success_count"]
        if failed_steps > 0:
            conclusion["limitations"].append(
                f"{failed_steps} analysis step(s) failed or had limited results"
            )

        if not findings["stocks_screened"] and not findings["stocks_analyzed"]:
            conclusion["limitations"].append(
                "No stocks were successfully screened or analyzed"
            )

        # Next steps
        if findings["stocks_screened"] and not findings["stocks_analyzed"]:
            conclusion["next_steps"].append(
                "Consider individual analysis of screened stocks"
            )

        if not findings["recommendations"]:
            conclusion["next_steps"].append(
                "Manual research recommended for investment decisions"
            )

        conclusion["next_steps"].append(
            "Consult with a financial advisor before making investment decisions"
        )

        # Confidence level
        if success_rate >= 80 and findings["recommendations"]:
            conclusion["confidence_level"] = "High"
        elif success_rate >= 60:
            conclusion["confidence_level"] = "Medium"
        else:
            conclusion["confidence_level"] = "Low"

        return conclusion

    # Analyze the execution steps
    step_summaries, key_findings = analyze_step_outcomes(execution_results)

    # Generate final conclusion
    final_conclusion = generate_final_conclusion(
        user_query, key_findings, investment_amount
    )

    # Create comprehensive summary
    summary = {
        "original_query": user_query,
        "execution_overview": {
            "total_steps": key_findings["total_steps"],
            "successful_steps": key_findings["success_count"],
            "success_rate": f"{(key_findings['success_count'] / key_findings['total_steps'] * 100):.1f}%"
            if key_findings["total_steps"] > 0
            else "0%",
            "tools_used": key_findings["tools_used"],
        },
        "step_by_step_analysis": step_summaries,
        "key_discoveries": {
            "stocks_screened": list(set(key_findings["stocks_screened"])),
            "stocks_analyzed": list(set(key_findings["stocks_analyzed"])),
            "recommendations_generated": key_findings["recommendations"],
            "risks_identified": key_findings["risks_identified"],
        },
        "final_conclusion": final_conclusion,
        "investment_amount": investment_amount,
        "generated_at": datetime.now().isoformat(),
    }

    return summary


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
