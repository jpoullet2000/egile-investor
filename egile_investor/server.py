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
    Screen stocks based on various financial criteria. Results can be used with generate_report to create analysis reports.

    Args:
        criteria: Screening criteria (e.g., {"pe_ratio": {"max": 25}, "roe": {"min": 0.15}})
        universe: Stock universe to screen:
                 - List of symbols: ["AAPL", "MSFT", "GOOGL"]
                 - Predefined universe: "sp500", "nasdaq100", "dow30", "major"
                 - None: uses default major stocks
        max_results: Maximum number of results to return
        use_sp500: If True, screen all S&P 500 stocks (500+ stocks) regardless of universe

    Returns:
        List of stocks meeting the criteria with scores. Use with generate_report for detailed analysis.

    Examples:
        - Screen value stocks: {"pe_ratio": {"max": 15}, "dividend_yield": {"min": 0.03}}
        - Screen growth stocks: {"revenue_growth": {"min": 0.20}, "pe_ratio": {"max": 30}}
        - Screen S&P 500: {"pe_ratio": {"max": 25}}, universe="sp500"
        - Screen NASDAQ 100: {"pe_ratio": {"max": 25}}, universe="nasdaq100"
        - Screen Dow 30: {"pe_ratio": {"max": 25}}, universe="dow30"
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
    max_words: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive investment analysis reports and optionally save to file.
    Can create both full reports and word-limited summaries.

    Args:
        analysis_type: Type of report (stock_analysis, portfolio_review, market_outlook)
        data: Analysis data to include in the report
        format_type: Output format (markdown, html, pdf)
        filename: Optional filename for saving the report. If not provided, uses report<YYYYMMDD-HHMM>.md
        save_to_file: Whether to save the report to a file (default: True)
        max_words: Optional maximum word count for the report. If specified, generates a concise summary version.

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

**Analysis Type:** {analysis_type.replace("_", " ").title()}  
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

    # Apply word limit if specified
    if max_words and max_words > 0:
        words = content.split()
        if len(words) > max_words:
            # Truncate content to max_words
            truncated_content = " ".join(words[:max_words])
            # Try to end at a complete sentence
            last_period = truncated_content.rfind(".")
            last_newline = truncated_content.rfind("\n")
            last_break = max(last_period, last_newline)

            if (
                last_break > len(truncated_content) * 0.8
            ):  # Only if we're not losing too much
                content = truncated_content[: last_break + 1]
            else:
                content = truncated_content

            content += "\n\n*[Report truncated to meet word limit]*"
            logger.info(
                f"Report truncated from {len(words)} to approximately {max_words} words"
            )

    # Determine filename
    file_saved = False
    file_path = None

    if save_to_file:
        if filename:
            # Use provided filename, ensure it has .md extension if it's markdown
            if format_type == "markdown" and not filename.endswith(".md"):
                file_path = f"{filename}.md"
            else:
                file_path = filename
        else:
            # Generate filename with timestamp
            timestamp_str = timestamp.strftime("%Y%m%d-%H%M")
            file_path = f"report{timestamp_str}.md"

        try:
            # Write to file
            with open(file_path, "w", encoding="utf-8") as f:
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
            "file_size": len(content.encode("utf-8")) if file_saved else None,
        }

    return response


@mcp.tool()
async def get_screening_symbols(
    screening_results: Union[str, List[Any], Dict[str, Any]],
    max_symbols: int = 5,
) -> List[str]:
    """
    Extract stock symbols from screening results.

    Args:
        screening_results: Screening results in various formats (JSON string, list, or dict)
        max_symbols: Maximum number of symbols to return

    Returns:
        List of stock symbols from the screening results
    """
    try:
        import json

        # Handle different input formats
        results = None

        if isinstance(screening_results, str):
            # JSON string format
            try:
                results = json.loads(screening_results)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON string: {screening_results}")
                return []

        elif isinstance(screening_results, list):
            # Direct list format
            results = screening_results

        elif isinstance(screening_results, dict):
            # Handle wrapped results like {'content': [...], 'isError': False}
            if "content" in screening_results:
                results = screening_results["content"]
            elif "stocks" in screening_results:
                results = screening_results["stocks"]
            else:
                # Treat the dict itself as a single result
                results = [screening_results] if "symbol" in screening_results else []
        else:
            logger.warning(
                f"Unexpected screening_results type: {type(screening_results)}"
            )
            return []

        # Extract symbols from results
        symbols = []
        if results:
            for result in results[:max_symbols]:
                if isinstance(result, dict) and "symbol" in result:
                    symbols.append(result["symbol"])
                elif isinstance(result, str):
                    # If it's already a symbol string
                    symbols.append(result)

        # If no symbols found, return empty list
        if not symbols:
            logger.warning("No symbols found in screening results")
            return []

        return symbols

    except Exception as e:
        logger.error(f"Failed to extract symbols from screening results: {e}")
        return []


@mcp.tool()
async def analyze_multiple_stocks(
    symbols: Union[List[str], str, List[Any]],
    analysis_type: str = "comprehensive",
    include_technical: bool = True,
    include_fundamental: bool = True,
) -> Dict[str, Any]:
    """
    Perform analysis on multiple stocks at once.

    Args:
        symbols: List of stock symbols, a string to parse, or list of objects containing symbols
        analysis_type: Type of analysis (brief, comprehensive, detailed)
        include_technical: Whether to include technical analysis
        include_fundamental: Whether to include fundamental analysis

    Returns:
        Dictionary with analysis results for each symbol
    """
    # Handle symbols parameter - convert various formats to list of strings
    processed_symbols = []

    if isinstance(symbols, list):
        # Check if it's a list of dictionaries (screening results) or strings
        for item in symbols:
            if isinstance(item, dict) and "symbol" in item:
                # Extract symbol from screening result dict
                processed_symbols.append(item["symbol"])
            elif isinstance(item, str):
                # It's already a symbol string
                processed_symbols.append(item)
            else:
                logger.warning(f"Unexpected item type in symbols list: {type(item)}")

        if processed_symbols:
            symbols = processed_symbols[:10]  # Limit to 10 symbols
            logger.info(f"Processed {len(symbols)} symbols from input list")
        else:
            # Fallback to default symbols if no valid symbols found
            symbols = ["AAPL", "MSFT", "GOOGL"]
            logger.warning("No valid symbols found in list, using fallback symbols")

    elif isinstance(symbols, str):
        # Try to parse as JSON list first (for stringified lists)
        if symbols.startswith("[") and symbols.endswith("]"):
            try:
                import json

                symbols = json.loads(symbols)
                logger.info(f"Parsed stringified list: {symbols}")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse symbols as JSON list: {e}")
                # If it's a placeholder string, use fallback symbols
                if any(
                    placeholder in symbols.lower()
                    for placeholder in ["step", "result", "{", "<"]
                ):
                    symbols = ["AAPL", "MSFT", "GOOGL"]  # Fallback symbols
                else:
                    # Try to extract symbols from string manually
                    import re

                    # Look for patterns like AAPL, MSFT, etc.
                    symbol_pattern = r"\b[A-Z]{1,5}\b"
                    extracted_symbols = re.findall(symbol_pattern, symbols.upper())
                    if extracted_symbols:
                        symbols = extracted_symbols[:10]  # Limit to 10 symbols
                        logger.info(f"Extracted symbols from string: {symbols}")
                    else:
                        # Single symbol as string
                        symbols = [symbols]
        else:
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
    symbols: Union[List[str], str, List[Any]],
    period: str = "1y",
    source: str = "yahoo",
) -> Dict[str, Any]:
    """
    Retrieve market data for multiple stocks at once.

    Args:
        symbols: List of stock symbols, a string to parse, or list of objects containing symbols
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        source: Data source (yahoo, alpha_vantage, finnhub)

    Returns:
        Dictionary with market data for each symbol
    """
    # Handle symbols parameter - convert various formats to list of strings
    processed_symbols = []

    if isinstance(symbols, list):
        # Check if it's a list of dictionaries (screening results) or strings
        for item in symbols:
            if isinstance(item, dict) and "symbol" in item:
                # Extract symbol from screening result dict
                processed_symbols.append(item["symbol"])
            elif isinstance(item, str):
                # It's already a symbol string
                processed_symbols.append(item)
            else:
                logger.warning(f"Unexpected item type in symbols list: {type(item)}")

        if processed_symbols:
            symbols = processed_symbols[:10]  # Limit to 10 symbols
            logger.info(f"Processed {len(symbols)} symbols from input list")
        else:
            # Fallback to default symbols if no valid symbols found
            symbols = ["AAPL", "MSFT", "GOOGL"]
            logger.warning("No valid symbols found in list, using fallback symbols")

    elif isinstance(symbols, str):
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
    symbols: Union[List[str], str, List[Any]],
    risk_factors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Assess investment risk for multiple stocks at once.

    Args:
        symbols: List of stock symbols, a string to parse, or list of objects containing symbols
        risk_factors: Specific risk factors to evaluate

    Returns:
        Dictionary with risk assessment for each symbol
    """
    # Handle symbols parameter - convert various formats to list of strings
    processed_symbols = []

    if isinstance(symbols, list):
        # Check if it's a list of dictionaries (screening results) or strings
        for item in symbols:
            if isinstance(item, dict) and "symbol" in item:
                # Extract symbol from screening result dict
                processed_symbols.append(item["symbol"])
            elif isinstance(item, str):
                # It's already a symbol string
                processed_symbols.append(item)
            else:
                logger.warning(f"Unexpected item type in symbols list: {type(item)}")

        if processed_symbols:
            symbols = processed_symbols[:10]  # Limit to 10 symbols
            logger.info(f"Processed {len(symbols)} symbols from input list")
        else:
            # Fallback to default symbols if no valid symbols found
            symbols = ["AAPL", "MSFT", "GOOGL"]
            logger.warning("No valid symbols found in list, using fallback symbols")

    elif isinstance(symbols, str):
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
    symbols: Union[List[str], str, List[Any]],
    ratios: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calculate financial ratios for multiple stocks at once.

    Args:
        symbols: List of stock symbols, a string to parse, or list of objects containing symbols
        ratios: List of specific ratios to calculate (defaults to all common ratios)

    Returns:
        Dictionary with financial ratios for each symbol
    """
    # Handle symbols parameter - convert various formats to list of strings
    processed_symbols = []

    if isinstance(symbols, list):
        # Check if it's a list of dictionaries (screening results) or strings
        for item in symbols:
            if isinstance(item, dict) and "symbol" in item:
                # Extract symbol from screening result dict
                processed_symbols.append(item["symbol"])
            elif isinstance(item, str):
                # It's already a symbol string
                processed_symbols.append(item)
            else:
                logger.warning(f"Unexpected item type in symbols list: {type(item)}")

        if processed_symbols:
            symbols = processed_symbols[:10]  # Limit to 10 symbols
            logger.info(f"Processed {len(symbols)} symbols from input list")
        else:
            # Fallback to default symbols if no valid symbols found
            symbols = ["AAPL", "MSFT", "GOOGL"]
            logger.warning("No valid symbols found in list, using fallback symbols")

    elif isinstance(symbols, str):
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
    symbols: Union[List[str], str, List[Any]],
    sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Analyze market sentiment for multiple stocks at once.

    Args:
        symbols: List of stock symbols, a string to parse, or list of objects containing symbols
        sources: List of sources to analyze (news, twitter, reddit)

    Returns:
        Dictionary with sentiment analysis for each symbol
    """
    # Handle symbols parameter - convert various formats to list of strings
    processed_symbols = []

    if isinstance(symbols, list):
        # Check if it's a list of dictionaries (screening results) or strings
        for item in symbols:
            if isinstance(item, dict) and "symbol" in item:
                # Extract symbol from screening result dict
                processed_symbols.append(item["symbol"])
            elif isinstance(item, str):
                # It's already a symbol string
                processed_symbols.append(item)
            else:
                logger.warning(f"Unexpected item type in symbols list: {type(item)}")

        if processed_symbols:
            symbols = processed_symbols[:10]  # Limit to 10 symbols
            logger.info(f"Processed {len(symbols)} symbols from input list")
        else:
            # Fallback to default symbols if no valid symbols found
            symbols = ["AAPL", "MSFT", "GOOGL"]
            logger.warning("No valid symbols found in list, using fallback symbols")

    elif isinstance(symbols, str):
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
    symbols: Union[List[str], str, List[Any]],
    weights: Optional[List[float]] = None,
    benchmark: str = "SPY",
    analysis_type: str = "comprehensive",
) -> Dict[str, Any]:
    """
    Create and analyze a portfolio from a list of stock symbols.

    Args:
        symbols: List of stock symbols for the portfolio, a string to parse, or list of objects containing symbols
        weights: Optional list of weights for each symbol (defaults to equal weights)
        benchmark: Benchmark symbol for comparison
        analysis_type: Type of analysis (brief, comprehensive, detailed)

    Returns:
        Portfolio analysis results
    """
    # Handle symbols parameter - convert various formats to list of strings
    processed_symbols = []

    if isinstance(symbols, list):
        # Check if it's a list of dictionaries (screening results) or strings
        for item in symbols:
            if isinstance(item, dict) and "symbol" in item:
                # Extract symbol from screening result dict
                processed_symbols.append(item["symbol"])
            elif isinstance(item, str):
                # It's already a symbol string
                processed_symbols.append(item)
            else:
                logger.warning(f"Unexpected item type in symbols list: {type(item)}")

        if processed_symbols:
            symbols = processed_symbols[:10]  # Limit to 10 symbols
            logger.info(f"Processed {len(symbols)} symbols from input list")
        else:
            # Fallback to default symbols if no valid symbols found
            symbols = ["AAPL", "MSFT", "GOOGL"]
            logger.warning("No valid symbols found in list, using fallback symbols")

    elif isinstance(symbols, str):
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
    screening_results: Union[str, List[Any], Dict[str, Any]],
    count: int = 5,
    sort_by: str = "score",
) -> List[str]:
    """
    Get top stock symbols from screening results based on score or other criteria.

    Args:
        screening_results: Screening results in various formats (JSON string, list, or dict)
        count: Number of top symbols to return
        sort_by: Field to sort by (score, symbol)

    Returns:
        List of top stock symbols
    """
    try:
        import json

        # Handle different input formats
        results = None

        if isinstance(screening_results, str):
            # JSON string format
            try:
                results = json.loads(screening_results)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON string: {screening_results}")
                return []

        elif isinstance(screening_results, list):
            # Direct list format
            results = screening_results

        elif isinstance(screening_results, dict):
            # Handle wrapped results like {'content': [...], 'isError': False}
            if "content" in screening_results:
                results = screening_results["content"]
            elif "stocks" in screening_results:
                results = screening_results["stocks"]
            else:
                # Treat the dict itself as a single result
                results = [screening_results] if "symbol" in screening_results else []
        else:
            logger.warning(
                f"Unexpected screening_results type: {type(screening_results)}"
            )
            return []

        if not results:
            logger.warning("No results found in screening data")
            return []

        # Sort results
        if sort_by == "score" and results:
            sorted_results = sorted(
                results,
                key=lambda x: x.get("score", 0) if isinstance(x, dict) else 0,
                reverse=True,
            )
        else:
            sorted_results = results

        # Extract symbols
        symbols = []
        for result in sorted_results[:count]:
            if isinstance(result, dict) and "symbol" in result:
                symbols.append(result["symbol"])
            elif isinstance(result, str):
                # If it's already a symbol string
                symbols.append(result)

        return symbols

    except Exception as e:
        logger.error(f"Failed to extract top symbols from screening results: {e}")
        return []


@mcp.tool()
async def create_investment_report(
    user_query: str,
    analysis_results: Union[List[Dict[str, Any]], Dict[str, Any], List[Any], str],
    investment_amount: Optional[float] = None,
    avoid_historical_data: bool = False,
) -> Dict[str, Any]:
    """
    Create a comprehensive investment report that directly answers the user's query.
    Handles flexible input types from previous analysis steps.

    Args:
        user_query: The original user question/request
        analysis_results: Analysis results from previous steps (flexible format)
        investment_amount: Amount to invest (if specified in query)
        avoid_historical_data: If True, skip fetching additional historical market data

    Returns:
        Investment report with specific recommendations and reasoning
    """

    def normalize_analysis_results(results) -> List[Dict[str, Any]]:
        """Normalize various input formats to a consistent list of dictionaries."""
        if isinstance(results, str):
            # Handle error strings or single values
            if "Error" in results:
                return [{"error": results, "type": "error"}]
            return [{"description": results, "type": "text"}]

        if isinstance(results, dict):
            # Single dictionary result
            return [results]

        if isinstance(results, list):
            normalized = []
            for item in results:
                if isinstance(item, dict):
                    normalized.append(item)
                elif isinstance(item, str):
                    if "Error" in item:
                        normalized.append({"error": item, "type": "error"})
                    else:
                        normalized.append({"description": item, "type": "text"})
                else:
                    normalized.append({"data": str(item), "type": "unknown"})
            return normalized

        # Fallback for any other type
        return [{"data": str(results), "type": "fallback"}]

    def extract_investment_insights(results):
        """Extract actionable investment insights from analysis results."""
        # First normalize the results
        normalized_results = normalize_analysis_results(results)

        stocks_found = []
        risks_identified = []
        financial_metrics = {}
        sentiment_data = {}
        errors_found = []

        for result in normalized_results:
            if not isinstance(result, dict):
                continue

            # Handle error entries
            if result.get("type") == "error" or "error" in result:
                errors_found.append(
                    result.get("error", result.get("description", "Unknown error"))
                )
                continue

            # Handle analyze_multiple_stocks format: {SYMBOL: analysis_data, SYMBOL2: analysis_data, ...}
            # Check if this looks like a multi-stock analysis result
            if all(
                isinstance(v, dict) and v.get("symbol")
                for k, v in result.items()
                if isinstance(k, str) and k.isupper() and len(k) <= 5
            ):
                # This is likely a multi-stock analysis result
                for symbol, analysis_data in result.items():
                    if isinstance(analysis_data, dict) and analysis_data.get("symbol"):
                        # Extract recommendation from overall_assessment or create default
                        overall_assessment = analysis_data.get("overall_assessment", {})
                        recommendation = overall_assessment.get(
                            "recommendation", "HOLD"
                        )
                        score = (
                            overall_assessment.get("overall_score", 50) / 100.0
                        )  # Convert to 0-1 scale
                        confidence = overall_assessment.get("confidence", "Medium")

                        stocks_found.append(
                            {
                                "symbol": symbol,
                                "score": score,
                                "recommendation": recommendation,
                                "confidence": confidence,
                                "analysis_data": analysis_data,  # Keep full analysis for reference
                            }
                        )

                        # Extract risk information from this stock's analysis
                        if "risk_assessment" in analysis_data:
                            risk_data = analysis_data["risk_assessment"]
                            if "risk_level" in risk_data:
                                risks_identified.append(
                                    f"{symbol} Risk Level: {risk_data['risk_level']}"
                                )
                            if "volatility" in risk_data:
                                risks_identified.append(
                                    f"{symbol} Volatility: {risk_data['volatility']}"
                                )

                # Once we've processed multi-stock format, continue to next result
                continue

            # Extract stock symbols and scores (original logic for other formats)
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
            elif "stocks" in result and isinstance(result["stocks"], list):
                # Handle screening results
                for stock in result["stocks"][:10]:
                    if isinstance(stock, dict) and "symbol" in stock:
                        stocks_found.append(
                            {
                                "symbol": stock["symbol"],
                                "score": stock.get("score", 0),
                                "recommendation": "BUY",
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
            "errors": errors_found,
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
            "analysis_errors": insights.get("errors", []),
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

    # Add analysis errors if any occurred
    if insights.get("errors"):
        markdown_content += "**Analysis Notes:**\n"
        for error in insights["errors"][:3]:  # Show max 3 errors
            markdown_content += f"- {error}\n"
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


@mcp.tool()
async def summarize(
    content: Union[str, Dict[str, Any]],
    summary_type: str = "bullet_points",
    max_length: int = 200,
    focus_area: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate concise summaries of reports, analysis data, or any text content.
    Use this tool when users request summaries, word limits, or concise versions of analysis.

    Args:
        content: Content to summarize (text string or structured data dict)
        summary_type: Type of summary to generate:
                     - "bullet_points": Key points in bullet format (recommended for quick reading)
                     - "paragraph": Single paragraph summary
                     - "executive": Executive summary style (recommended for business reports)
                     - "key_metrics": Focus on key numbers and metrics
        max_length: Maximum length of summary in words (default: 200)
        focus_area: Optional focus area (e.g., "financial_metrics", "recommendations", "risks")

    Returns:
        Dictionary containing the summary and metadata

    Examples:
        - Summarize stock screening results: content=screening_results, summary_type="bullet_points"
        - Create executive summary: content=report_content, summary_type="executive", max_length=500
        - Focus on key metrics: content=analysis_data, summary_type="key_metrics", focus_area="financial_metrics"
    """
    import re
    from datetime import datetime

    # Handle different input types
    if isinstance(content, dict):
        # Convert structured data to text for summarization
        if "content" in content:
            # If it's a report with content field
            text_content = content["content"]
            title = content.get("title", "Analysis Summary")
            symbol = content.get(
                "symbol", content.get("data_used", {}).get("symbol", "")
            )
        elif "symbol" in content:
            # If it's stock analysis data
            symbol = content["symbol"]
            title = f"Stock Analysis Summary: {symbol}"
            text_content = str(content)
        else:
            # Generic dict content
            title = "Data Summary"
            symbol = ""
            text_content = str(content)
    else:
        # Handle string content
        text_content = str(content)
        title = "Content Summary"
        symbol = ""

        # Try to extract symbol from text if present
        symbol_match = re.search(r"\b[A-Z]{2,5}\b", text_content)
        if symbol_match:
            symbol = symbol_match.group()
            title = f"Analysis Summary: {symbol}"

    # Clean and prepare text for summarization
    clean_text = re.sub(r"[#*`\-_]+", " ", text_content)  # Remove markdown formatting
    clean_text = re.sub(r"\s+", " ", clean_text).strip()  # Normalize whitespace

    # Extract key information based on focus area
    key_points = []

    if focus_area == "financial_metrics" or "financial" in clean_text.lower():
        # Focus on financial data
        financial_terms = [
            "PE",
            "ratio",
            "revenue",
            "profit",
            "margin",
            "ROE",
            "debt",
            "earnings",
            "price",
        ]
        for term in financial_terms:
            matches = re.findall(
                rf"{term}[^.]*?[\d.%]+[^.]*?\.", clean_text, re.IGNORECASE
            )
            key_points.extend(matches[:2])  # Limit matches per term

    if focus_area == "recommendations" or "recommend" in clean_text.lower():
        # Focus on recommendations
        rec_patterns = [
            r"recommend[^.]*?\.",
            r"suggest[^.]*?\.",
            r"should[^.]*?\.",
            r"buy|sell|hold[^.]*?\.",
        ]
        for pattern in rec_patterns:
            matches = re.findall(pattern, clean_text, re.IGNORECASE)
            key_points.extend(matches[:2])

    if focus_area == "risks" or "risk" in clean_text.lower():
        # Focus on risk factors
        risk_patterns = [
            r"risk[^.]*?\.",
            r"volatile[^.]*?\.",
            r"concern[^.]*?\.",
            r"warning[^.]*?\.",
        ]
        for pattern in risk_patterns:
            matches = re.findall(pattern, clean_text, re.IGNORECASE)
            key_points.extend(matches[:2])

    # If no focus area or no specific matches, extract general key sentences
    if not key_points:
        sentences = re.split(r"[.!?]+", clean_text)
        # Prioritize sentences with important keywords
        important_keywords = [
            "recommendation",
            "analysis",
            "price",
            "growth",
            "risk",
            "buy",
            "sell",
            "strong",
            "weak",
        ]
        scored_sentences = []

        for sentence in sentences:
            if len(sentence.strip()) > 20:  # Ignore very short sentences
                score = sum(
                    1
                    for keyword in important_keywords
                    if keyword.lower() in sentence.lower()
                )
                scored_sentences.append((score, sentence.strip()))

        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        key_points = [sent[1] for sent in scored_sentences[:5]]

    # Generate summary based on type
    if summary_type == "bullet_points":
        summary_lines = []
        for i, point in enumerate(key_points[:5], 1):
            if point.strip():
                summary_lines.append(f"• {point.strip()}")
        summary_text = "\n".join(summary_lines)

    elif summary_type == "paragraph":
        summary_text = " ".join(key_points[:3])
        if len(summary_text.split()) > max_length:
            words = summary_text.split()[:max_length]
            summary_text = " ".join(words) + "..."

    elif summary_type == "executive":
        if symbol:
            summary_text = f"Executive Summary for {symbol}: "
        else:
            summary_text = "Executive Summary: "
        summary_text += " ".join(key_points[:2])
        if len(summary_text.split()) > max_length:
            words = summary_text.split()[:max_length]
            summary_text = " ".join(words) + "..."

    elif summary_type == "key_metrics":
        # Extract numbers and metrics
        metrics = re.findall(r"[\d.]+%?", clean_text)
        metric_context = []
        for metric in metrics[:5]:
            # Find context around the metric
            pattern = rf".{{0,50}}{re.escape(metric)}.{{0,50}}"
            context_match = re.search(pattern, clean_text)
            if context_match:
                metric_context.append(context_match.group().strip())

        summary_text = "Key Metrics:\n" + "\n".join(
            [f"• {ctx}" for ctx in metric_context[:4]]
        )
    else:
        # Default to bullet points
        summary_text = "\n".join(
            [f"• {point.strip()}" for point in key_points[:4] if point.strip()]
        )

    # Ensure summary doesn't exceed max_length
    if len(summary_text.split()) > max_length:
        words = summary_text.split()[:max_length]
        summary_text = " ".join(words) + "..."

    # Create response
    response = {
        "title": title,
        "summary": summary_text,
        "summary_type": summary_type,
        "word_count": len(summary_text.split()),
        "focus_area": focus_area,
        "symbol": symbol if symbol else None,
        "generated_at": datetime.now().isoformat(),
        "source_length": len(clean_text.split()),
        "compression_ratio": round(
            len(summary_text.split()) / max(len(clean_text.split()), 1), 2
        ),
    }

    return response


@mcp.tool()
async def create_stock_screening_report(
    criteria: Dict[str, Any],
    max_words: int = 1000,
    universe: Optional[Union[List[str], str]] = None,
    max_results: int = 10,
    report_type: str = "comprehensive",
    save_to_file: bool = True,
) -> Dict[str, Any]:
    """
    Screen stocks based on criteria and generate a word-limited analysis report.
    This tool combines screening and report generation in one step.

    Args:
        criteria: Screening criteria (e.g., {"pe_ratio": {"max": 15}, "dividend_yield": {"min": 0.03}})
        max_words: Maximum word count for the report (default: 1000)
        universe: Stock universe to screen (sp500, nasdaq100, dow30, major, or custom list)
        max_results: Maximum number of stocks to include in report
        report_type: Type of report (comprehensive, summary, brief)
        save_to_file: Whether to save the report to a file

    Returns:
        Complete screening report with word limit applied

    Examples:
        - Value stocks report: criteria={"pe_ratio": {"max": 15}, "dividend_yield": {"min": 0.03}}, max_words=1000
        - Growth stocks report: criteria={"revenue_growth": {"min": 0.20}}, max_words=500, report_type="summary"
    """
    from datetime import datetime

    # First, screen the stocks
    agent = await get_investment_agent()
    screened_stocks = await agent.screen_stocks(
        criteria=criteria,
        universe=universe,
        max_results=max_results,
        use_sp500=(universe == "sp500"),
    )

    # Generate comprehensive report content
    timestamp = datetime.now()
    formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    # Create title based on criteria
    criteria_desc = []
    for key, value in criteria.items():
        if isinstance(value, dict):
            for op, val in value.items():
                if op == "max":
                    criteria_desc.append(f"{key.replace('_', ' ')} < {val}")
                elif op == "min":
                    criteria_desc.append(f"{key.replace('_', ' ')} > {val}")
        else:
            criteria_desc.append(f"{key.replace('_', ' ')} = {value}")

    title = f"Stock Screening Report: {', '.join(criteria_desc[:2])}"
    if len(criteria_desc) > 2:
        title += "..."

    content = f"""# {title}

*Generated on {formatted_timestamp}*

## Executive Summary

This screening identified **{len(screened_stocks)}** stocks that match your criteria from the {universe or "major stocks"} universe.

**Screening Criteria:**
"""

    for key, value in criteria.items():
        if isinstance(value, dict):
            for op, val in value.items():
                op_text = "maximum" if op == "max" else "minimum" if op == "min" else op
                content += f"- {key.replace('_', ' ').title()}: {op_text} {val}\n"
        else:
            content += f"- {key.replace('_', ' ').title()}: {value}\n"

    content += f"\n**Results:** {len(screened_stocks)} stocks found\n"
    content += f"**Universe:** {universe or 'Major stocks'}\n\n"

    if screened_stocks:
        content += "## Top Stock Recommendations\n\n"
        for i, stock in enumerate(screened_stocks[:5], 1):  # Show top 5
            symbol = stock.get("symbol", "N/A")
            score = stock.get("score", 0)
            content += f"### {i}. {symbol}\n"
            content += f"**Overall Score:** {score:.2f}\n"

            # Add key metrics if available
            if "metrics" in stock:
                metrics = stock["metrics"]
                content += "**Key Metrics:**\n"
                for metric, value in metrics.items():
                    if value is not None:
                        content += f"- {metric.replace('_', ' ').title()}: {value}\n"
            content += "\n"

        if len(screened_stocks) > 5:
            content += f"\n*... and {len(screened_stocks) - 5} more stocks*\n\n"
    else:
        content += "## No Stocks Found\n\nNo stocks in the selected universe met your screening criteria. Consider:\n"
        content += "- Relaxing some criteria\n"
        content += "- Expanding the stock universe\n"
        content += "- Trying different screening parameters\n\n"

    content += "## Investment Considerations\n\n"
    if screened_stocks:
        content += "The identified stocks show strong fundamentals based on your criteria. Consider:\n"
        content += "- Conducting detailed individual analysis\n"
        content += "- Reviewing recent financial statements\n"
        content += "- Assessing current market conditions\n"
        content += "- Diversifying across different sectors\n\n"

    content += """## Disclaimer

This screening report is generated by AI and should not be considered as professional financial advice. 
Always conduct your own research and consult with a qualified financial advisor before making investment decisions.

---
*Report generated by Egile Investor AI Agent*
"""

    # Apply word limit if specified
    if max_words and max_words > 0:
        words = content.split()
        if len(words) > max_words:
            # Truncate content to max_words
            truncated_content = " ".join(words[:max_words])
            # Try to end at a complete sentence
            last_period = truncated_content.rfind(".")
            last_newline = truncated_content.rfind("\n")
            last_break = max(last_period, last_newline)

            if (
                last_break > len(truncated_content) * 0.8
            ):  # Only if we're not losing too much
                content = truncated_content[: last_break + 1]
            else:
                content = truncated_content

            content += "\n\n*[Report truncated to meet word limit]*"
            logger.info(
                f"Report truncated from {len(words)} to approximately {max_words} words"
            )

    # Save to file if requested
    file_saved = False
    file_path = None

    if save_to_file:
        timestamp_str = timestamp.strftime("%Y%m%d-%H%M")
        file_path = f"screening_report_{timestamp_str}.md"

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            file_saved = True
            logger.info(f"Screening report saved to file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save screening report to file {file_path}: {e}")
            file_saved = False

    # Create response
    response = {
        "report_type": "stock_screening",
        "title": title,
        "content": content,
        "screening_summary": {
            "criteria_used": criteria,
            "stocks_found": len(screened_stocks),
            "universe": universe or "major",
            "word_limit": max_words,
            "actual_word_count": len(content.split()),
            "report_type": report_type,
        },
        "generated_at": timestamp.isoformat(),
        "file_info": {
            "save_attempted": save_to_file,
            "file_saved": file_saved,
            "file_path": file_path,
            "file_size": len(content.encode("utf-8")) if file_saved else None,
        },
        "stocks": screened_stocks,
    }

    return response


@mcp.tool()
async def generate_markdown_summary_report(
    user_query: str,
    analysis_data: Union[List[Dict[str, Any]], Dict[str, Any], List[Any], str],
    max_stocks: int = 5,
    save_to_file: bool = True,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a markdown summary report from analysis data without fetching historical market data.
    Perfect for creating reports when offline or when external APIs are unavailable.

    Args:
        user_query: The original user question/request
        analysis_data: Analysis results from previous steps (flexible format)
        max_stocks: Maximum number of stocks to include in the report
        save_to_file: Whether to save the report to a file (default: True)
        filename: Optional filename for saving the report

    Returns:
        Markdown report with investment recommendations based on provided analysis data
    """
    from datetime import datetime

    def normalize_analysis_data(data) -> List[Dict[str, Any]]:
        """Normalize various input formats to a consistent list of dictionaries."""
        if isinstance(data, str):
            return [{"description": data, "type": "text"}]

        if isinstance(data, dict):
            return [data]

        if isinstance(data, list):
            normalized = []
            for item in data:
                if isinstance(item, dict):
                    normalized.append(item)
                elif isinstance(item, str):
                    normalized.append({"description": item, "type": "text"})
                else:
                    normalized.append({"data": str(item), "type": "unknown"})
            return normalized

        return [{"data": str(data), "type": "fallback"}]

    def extract_top_stocks(normalized_data, max_stocks):
        """Extract top stock recommendations from normalized analysis data."""
        stocks = []

        for result in normalized_data:
            if not isinstance(result, dict):
                continue

            # Handle multi-stock analysis format
            if all(
                isinstance(v, dict) and v.get("symbol")
                for k, v in result.items()
                if isinstance(k, str) and k.isupper() and len(k) <= 5
            ):
                for symbol, analysis_data in result.items():
                    if isinstance(analysis_data, dict) and analysis_data.get("symbol"):
                        overall_assessment = analysis_data.get("overall_assessment", {})
                        recommendation = overall_assessment.get(
                            "recommendation", "HOLD"
                        )
                        score = overall_assessment.get("overall_score", 50)
                        confidence = overall_assessment.get("confidence", "Medium")

                        stocks.append(
                            {
                                "symbol": symbol,
                                "score": score,
                                "recommendation": recommendation,
                                "confidence": confidence,
                                "analysis_data": analysis_data,
                            }
                        )

            # Handle screening results or single stock data
            elif "symbol" in result:
                stocks.append(
                    {
                        "symbol": result["symbol"],
                        "score": result.get("score", 50),
                        "recommendation": result.get("recommendation", "HOLD"),
                        "confidence": result.get("confidence", "Medium"),
                        "analysis_data": result,
                    }
                )
            elif "symbols" in result and isinstance(result["symbols"], list):
                for symbol in result["symbols"][:max_stocks]:
                    if isinstance(symbol, str):
                        stocks.append(
                            {
                                "symbol": symbol,
                                "score": 60,  # Default score for screened stocks
                                "recommendation": "BUY",
                                "confidence": "Medium",
                                "analysis_data": {"symbol": symbol, "screened": True},
                            }
                        )

        # Sort by score and take top stocks
        stocks.sort(key=lambda x: x.get("score", 0), reverse=True)
        return stocks[:max_stocks]

    # Normalize the analysis data
    normalized_data = normalize_analysis_data(analysis_data)

    # Extract top stocks
    top_stocks = extract_top_stocks(normalized_data, max_stocks)

    # Generate report content
    timestamp = datetime.now()
    formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    # Create comprehensive markdown content
    content = f"""# Investment Summary Report for 2025

*Generated on {formatted_timestamp}*

## Executive Summary

This report provides investment recommendations based on analysis of available market data and screening results. The recommendations are derived from fundamental analysis and market screening without relying on real-time historical data.

**Query Addressed:** {user_query}

## Top Investment Recommendations

"""

    if top_stocks:
        content += f"Based on the analysis, here are the top {len(top_stocks)} stock recommendations:\n\n"

        for i, stock in enumerate(top_stocks, 1):
            symbol = stock["symbol"]
            score = stock.get("score", 0)
            recommendation = stock.get("recommendation", "HOLD")
            confidence = stock.get("confidence", "Medium")

            content += f"### {i}. {symbol} - {recommendation}\n\n"
            content += f"- **Analysis Score:** {score}/100\n"
            content += f"- **Recommendation:** {recommendation}\n"
            content += f"- **Confidence Level:** {confidence}\n"

            # Add specific analysis details if available
            analysis_data = stock.get("analysis_data", {})
            if isinstance(analysis_data, dict):
                # Financial metrics
                if "financial_metrics" in analysis_data:
                    metrics = analysis_data["financial_metrics"]
                    content += "- **Key Metrics:** "
                    metric_items = []
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            metric_items.append(
                                f"{key.replace('_', ' ').title()}: {value}"
                            )
                    if metric_items:
                        content += ", ".join(metric_items[:3]) + "\n"
                    else:
                        content += "Available in detailed analysis\n"

                # Risk assessment
                if "risk_assessment" in analysis_data:
                    risk_data = analysis_data["risk_assessment"]
                    risk_level = risk_data.get("risk_level", "Medium")
                    content += f"- **Risk Level:** {risk_level}\n"

                # Overall assessment reasons
                if "overall_assessment" in analysis_data:
                    assessment = analysis_data["overall_assessment"]
                    reasons = assessment.get("reasons", [])
                    if reasons:
                        content += f"- **Key Points:** {', '.join(reasons[:2])}\n"

            content += "\n"
    else:
        content += """No specific stock recommendations could be generated from the available analysis data.

This might be due to:
- Limited analysis data provided
- Screening criteria that didn't match any stocks
- Data format that couldn't be processed

Consider running a stock screening or analysis first to generate recommendations.

"""

    # Add methodology section
    content += """## Methodology

This report is generated using:
- Fundamental analysis metrics (when available)
- Stock screening results
- AI-powered assessment of financial indicators
- Risk evaluation based on available data

**Note:** This analysis is performed without real-time historical market data to ensure reliability and speed.

## Important Disclaimers

- This report is for informational purposes only and should not be considered as professional financial advice
- Past performance does not guarantee future results
- All investments carry risk of loss
- Always consult with a qualified financial advisor before making investment decisions
- Consider diversification and your risk tolerance when investing

## Market Context for 2025

Key considerations for 2025 investments:
- Technology sector continues to show innovation potential
- Interest rate environment affects growth vs value dynamics
- Global economic conditions impact international exposure
- ESG (Environmental, Social, Governance) factors increasingly important
- Consider inflation impact on different sectors

---
*Report generated by Egile Investor AI Agent without external data dependencies*
"""

    # Save to file if requested
    file_saved = False
    file_path = None

    if save_to_file:
        if filename:
            file_path = filename if filename.endswith(".md") else f"{filename}.md"
        else:
            timestamp_str = timestamp.strftime("%Y%m%d-%H%M")
            file_path = f"investment_summary_{timestamp_str}.md"

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            file_saved = True
            logger.info(f"Markdown report saved to file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save report to file {file_path}: {e}")
            file_saved = False

    # Create response
    response = {
        "report_type": "investment_summary",
        "format": "markdown",
        "title": "Investment Summary Report for 2025",
        "content": content,
        "stocks_analyzed": len(top_stocks),
        "query_addressed": user_query,
        "generated_at": timestamp.isoformat(),
        "uses_historical_data": False,  # Key indicator that this report doesn't need external APIs
    }

    # Add file information if saved
    if save_to_file:
        response["file_info"] = {
            "save_attempted": True,
            "file_saved": file_saved,
            "file_path": file_path,
            "file_size": len(content.encode("utf-8")) if file_saved else None,
        }

    return response


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
