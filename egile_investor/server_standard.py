"""
MCP Server for Egile Investor - Standard MCP Implementation

This server exposes investment agent capabilities as MCP tools using standard MCP.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date
import json

import structlog
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .agent import InvestmentAgent
from .config import InvestmentAgentConfig, AzureOpenAIConfig


logger = structlog.get_logger(__name__)

# Current date context for analysis
CURRENT_DATE = date(2025, 6, 26)  # June 26, 2025
CURRENT_YEAR = 2025
CURRENT_QUARTER = "Q2 2025"


def get_date_context() -> Dict[str, Any]:
    """Get current date context for analysis tools."""
    return {
        "current_date": CURRENT_DATE.isoformat(),
        "current_year": CURRENT_YEAR,
        "current_quarter": CURRENT_QUARTER,
        "analysis_context": f"Analysis performed on {CURRENT_DATE.strftime('%B %d, %Y')}",
        "market_year": CURRENT_YEAR,
        "ytd_period": f"January 1, {CURRENT_YEAR} to {CURRENT_DATE.strftime('%B %d, %Y')}",
    }


# Create MCP server
server = Server("egile-investor")

# Global investment agent instance
investment_agent: Optional[InvestmentAgent] = None


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


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="analyze_stock",
            description="Perform comprehensive stock analysis with technical and fundamental insights. Analysis is performed with context of current date: June 26, 2025.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol (e.g., 'AAPL', 'MSFT')",
                    },
                    "analysis_type": {
                        "type": "string",
                        "default": "comprehensive",
                        "description": "Type of analysis (brief, comprehensive, detailed)",
                    },
                    "include_technical": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to include technical analysis",
                    },
                    "include_fundamental": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to include fundamental analysis",
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="get_market_data",
            description="Retrieve real-time and historical market data for a stock.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol"},
                    "period": {
                        "type": "string",
                        "default": "1y",
                        "description": "Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)",
                    },
                    "source": {
                        "type": "string",
                        "default": "yahoo",
                        "description": "Data source (yahoo, alpha_vantage, finnhub)",
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="get_current_price",
            description="Get the most current stock price with multiple fallback methods. This tool focuses specifically on getting the latest available price data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol (e.g., 'AAPL', 'AMZN')",
                    },
                    "include_details": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to include additional price details (change, volume, etc.)",
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="screen_stocks",
            description="Screen stocks based on various financial criteria with current market context as of June 26, 2025.",
            inputSchema={
                "type": "object",
                "properties": {
                    "criteria": {"type": "object", "description": "Screening criteria"},
                    "universe": {
                        "type": ["array", "string", "null"],
                        "description": "Stock universe to screen",
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of results",
                    },
                    "use_sp500": {
                        "type": "boolean",
                        "default": False,
                        "description": "Screen all S&P 500 stocks",
                    },
                },
                "required": ["criteria"],
            },
        ),
        Tool(
            name="validate_symbol",
            description="Validate a stock symbol and suggest corrections for common mistakes or similar symbols.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol to validate (e.g., 'AMZ', 'APPL')",
                    },
                    "suggest_alternatives": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to suggest alternative symbols",
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="debug_price_data",
            description="Debug tool to show all available price fields for a stock to identify data issues.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol to debug"},
                },
                "required": ["symbol"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    try:
        agent = await get_investment_agent()
        date_context = get_date_context()

        if name == "analyze_stock":
            symbol = arguments.get("symbol")
            analysis_type = arguments.get("analysis_type", "comprehensive")
            include_technical = arguments.get("include_technical", True)
            include_fundamental = arguments.get("include_fundamental", True)

            result = await agent.analyze_stock(
                symbol=symbol,
                analysis_type=analysis_type,
                include_technical=include_technical,
                include_fundamental=include_fundamental,
            )

            # Add date context to results
            if isinstance(result, dict):
                result.update(
                    {
                        "analysis_date": date_context["current_date"],
                        "market_context": f"Analysis for {symbol} as of {date_context['analysis_context']}",
                        "current_year": date_context["current_year"],
                        "ytd_context": date_context["ytd_period"],
                    }
                )

            return [TextContent(type="text", text=json.dumps(result, default=str))]

        elif name == "get_market_data":
            symbol = arguments.get("symbol")
            period = arguments.get("period", "1y")
            source = arguments.get("source", "yahoo")

            result = await agent.get_stock_data(
                symbol=symbol,
                period=period,
                source=source,
            )

            return [TextContent(type="text", text=json.dumps(result, default=str))]

        elif name == "get_current_price":
            symbol = arguments.get("symbol")
            include_details = arguments.get("include_details", True)

            # Get basic stock data with multiple fallbacks
            try:
                stock_data = await agent.get_stock_data(
                    symbol=symbol, period="1d", source="yahoo"
                )
            except Exception as e:
                # If 1d fails, try 5d for more data
                stock_data = await agent.get_stock_data(
                    symbol=symbol, period="5d", source="yahoo"
                )

            info = stock_data.get("info", {})
            hist_data = stock_data.get("historical_data", {})

            # Enhanced price extraction with validation
            current_price = None
            price_source = "unknown"
            price_timestamp = None
            confidence = "low"

            # Method 1: Try current/real-time price fields first
            if info.get("currentPrice") and info["currentPrice"] > 0:
                current_price = info["currentPrice"]
                price_source = "currentPrice_field"
                confidence = "high"
                price_timestamp = info.get("regularMarketTime")
            elif info.get("regularMarketPrice") and info["regularMarketPrice"] > 0:
                current_price = info["regularMarketPrice"]
                price_source = "regularMarketPrice_field"
                confidence = "high"
                price_timestamp = info.get("regularMarketTime")
            
            # Method 2: If no current price, use most recent close but with lower confidence
            elif info.get("previousClose") and info["previousClose"] > 0:
                current_price = info["previousClose"]
                price_source = "previousClose_field"
                confidence = "medium"

            # Method 3: Latest from historical data as last resort
            elif hist_data.get("Close"):
                close_prices = hist_data["Close"]
                if close_prices:
                    # Get the most recent date
                    latest_date = max(close_prices.keys())
                    latest_price = close_prices[latest_date]
                    if latest_price > 0:
                        current_price = latest_price
                        price_source = f"historical_close_{latest_date[:10]}"
                        confidence = "low"
                        price_timestamp = latest_date

            # Data validation - check for obviously stale data
            data_quality_warnings = []
            market_state = info.get("marketState", "unknown")
            
            # If market is open but we only have previousClose, that's suspicious
            if (market_state in ["REGULAR", "PRE", "POST"] and 
                price_source == "previousClose_field"):
                data_quality_warnings.append("Market appears open but only previousClose available")
                confidence = "low"
            
            # Check if the price seems reasonable (basic sanity check)
            if current_price:
                # Very basic range check - prices should be positive and reasonable
                if current_price <= 0:
                    data_quality_warnings.append("Price is zero or negative")
                    confidence = "very_low"
                elif current_price > 10000:  # Prices over $10k are rare
                    data_quality_warnings.append("Price seems unusually high")
                elif current_price < 0.01:  # Prices under 1 cent are rare for major stocks
                    data_quality_warnings.append("Price seems unusually low")

            # Prepare enhanced result
            result = {
                "symbol": symbol,
                "current_price": current_price,
                "price_source": price_source,
                "confidence": confidence,
                "price_timestamp": price_timestamp,
                "market_state": market_state,
                "data_quality_warnings": data_quality_warnings,
                "timestamp": date_context["current_date"],
                "retrieval_context": f"Price retrieved on {date_context['analysis_context']}",
            }

            if include_details and info:
                # Add additional price details
                result.update({
                    "company_name": info.get("longName") or info.get("shortName"),
                    "currency": info.get("currency", "USD"),
                    "previous_close": info.get("previousClose"),
                    "regular_market_change": info.get("regularMarketChange"),
                    "regular_market_change_percent": info.get("regularMarketChangePercent"),
                    "market_cap": info.get("marketCap"),
                    "volume": info.get("volume") or info.get("regularMarketVolume"),
                    "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                    "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                    "bid": info.get("bid"),
                    "ask": info.get("ask"),
                    "day_high": info.get("dayHigh"),
                    "day_low": info.get("dayLow"),
                    # Additional validation info
                    "alternative_prices": {
                        k: v for k, v in {
                            "currentPrice": info.get("currentPrice"),
                            "regularMarketPrice": info.get("regularMarketPrice"), 
                            "previousClose": info.get("previousClose"),
                            "bid": info.get("bid"),
                            "ask": info.get("ask")
                        }.items() if v is not None and v > 0
                    }
                })

            return [TextContent(type="text", text=json.dumps(result, default=str))]

        elif name == "screen_stocks":
            criteria = arguments.get("criteria")
            universe = arguments.get("universe")
            max_results = arguments.get("max_results", 10)
            use_sp500 = arguments.get("use_sp500", False)

            screened = await agent.screen_stocks(
                criteria=criteria,
                universe=universe,
                max_results=max_results,
                use_sp500=use_sp500,
            )

            # Add date context to screening results
            if isinstance(screened, list):
                for stock in screened:
                    if isinstance(stock, dict):
                        stock.update(
                            {
                                "screening_date": date_context["current_date"],
                                "market_year": date_context["current_year"],
                                "screening_context": f"Screened on {date_context['analysis_context']}",
                            }
                        )

            return [TextContent(type="text", text=json.dumps(screened, default=str))]

        elif name == "validate_symbol":
            symbol = arguments.get("symbol", "").upper()
            suggest_alternatives = arguments.get("suggest_alternatives", True)

            # Common symbol corrections
            symbol_corrections = {
                "AMZ": "AMZN",  # Amazon
                "APPL": "AAPL",  # Apple
                "GOOG": "GOOGL",  # Google/Alphabet
                "TESLA": "TSLA",  # Tesla
                "MSFT": "MSFT",  # Microsoft (correct)
                "BRK": "BRK.B",  # Berkshire Hathaway
                "FB": "META",  # Meta (formerly Facebook)
                "TWTR": "X",  # X (formerly Twitter) - though may be delisted
            }

            result = {
                "original_symbol": symbol,
                "is_valid": True,
                "corrected_symbol": None,
                "suggestions": [],
                "message": "",
                "timestamp": date_context["current_date"],
            }

            # Check if this is a known incorrect symbol
            if symbol in symbol_corrections:
                corrected = symbol_corrections[symbol]
                result.update(
                    {
                        "is_valid": False,
                        "corrected_symbol": corrected,
                        "message": f"'{symbol}' is not a valid symbol. Did you mean '{corrected}'?",
                        "suggestions": [corrected],
                    }
                )

            # Try to validate by attempting to get basic info
            elif suggest_alternatives:
                try:
                    # Try to get minimal data to see if symbol exists
                    test_data = await agent.get_stock_data(
                        symbol=symbol, period="1d", source="yahoo"
                    )
                    info = test_data.get("info", {})

                    if not info or not info.get("longName"):
                        # Symbol might be invalid
                        result.update(
                            {
                                "is_valid": False,
                                "message": f"'{symbol}' appears to be invalid or delisted. Check the symbol and try again.",
                            }
                        )

                        # Suggest common alternatives based on partial matching
                        common_symbols = [
                            "AAPL",
                            "AMZN",
                            "GOOGL",
                            "MSFT",
                            "TSLA",
                            "META",
                            "NVDA",
                            "JPM",
                            "JNJ",
                            "V",
                        ]
                        partial_matches = [
                            s
                            for s in common_symbols
                            if symbol[:2] in s or s[:2] in symbol
                        ]
                        if partial_matches:
                            result["suggestions"] = partial_matches[:3]
                            result["message"] += (
                                f" Possible alternatives: {', '.join(partial_matches[:3])}"
                            )
                    else:
                        # Symbol appears valid
                        company_name = info.get("longName", "Unknown Company")
                        result.update(
                            {
                                "is_valid": True,
                                "message": f"'{symbol}' is valid: {company_name}",
                                "company_name": company_name,
                            }
                        )

                except Exception as e:
                    result.update(
                        {
                            "is_valid": False,
                            "message": f"Could not validate '{symbol}': {str(e)}",
                            "error": str(e),
                        }
                    )

            return [TextContent(type="text", text=json.dumps(result, default=str))]

        elif name == "screen_stocks":
            criteria = arguments.get("criteria")
            universe = arguments.get("universe")
            max_results = arguments.get("max_results", 10)
            use_sp500 = arguments.get("use_sp500", False)

            screened = await agent.screen_stocks(
                criteria=criteria,
                universe=universe,
                max_results=max_results,
                use_sp500=use_sp500,
            )

            # Add date context to screening results
            if isinstance(screened, list):
                for stock in screened:
                    if isinstance(stock, dict):
                        stock.update(
                            {
                                "screening_date": date_context["current_date"],
                                "market_year": date_context["current_year"],
                                "screening_context": f"Screened on {date_context['analysis_context']}",
                            }
                        )

            return [TextContent(type="text", text=json.dumps(screened, default=str))]

        elif name == "validate_symbol":
            symbol = arguments.get("symbol", "").upper()
            suggest_alternatives = arguments.get("suggest_alternatives", True)

            # Common symbol corrections
            symbol_corrections = {
                "AMZ": "AMZN",  # Amazon
                "APPL": "AAPL",  # Apple
                "GOOG": "GOOGL",  # Google/Alphabet
                "TESLA": "TSLA",  # Tesla
                "MSFT": "MSFT",  # Microsoft (correct)
                "BRK": "BRK.B",  # Berkshire Hathaway
                "FB": "META",  # Meta (formerly Facebook)
                "TWTR": "X",  # X (formerly Twitter) - though may be delisted
            }

            result = {
                "original_symbol": symbol,
                "is_valid": True,
                "corrected_symbol": None,
                "suggestions": [],
                "message": "",
                "timestamp": date_context["current_date"],
            }

            # Check if this is a known incorrect symbol
            if symbol in symbol_corrections:
                corrected = symbol_corrections[symbol]
                result.update(
                    {
                        "is_valid": False,
                        "corrected_symbol": corrected,
                        "message": f"'{symbol}' is not a valid symbol. Did you mean '{corrected}'?",
                        "suggestions": [corrected],
                    }
                )

            # Try to validate by attempting to get basic info
            elif suggest_alternatives:
                try:
                    # Try to get minimal data to see if symbol exists
                    test_data = await agent.get_stock_data(
                        symbol=symbol, period="1d", source="yahoo"
                    )
                    info = test_data.get("info", {})

                    if not info or not info.get("longName"):
                        # Symbol might be invalid
                        result.update(
                            {
                                "is_valid": False,
                                "message": f"'{symbol}' appears to be invalid or delisted. Check the symbol and try again.",
                            }
                        )

                        # Suggest common alternatives based on partial matching
                        common_symbols = [
                            "AAPL",
                            "AMZN",
                            "GOOGL",
                            "MSFT",
                            "TSLA",
                            "META",
                            "NVDA",
                            "JPM",
                            "JNJ",
                            "V",
                        ]
                        partial_matches = [
                            s
                            for s in common_symbols
                            if symbol[:2] in s or s[:2] in symbol
                        ]
                        if partial_matches:
                            result["suggestions"] = partial_matches[:3]
                            result["message"] += (
                                f" Possible alternatives: {', '.join(partial_matches[:3])}"
                            )
                    else:
                        # Symbol appears valid
                        company_name = info.get("longName", "Unknown Company")
                        result.update(
                            {
                                "is_valid": True,
                                "message": f"'{symbol}' is valid: {company_name}",
                                "company_name": company_name,
                            }
                        )

                except Exception as e:
                    result.update(
                        {
                            "is_valid": False,
                            "message": f"Could not validate '{symbol}': {str(e)}",
                            "error": str(e),
                        }
                    )

            return [TextContent(type="text", text=json.dumps(result, default=str))]

        elif name == "debug_price_data":
            symbol = arguments.get("symbol")
            
            # Get comprehensive stock data
            stock_data = await agent.get_stock_data(
                symbol=symbol, period="5d", source="yahoo"
            )
            
            info = stock_data.get("info", {})
            hist_data = stock_data.get("historical_data", {})
            
            # Extract all price-related fields
            price_fields = {}
            
            # Info price fields
            price_related_keys = [
                "currentPrice", "regularMarketPrice", "previousClose",
                "bid", "ask", "open", "dayLow", "dayHigh",
                "regularMarketOpen", "regularMarketDayLow", "regularMarketDayHigh",
                "regularMarketPreviousClose", "preMarketPrice", "postMarketPrice",
                "fiftyTwoWeekLow", "fiftyTwoWeekHigh", "marketCap"
            ]
            
            for key in price_related_keys:
                if key in info and info[key] is not None:
                    price_fields[f"info.{key}"] = info[key]
            
            # Historical data - last 3 days
            if hist_data and "Close" in hist_data:
                close_prices = hist_data["Close"]
                sorted_dates = sorted(close_prices.keys())[-3:]  # Last 3 days
                for date in sorted_dates:
                    price_fields[f"historical.Close.{date[:10]}"] = close_prices[date]
            
            # Market metadata
            market_info = {
                "marketState": info.get("marketState"),
                "exchangeTimezoneName": info.get("exchangeTimezoneName"),
                "regularMarketTime": info.get("regularMarketTime"),
                "gmtOffSetMilliseconds": info.get("gmtOffSetMilliseconds"),
                "quoteType": info.get("quoteType"),
                "currency": info.get("currency"),
                "longName": info.get("longName"),
                "symbol": info.get("symbol"),
                "dataRetrievalTime": datetime.now().isoformat(),
            }
            
            # Try to determine the "best" current price
            best_price = None
            best_source = None
            
            if info.get("currentPrice"):
                best_price = info["currentPrice"]
                best_source = "currentPrice"
            elif info.get("regularMarketPrice"):
                best_price = info["regularMarketPrice"]
                best_source = "regularMarketPrice"
            elif info.get("previousClose"):
                best_price = info["previousClose"]
                best_source = "previousClose"
            elif hist_data.get("Close"):
                close_prices = hist_data["Close"]
                if close_prices:
                    latest_date = max(close_prices.keys())
                    best_price = close_prices[latest_date]
                    best_source = f"historical_close_{latest_date[:10]}"
            
            result = {
                "symbol": symbol,
                "debug_timestamp": datetime.now().isoformat(),
                "market_info": market_info,
                "all_price_fields": price_fields,
                "recommended_current_price": best_price,
                "recommended_source": best_source,
                "total_price_fields_found": len(price_fields),
                "data_freshness_check": {
                    "has_currentPrice": bool(info.get("currentPrice")),
                    "has_regularMarketPrice": bool(info.get("regularMarketPrice")),
                    "has_recent_historical": bool(hist_data.get("Close")),
                    "market_state": info.get("marketState"),
                }
            }
            
            return [TextContent(type="text", text=json.dumps(result, default=str, indent=2))]
            
        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")

        # Enhanced error handling with symbol suggestions
        error_result = {"error": str(e), "tool": name, "arguments": arguments}

        # If it's a symbol-related error, provide helpful suggestions
        if "symbol" in arguments and (
            "symbol" in str(e).lower()
            or "not found" in str(e).lower()
            or "delisted" in str(e).lower()
        ):
            symbol = arguments.get("symbol", "").upper()

            # Common symbol corrections
            symbol_corrections = {
                "AMZ": "AMZN (Amazon)",
                "APPL": "AAPL (Apple)",
                "GOOG": "GOOGL (Google/Alphabet)",
                "TESLA": "TSLA (Tesla)",
                "FB": "META (Meta/Facebook)",
                "TWTR": "X (formerly Twitter)",
                "BRK": "BRK.B (Berkshire Hathaway)",
            }

            if symbol in symbol_corrections:
                error_result["suggestion"] = (
                    f"Did you mean {symbol_corrections[symbol]}?"
                )
                error_result["corrected_symbol"] = symbol_corrections[symbol].split()[0]
            else:
                error_result["suggestion"] = (
                    f"'{symbol}' may be invalid or delisted. Please verify the symbol."
                )

        return [TextContent(type="text", text=json.dumps(error_result))]


async def main():
    """Run the MCP server."""
    logger.info("Starting Egile Investor MCP Server (Standard MCP Implementation)...")

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        raise


# Main execution
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
