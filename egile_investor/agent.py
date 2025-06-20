"""
Investment Agent for intelligent investment analysis and portfolio management.

This agent provides intelligent routing and execution of investment tasks
across different data sources and tools, optimized for financial workflows.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

import structlog
import yfinance as yf
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import finnhub

from .client import AzureOpenAIClient
from .config import InvestmentAgentConfig, AzureOpenAIConfig
from .exceptions import InvestmentAgentError, MarketDataError, InvalidSymbolError


logger = structlog.get_logger(__name__)


class InvestmentAgent:
    """
    Intelligent investment analysis automation agent.

    Features:
    - Intelligent stock discovery and analysis
    - Context-aware investment recommendations
    - Market trend analysis and insights
    - Multi-source financial data integration (Yahoo, Alpha Vantage, Finnhub)
    - Portfolio analysis and optimization
    - Risk assessment and management
    - Technical and fundamental analysis
    - Automated report generation
    """

    def __init__(
        self,
        config: Optional[InvestmentAgentConfig] = None,
        openai_config: Optional[AzureOpenAIConfig] = None,
    ):
        """
        Initialize the investment agent.

        Args:
            config: Investment agent configuration
            openai_config: Azure OpenAI configuration
        """
        self.config = config or InvestmentAgentConfig(
            openai_config=openai_config or AzureOpenAIConfig.from_environment()
        )
        self.openai_client = AzureOpenAIClient(self.config.openai_config)

        # Cache for results and optimization
        self._stock_cache: Dict[str, Any] = {}
        self._analysis_cache: Dict[str, Any] = {}

        # Initialize external clients
        self._alpha_vantage_client = None
        self._finnhub_client = None

        if self.config.alpha_vantage_api_key:
            self._alpha_vantage_ts = TimeSeries(key=self.config.alpha_vantage_api_key)
            self._alpha_vantage_fd = FundamentalData(
                key=self.config.alpha_vantage_api_key
            )

        if self.config.finnhub_api_key:
            self._finnhub_client = finnhub.Client(api_key=self.config.finnhub_api_key)

        logger.info(
            "Initializing Investment Agent",
            agent_name=self.config.name,
            investment_focus=self.config.investment_focus,
            risk_tolerance=self.config.risk_tolerance,
            max_stocks_per_analysis=self.config.max_stocks_per_analysis,
        )

    async def get_stock_data(
        self,
        symbol: str,
        period: str = "1y",
        source: str = "yahoo",
    ) -> Dict[str, Any]:
        """
        Get stock data from various sources.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period for data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            source: Data source ('yahoo', 'alpha_vantage', 'finnhub')

        Returns:
            Dictionary containing stock data and metadata
        """
        try:
            cache_key = f"{symbol}_{period}_{source}"

            # Check cache first
            if cache_key in self._stock_cache:
                cached_data = self._stock_cache[cache_key]
                cache_time = cached_data.get("timestamp", datetime.min)
                if datetime.now() - cache_time < timedelta(
                    minutes=self.config.cache_duration_minutes
                ):
                    logger.debug(f"Using cached data for {symbol}")
                    return cached_data["data"]

            logger.info(f"Fetching stock data for {symbol} from {source}")

            if source == "yahoo":
                data = await self._get_yahoo_data(symbol, period)
            elif source == "alpha_vantage":
                data = await self._get_alpha_vantage_data(symbol, period)
            elif source == "finnhub":
                data = await self._get_finnhub_data(symbol, period)
            else:
                raise MarketDataError(f"Unsupported data source: {source}")

            # Cache the result
            self._stock_cache[cache_key] = {"data": data, "timestamp": datetime.now()}

            return data

        except Exception as e:
            logger.error(f"Failed to get stock data for {symbol}: {e}")
            raise MarketDataError(f"Failed to get stock data for {symbol}: {e}")

    async def _get_yahoo_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """Get data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)

            # Get historical data
            hist = ticker.history(period=period)
            if hist.empty:
                raise InvalidSymbolError(symbol)

            # Get basic info
            info = ticker.info

            # Get financial data
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow

            return {
                "symbol": symbol,
                "source": "yahoo",
                "historical_data": hist.to_dict(),
                "info": info,
                "financials": financials.to_dict() if not financials.empty else {},
                "balance_sheet": balance_sheet.to_dict()
                if not balance_sheet.empty
                else {},
                "cash_flow": cash_flow.to_dict() if not cash_flow.empty else {},
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            raise MarketDataError(f"Yahoo Finance error: {e}")

    async def _get_alpha_vantage_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """Get data from Alpha Vantage."""
        if not self._alpha_vantage_ts:
            raise MarketDataError("Alpha Vantage API key not configured")

        try:
            # Get daily data
            data, meta_data = self._alpha_vantage_ts.get_daily_adjusted(
                symbol=symbol, outputsize="full"
            )

            # Get company overview
            overview, _ = self._alpha_vantage_fd.get_company_overview(symbol)

            return {
                "symbol": symbol,
                "source": "alpha_vantage",
                "historical_data": data,
                "meta_data": meta_data,
                "company_overview": overview,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Alpha Vantage error for {symbol}: {e}")
            raise MarketDataError(f"Alpha Vantage error: {e}")

    async def _get_finnhub_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """Get data from Finnhub."""
        if not self._finnhub_client:
            raise MarketDataError("Finnhub API key not configured")

        try:
            # Get company profile
            profile = self._finnhub_client.company_profile2(symbol=symbol)

            # Get basic financials
            financials = self._finnhub_client.company_basic_financials(symbol, "all")

            # Get historical data (last year)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)

            candles = self._finnhub_client.stock_candles(
                symbol, "D", int(start_date.timestamp()), int(end_date.timestamp())
            )

            return {
                "symbol": symbol,
                "source": "finnhub",
                "profile": profile,
                "financials": financials,
                "historical_data": candles,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Finnhub error for {symbol}: {e}")
            raise MarketDataError(f"Finnhub error: {e}")

    async def analyze_stock(
        self,
        symbol: str,
        analysis_type: str = "comprehensive",
        include_technical: bool = True,
        include_fundamental: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive stock analysis.

        Args:
            symbol: Stock symbol to analyze
            analysis_type: Type of analysis (brief, comprehensive, detailed)
            include_technical: Whether to include technical analysis
            include_fundamental: Whether to include fundamental analysis

        Returns:
            Complete stock analysis results
        """
        try:
            logger.info(f"Starting stock analysis for {symbol}")

            # Get stock data from primary source
            stock_data = await self.get_stock_data(symbol, period="1y")

            analysis_results = {
                "symbol": symbol,
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat(),
                "stock_data": stock_data,
            }

            # Perform technical analysis if requested
            if include_technical:
                technical_analysis = await self._perform_technical_analysis(stock_data)
                analysis_results["technical_analysis"] = technical_analysis

            # Perform fundamental analysis if requested
            if include_fundamental:
                fundamental_analysis = await self._perform_fundamental_analysis(
                    stock_data
                )
                analysis_results["fundamental_analysis"] = fundamental_analysis

            # Generate AI-powered investment recommendation
            ai_analysis = await self._generate_ai_analysis(symbol, analysis_results)
            analysis_results["ai_analysis"] = ai_analysis

            # Calculate overall score and recommendation
            overall_assessment = await self._calculate_overall_assessment(
                analysis_results
            )
            analysis_results["overall_assessment"] = overall_assessment

            logger.info(f"Completed stock analysis for {symbol}")
            return analysis_results

        except Exception as e:
            logger.error(f"Stock analysis failed for {symbol}: {e}")
            raise InvestmentAgentError(f"Stock analysis failed: {e}")

    async def _perform_technical_analysis(
        self, stock_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform technical analysis on stock data."""
        try:
            # Extract historical data
            if stock_data["source"] == "yahoo":
                hist_data = pd.DataFrame(stock_data["historical_data"])
                if hist_data.empty:
                    return {"error": "No historical data available"}

                # Calculate technical indicators
                close_prices = hist_data["Close"]

                # Simple Moving Averages
                sma_20 = close_prices.rolling(window=20).mean()
                sma_50 = close_prices.rolling(window=50).mean()
                sma_200 = close_prices.rolling(window=200).mean()

                # Relative Strength Index (RSI)
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))

                # MACD
                ema_12 = close_prices.ewm(span=12).mean()
                ema_26 = close_prices.ewm(span=26).mean()
                macd = ema_12 - ema_26
                signal = macd.ewm(span=9).mean()

                # Current values
                current_price = close_prices.iloc[-1]
                current_rsi = rsi.iloc[-1]
                current_macd = macd.iloc[-1]
                current_signal = signal.iloc[-1]

                return {
                    "current_price": current_price,
                    "sma_20": sma_20.iloc[-1],
                    "sma_50": sma_50.iloc[-1],
                    "sma_200": sma_200.iloc[-1],
                    "rsi": current_rsi,
                    "macd": current_macd,
                    "macd_signal": current_signal,
                    "trend_analysis": {
                        "short_term": "bullish"
                        if current_price > sma_20.iloc[-1]
                        else "bearish",
                        "medium_term": "bullish"
                        if current_price > sma_50.iloc[-1]
                        else "bearish",
                        "long_term": "bullish"
                        if current_price > sma_200.iloc[-1]
                        else "bearish",
                    },
                    "momentum": {
                        "rsi_signal": "overbought"
                        if current_rsi > 70
                        else "oversold"
                        if current_rsi < 30
                        else "neutral",
                        "macd_signal": "bullish"
                        if current_macd > current_signal
                        else "bearish",
                    },
                }

        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return {"error": f"Technical analysis failed: {e}"}

    async def _perform_fundamental_analysis(
        self, stock_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform fundamental analysis on stock data."""
        try:
            if stock_data["source"] == "yahoo":
                info = stock_data.get("info", {})

                # Extract key fundamental metrics
                pe_ratio = info.get("trailingPE")
                pb_ratio = info.get("priceToBook")
                roe = info.get("returnOnEquity")
                profit_margin = info.get("profitMargins")
                revenue_growth = info.get("revenueGrowth")
                earnings_growth = info.get("earningsGrowth")
                debt_to_equity = info.get("debtToEquity")
                current_ratio = info.get("currentRatio")

                # Calculate fundamental score
                fundamental_score = 0
                max_score = 8

                if pe_ratio and 5 <= pe_ratio <= 25:
                    fundamental_score += 1
                if pb_ratio and pb_ratio <= 3:
                    fundamental_score += 1
                if roe and roe >= 0.15:
                    fundamental_score += 1
                if profit_margin and profit_margin >= 0.10:
                    fundamental_score += 1
                if revenue_growth and revenue_growth >= 0.05:
                    fundamental_score += 1
                if earnings_growth and earnings_growth >= 0.05:
                    fundamental_score += 1
                if debt_to_equity and debt_to_equity <= 0.5:
                    fundamental_score += 1
                if current_ratio and current_ratio >= 1.5:
                    fundamental_score += 1

                return {
                    "pe_ratio": pe_ratio,
                    "pb_ratio": pb_ratio,
                    "roe": roe,
                    "profit_margin": profit_margin,
                    "revenue_growth": revenue_growth,
                    "earnings_growth": earnings_growth,
                    "debt_to_equity": debt_to_equity,
                    "current_ratio": current_ratio,
                    "fundamental_score": fundamental_score,
                    "max_score": max_score,
                    "score_percentage": (fundamental_score / max_score) * 100,
                    "valuation": "undervalued"
                    if fundamental_score >= 6
                    else "overvalued"
                    if fundamental_score <= 3
                    else "fairly_valued",
                }

        except Exception as e:
            logger.error(f"Fundamental analysis failed: {e}")
            return {"error": f"Fundamental analysis failed: {e}"}

    async def _generate_ai_analysis(
        self, symbol: str, analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate AI-powered investment analysis."""
        try:
            analysis_request = f"Provide investment analysis for {symbol} stock"

            # Prepare context data for AI analysis
            context_data = {
                "symbol": symbol,
                "technical_analysis": analysis_data.get("technical_analysis", {}),
                "fundamental_analysis": analysis_data.get("fundamental_analysis", {}),
                "stock_info": analysis_data.get("stock_data", {}).get("info", {}),
            }

            ai_response = await self.openai_client.investment_analysis(
                analysis_request=analysis_request,
                context_data=context_data,
                analysis_type=analysis_data.get("analysis_type", "comprehensive"),
            )

            return {
                "ai_recommendation": ai_response,
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"AI analysis failed for {symbol}: {e}")
            return {"error": f"AI analysis failed: {e}"}

    async def _calculate_overall_assessment(
        self, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall investment assessment."""
        try:
            technical_score = 0
            fundamental_score = 0

            # Technical score (0-100)
            tech_analysis = analysis_results.get("technical_analysis", {})
            if "trend_analysis" in tech_analysis:
                trends = tech_analysis["trend_analysis"]
                for trend in trends.values():
                    if trend == "bullish":
                        technical_score += 33.33

            # Fundamental score (0-100)
            fund_analysis = analysis_results.get("fundamental_analysis", {})
            if "score_percentage" in fund_analysis:
                fundamental_score = fund_analysis["score_percentage"]

            # Overall score (weighted average)
            overall_score = technical_score * 0.4 + fundamental_score * 0.6

            # Determine recommendation
            if overall_score >= 70:
                recommendation = "BUY"
                confidence = "High"
            elif overall_score >= 50:
                recommendation = "HOLD"
                confidence = "Medium"
            else:
                recommendation = "SELL"
                confidence = "Low"

            return {
                "technical_score": technical_score,
                "fundamental_score": fundamental_score,
                "overall_score": overall_score,
                "recommendation": recommendation,
                "confidence": confidence,
                "risk_level": self._assess_risk_level(analysis_results),
            }

        except Exception as e:
            logger.error(f"Overall assessment calculation failed: {e}")
            return {"error": f"Assessment calculation failed: {e}"}

    def _assess_risk_level(self, analysis_results: Dict[str, Any]) -> str:
        """Assess the risk level of the investment."""
        try:
            risk_factors = 0

            # Check various risk factors
            stock_info = analysis_results.get("stock_data", {}).get("info", {})

            # Volatility (beta)
            beta = stock_info.get("beta")
            if beta and beta > 1.5:
                risk_factors += 1

            # Debt levels
            debt_to_equity = stock_info.get("debtToEquity")
            if debt_to_equity and debt_to_equity > 0.5:
                risk_factors += 1

            # Profitability
            profit_margin = stock_info.get("profitMargins")
            if not profit_margin or profit_margin < 0:
                risk_factors += 1

            # Market cap (smaller companies are riskier)
            market_cap = stock_info.get("marketCap")
            if market_cap and market_cap < 1000000000:  # Less than 1B
                risk_factors += 1

            if risk_factors >= 3:
                return "High"
            elif risk_factors >= 2:
                return "Medium"
            else:
                return "Low"

        except Exception:
            return "Unknown"

    async def screen_stocks(
        self,
        criteria: Dict[str, Any],
        universe: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Screen stocks based on specified criteria.

        Args:
            criteria: Screening criteria (PE ratio, market cap, etc.)
            universe: List of symbols to screen (defaults to S&P 500)

        Returns:
            List of stocks meeting the criteria
        """
        try:
            logger.info("Starting stock screening")

            # Default universe (simplified - in practice would use actual S&P 500 list)
            if not universe:
                universe = [
                    "AAPL",
                    "MSFT",
                    "GOOGL",
                    "AMZN",
                    "TSLA",
                    "META",
                    "NVDA",
                    "JPM",
                    "V",
                    "JNJ",
                ]

            screened_stocks = []

            for symbol in universe[: self.config.max_stocks_per_analysis]:
                try:
                    stock_data = await self.get_stock_data(symbol)

                    if self._meets_criteria(stock_data, criteria):
                        screened_stocks.append(
                            {
                                "symbol": symbol,
                                "score": self._calculate_screening_score(
                                    stock_data, criteria
                                ),
                                "data": stock_data,
                            }
                        )

                except Exception as e:
                    logger.warning(f"Failed to screen {symbol}: {e}")
                    continue

            # Sort by score
            screened_stocks.sort(key=lambda x: x["score"], reverse=True)

            logger.info(
                f"Screening completed. Found {len(screened_stocks)} stocks meeting criteria"
            )
            return screened_stocks

        except Exception as e:
            logger.error(f"Stock screening failed: {e}")
            raise InvestmentAgentError(f"Stock screening failed: {e}")

    def _meets_criteria(
        self, stock_data: Dict[str, Any], criteria: Dict[str, Any]
    ) -> bool:
        """Check if stock meets screening criteria."""
        try:
            info = stock_data.get("info", {})

            # Check each criterion
            for key, value in criteria.items():
                stock_value = info.get(key)

                if stock_value is None:
                    continue

                if isinstance(value, dict):
                    # Range criteria
                    min_val = value.get("min")
                    max_val = value.get("max")

                    if min_val is not None and stock_value < min_val:
                        return False
                    if max_val is not None and stock_value > max_val:
                        return False
                else:
                    # Exact match criteria
                    if stock_value != value:
                        return False

            return True

        except Exception:
            return False

    def _calculate_screening_score(
        self, stock_data: Dict[str, Any], criteria: Dict[str, Any]
    ) -> float:
        """Calculate a score for screened stocks."""
        # Simplified scoring - would be more sophisticated in practice
        score = 0
        info = stock_data.get("info", {})

        # Score based on common investment metrics
        pe_ratio = info.get("trailingPE")
        if pe_ratio and 10 <= pe_ratio <= 20:
            score += 25

        roe = info.get("returnOnEquity")
        if roe and roe >= 0.15:
            score += 25

        revenue_growth = info.get("revenueGrowth")
        if revenue_growth and revenue_growth >= 0.05:
            score += 25

        profit_margin = info.get("profitMargins")
        if profit_margin and profit_margin >= 0.10:
            score += 25

        return score
