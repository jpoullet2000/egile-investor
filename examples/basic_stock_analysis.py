"""
Basic Stock Analysis Example

This example demonstrates how to use the Egile Investor package
to perform basic stock analysis.
"""

import asyncio
from egile_investor import InvestmentAgent, InvestmentAgentConfig, AzureOpenAIConfig


async def basic_stock_analysis():
    """Perform basic stock analysis on a few stocks."""

    # Create configuration
    config = InvestmentAgentConfig(
        name="BasicAnalysisAgent",
        investment_focus=["stocks"],
        risk_tolerance="moderate",
        openai_config=AzureOpenAIConfig.from_environment(),
    )

    # Initialize agent
    agent = InvestmentAgent(config=config)

    # Stocks to analyze
    symbols = ["AAPL", "MSFT", "GOOGL"]

    print("=== Basic Stock Analysis ===\n")

    for symbol in symbols:
        try:
            print(f"Analyzing {symbol}...")

            # Get basic stock data
            stock_data = await agent.get_stock_data(symbol)
            current_price = stock_data.get("info", {}).get("currentPrice", "N/A")

            print(f"{symbol} Current Price: ${current_price}")

            # Perform comprehensive analysis
            analysis = await agent.analyze_stock(
                symbol=symbol,
                analysis_type="brief",
                include_technical=True,
                include_fundamental=True,
            )

            # Display key results
            overall_assessment = analysis.get("overall_assessment", {})
            recommendation = overall_assessment.get("recommendation", "HOLD")
            confidence = overall_assessment.get("confidence", "Medium")
            risk_level = overall_assessment.get("risk_level", "Medium")

            print(f"{symbol} Recommendation: {recommendation}")
            print(f"{symbol} Confidence: {confidence}")
            print(f"{symbol} Risk Level: {risk_level}")

            # Show technical indicators
            tech_analysis = analysis.get("technical_analysis", {})
            if "rsi" in tech_analysis:
                rsi = tech_analysis["rsi"]
                print(f"{symbol} RSI: {rsi:.2f}")

            # Show fundamental metrics
            fund_analysis = analysis.get("fundamental_analysis", {})
            if "pe_ratio" in fund_analysis:
                pe_ratio = fund_analysis["pe_ratio"]
                print(f"{symbol} P/E Ratio: {pe_ratio}")

            print("-" * 50)

        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            continue

    print("\n=== Analysis Complete ===")


async def compare_stocks():
    """Compare multiple stocks side by side."""

    agent = InvestmentAgent()
    symbols = ["AAPL", "MSFT"]

    print("\n=== Stock Comparison ===\n")

    results = {}
    for symbol in symbols:
        try:
            analysis = await agent.analyze_stock(symbol, analysis_type="brief")
            results[symbol] = analysis.get("overall_assessment", {})
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")

    # Display comparison
    print(f"{'Stock':<10} {'Recommendation':<15} {'Risk Level':<12} {'Confidence':<12}")
    print("-" * 55)

    for symbol, assessment in results.items():
        recommendation = assessment.get("recommendation", "N/A")
        risk_level = assessment.get("risk_level", "N/A")
        confidence = assessment.get("confidence", "N/A")

        print(f"{symbol:<10} {recommendation:<15} {risk_level:<12} {confidence:<12}")


async def main():
    """Main function to run all examples."""
    await basic_stock_analysis()
    await compare_stocks()


if __name__ == "__main__":
    asyncio.run(main())
