#!/usr/bin/env python3
"""
Simple example demonstrating S&P 500 stock screening.
"""

import asyncio
from egile_investor.ai_agent import AIInvestmentAgent


async def screen_sp500_example():
    """Example of screening S&P 500 stocks for value opportunities."""

    print("🔍 Screening S&P 500 for Value Investment Opportunities")
    print("=" * 60)

    async with AIInvestmentAgent() as agent:
        # Screen S&P 500 for value stocks
        criteria = {
            "pe_ratio": {"max": 18},  # P/E ratio under 18
            "dividend_yield": {"min": 0.02},  # Dividend yield above 2%
            "roe": {"min": 0.12},  # Return on Equity above 12%
        }

        print("Screening criteria:")
        print("  - P/E Ratio: ≤ 18")
        print("  - Dividend Yield: ≥ 2%")
        print("  - ROE: ≥ 12%")
        print()

        # Method 1: Using use_sp500 parameter
        print("📊 Method 1: Using use_sp500=True")
        results1 = await agent.call_tool(
            "screen_stocks",
            {"criteria": criteria, "use_sp500": True, "max_results": 10},
        )

        print(f"Found {len(results1)} value opportunities in S&P 500:")
        for i, stock in enumerate(results1, 1):
            print(
                f"{i:2d}. {stock['symbol']:5s} - {stock.get('company_name', 'N/A')[:40]:40s} "
                f"(Score: {stock['score']:.2f}, Sector: {stock.get('sector', 'Unknown')})"
            )

        print()

        # Method 2: Using universe="sp500"
        print("📊 Method 2: Using universe='sp500'")
        results2 = await agent.call_tool(
            "screen_stocks",
            {"criteria": criteria, "universe": "sp500", "max_results": 5},
        )

        print("Top 5 matches using universe parameter:")
        for i, stock in enumerate(results2, 1):
            print(
                f"{i}. {stock['symbol']} - {stock.get('company_name', 'N/A')[:30]} (Score: {stock['score']:.2f})"
            )

        print()

        # Method 3: Compare with major stocks only
        print("📊 Method 3: Compare with major stocks screening")
        results3 = await agent.call_tool(
            "screen_stocks",
            {"criteria": criteria, "universe": "major", "max_results": 5},
        )

        print(f"Major stocks screening found {len(results3)} matches:")
        for i, stock in enumerate(results3, 1):
            print(
                f"{i}. {stock['symbol']} - {stock.get('company_name', 'N/A')[:30]} (Score: {stock['score']:.2f})"
            )


async def main():
    try:
        await screen_sp500_example()
        print("\n✅ S&P 500 screening example completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during screening: {e}")


if __name__ == "__main__":
    asyncio.run(main())
