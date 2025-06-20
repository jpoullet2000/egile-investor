"""
Stock Screening Example

This example demonstrates how to screen stocks based on
various financial criteria.
"""

import asyncio
from egile_investor import InvestmentAgent


async def basic_screening():
    """Perform basic stock screening."""

    print("=== Basic Stock Screening ===\n")

    agent = InvestmentAgent()

    # Define screening criteria
    criteria = {
        "trailingPE": {"min": 5, "max": 25},  # P/E ratio between 5 and 25
        "returnOnEquity": {"min": 0.15},  # ROE above 15%
        "profitMargins": {"min": 0.10},  # Profit margin above 10%
        "marketCap": {"min": 1000000000},  # Market cap above 1B
    }

    print("Screening Criteria:")
    for key, value in criteria.items():
        if isinstance(value, dict):
            range_str = f"{value.get('min', 'N/A')} to {value.get('max', 'N/A')}"
            print(f"  {key}: {range_str}")
        else:
            print(f"  {key}: {value}")

    print("\nScreening stocks...\n")

    try:
        # Perform screening
        results = await agent.screen_stocks(criteria=criteria)

        if results:
            print(f"Found {len(results)} stocks meeting criteria:\n")

            # Display results
            print(f"{'Symbol':<8} {'Score':<8} {'Company':<30}")
            print("-" * 50)

            for stock in results:
                symbol = stock["symbol"]
                score = stock["score"]
                company_name = (
                    stock.get("data", {}).get("info", {}).get("longName", "N/A")
                )

                # Truncate long company names
                if len(company_name) > 30:
                    company_name = company_name[:27] + "..."

                print(f"{symbol:<8} {score:<8.1f} {company_name:<30}")
        else:
            print("No stocks found meeting the criteria.")

    except Exception as e:
        print(f"Screening failed: {e}")


async def value_investing_screen():
    """Screen for value investing opportunities."""

    print("\n=== Value Investing Screen ===\n")

    agent = InvestmentAgent()

    # Value investing criteria
    criteria = {
        "trailingPE": {"min": 5, "max": 15},  # Low P/E ratio
        "priceToBook": {"max": 2.0},  # Low P/B ratio
        "returnOnEquity": {"min": 0.12},  # Good ROE
        "debtToEquity": {"max": 0.5},  # Low debt
        "currentRatio": {"min": 1.5},  # Good liquidity
    }

    print("Value Investing Criteria:")
    print("  - P/E Ratio: 5-15 (undervalued)")
    print("  - P/B Ratio: < 2.0 (book value)")
    print("  - ROE: > 12% (profitable)")
    print("  - Debt/Equity: < 0.5 (low debt)")
    print("  - Current Ratio: > 1.5 (liquid)")

    try:
        results = await agent.screen_stocks(criteria=criteria)

        print(f"\nFound {len(results)} value opportunities:")

        for stock in results:
            symbol = stock["symbol"]
            data = stock.get("data", {}).get("info", {})

            print(f"\n{symbol}:")
            print(f"  P/E: {data.get('trailingPE', 'N/A')}")
            print(f"  P/B: {data.get('priceToBook', 'N/A')}")
            print(f"  ROE: {data.get('returnOnEquity', 'N/A')}")
            print(f"  Debt/Equity: {data.get('debtToEquity', 'N/A')}")
            print(f"  Current Ratio: {data.get('currentRatio', 'N/A')}")
            print(f"  Score: {stock['score']}")

    except Exception as e:
        print(f"Value screening failed: {e}")


async def growth_investing_screen():
    """Screen for growth investing opportunities."""

    print("\n=== Growth Investing Screen ===\n")

    agent = InvestmentAgent()

    # Growth investing criteria
    criteria = {
        "revenueGrowth": {"min": 0.10},  # Revenue growth > 10%
        "earningsGrowth": {"min": 0.15},  # Earnings growth > 15%
        "returnOnEquity": {"min": 0.15},  # Strong ROE
        "profitMargins": {"min": 0.15},  # High profit margins
        "trailingPE": {"max": 40},  # Reasonable P/E for growth
    }

    print("Growth Investing Criteria:")
    print("  - Revenue Growth: > 10%")
    print("  - Earnings Growth: > 15%")
    print("  - ROE: > 15%")
    print("  - Profit Margins: > 15%")
    print("  - P/E Ratio: < 40")

    try:
        results = await agent.screen_stocks(criteria=criteria)

        print(f"\nFound {len(results)} growth opportunities:")

        for stock in results:
            symbol = stock["symbol"]
            data = stock.get("data", {}).get("info", {})

            print(f"\n{symbol}:")
            print(f"  Revenue Growth: {data.get('revenueGrowth', 'N/A')}")
            print(f"  Earnings Growth: {data.get('earningsGrowth', 'N/A')}")
            print(f"  ROE: {data.get('returnOnEquity', 'N/A')}")
            print(f"  Profit Margins: {data.get('profitMargins', 'N/A')}")
            print(f"  P/E: {data.get('trailingPE', 'N/A')}")
            print(f"  Score: {stock['score']}")

    except Exception as e:
        print(f"Growth screening failed: {e}")


async def dividend_screen():
    """Screen for dividend-paying stocks."""

    print("\n=== Dividend Stock Screen ===\n")

    agent = InvestmentAgent()

    # Dividend criteria
    criteria = {
        "dividendYield": {"min": 0.02},  # Dividend yield > 2%
        "payoutRatio": {"max": 0.70},  # Sustainable payout
        "returnOnEquity": {"min": 0.10},  # Profitable
        "debtToEquity": {"max": 0.6},  # Manageable debt
        "currentRatio": {"min": 1.2},  # Good liquidity
    }

    print("Dividend Stock Criteria:")
    print("  - Dividend Yield: > 2%")
    print("  - Payout Ratio: < 70% (sustainable)")
    print("  - ROE: > 10%")
    print("  - Debt/Equity: < 60%")
    print("  - Current Ratio: > 1.2")

    try:
        results = await agent.screen_stocks(criteria=criteria)

        print(f"\nFound {len(results)} dividend opportunities:")

        for stock in results:
            symbol = stock["symbol"]
            data = stock.get("data", {}).get("info", {})

            print(f"\n{symbol}:")
            print(f"  Dividend Yield: {data.get('dividendYield', 'N/A')}")
            print(f"  Payout Ratio: {data.get('payoutRatio', 'N/A')}")
            print(f"  ROE: {data.get('returnOnEquity', 'N/A')}")
            print(f"  Debt/Equity: {data.get('debtToEquity', 'N/A')}")
            print(f"  Score: {stock['score']}")

    except Exception as e:
        print(f"Dividend screening failed: {e}")


async def main():
    """Main function to run all screening examples."""
    await basic_screening()
    await value_investing_screen()
    await growth_investing_screen()
    await dividend_screen()


if __name__ == "__main__":
    asyncio.run(main())
