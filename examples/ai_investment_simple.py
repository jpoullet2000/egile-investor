"""
AI-Powered Investment Analysis Example

This example demonstrates how to use the AI Investment Agent
for intelligent, automated investment analysis.
"""

import asyncio
from egile_investor import ai_investment_analysis, AIInvestmentAgent


async def simple_ai_analysis():
    """Perform AI-powered investment analysis with a simple query."""

    print("=== Simple AI Investment Analysis ===\n")

    # Simple analysis query
    result = await ai_investment_analysis(
        "Analyze AAPL stock and tell me if I should buy, hold, or sell"
    )

    print("Task:", result["task"])
    print("\nPlan created by AI:")
    for step in result["plan"]:
        print(f"  Step {step['step']}: {step['description']}")

    print("\nExecution Summary:")
    summary = result["summary"]
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Successful: {summary['successful_steps']}")
    print(f"  Failed: {summary['failed_steps']}")
    print(f"  Tools used: {', '.join(summary['tools_used'])}")

    # Show results from successful steps
    successful_results = [r for r in result["execution_results"] if r.get("success")]
    for step_result in successful_results:
        print(f"\n--- {step_result['description']} ---")
        # Display key information from the result
        if isinstance(step_result["result"], dict):
            result_data = step_result["result"]
            if "overall_assessment" in result_data:
                assessment = result_data["overall_assessment"]
                print(f"Recommendation: {assessment.get('recommendation', 'N/A')}")
                print(f"Confidence: {assessment.get('confidence', 'N/A')}")
                print(f"Risk Level: {assessment.get('risk_level', 'N/A')}")


async def complex_ai_analysis():
    """Perform more complex AI-powered analysis."""

    print("\n=== Complex AI Investment Analysis ===\n")

    # More complex analysis
    async with AIInvestmentAgent() as agent:
        # Portfolio analysis
        portfolio_result = await agent.analyze(
            "I have a portfolio with AAPL, MSFT, and GOOGL. "
            "Analyze the risk and suggest if I should rebalance or add new positions"
        )

        print("Portfolio Analysis Task:", portfolio_result["task"])
        print(f"Steps executed: {portfolio_result['summary']['total_steps']}")

        # Stock screening
        screening_result = await agent.analyze(
            "Find the best technology stocks with P/E ratio under 25 and ROE above 15%"
        )

        print("\nStock Screening Task:", screening_result["task"])
        print(f"Steps executed: {screening_result['summary']['total_steps']}")

        # Market analysis
        market_result = await agent.analyze(
            "Analyze the current market conditions and recommend defensive stocks for a conservative investor"
        )

        print("\nMarket Analysis Task:", market_result["task"])
        print(f"Steps executed: {market_result['summary']['total_steps']}")


async def investment_scenarios():
    """Test different investment scenarios."""

    print("\n=== Investment Scenarios ===\n")

    scenarios = [
        "I'm 25 years old and want to invest $10,000 for retirement. What should I buy?",
        "I'm looking for dividend-paying stocks with low volatility",
        "Find growth stocks in the technology sector that are undervalued",
        "I want to hedge my portfolio against market downturns",
        "Compare TSLA vs traditional automotive stocks for investment",
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"Scenario {i}: {scenario}")

        try:
            result = await ai_investment_analysis(scenario)
            summary = result["summary"]

            print(f"  AI created {len(result['plan'])} step plan")
            print(
                f"  Executed {summary['successful_steps']}/{summary['total_steps']} steps successfully"
            )

            if summary["tools_used"]:
                print(f"  Tools used: {', '.join(summary['tools_used'])}")

        except Exception as e:
            print(f"  Error: {e}")

        print("-" * 60)


async def main():
    """Main function to run all AI examples."""
    await simple_ai_analysis()
    await complex_ai_analysis()
    await investment_scenarios()


if __name__ == "__main__":
    asyncio.run(main())
