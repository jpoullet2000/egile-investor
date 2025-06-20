#!/usr/bin/env python3
"""Test script to debug AI agent context passing."""

import asyncio
import sys

# Add the project root to the path
sys.path.insert(0, "/home/jbp/projects/egile-investor")

from egile_investor.ai_agent import AIInvestmentAgent


async def test_ai_context_passing():
    """Test the AI agent's ability to handle context passing between steps."""
    print("Testing AI agent context passing...")

    try:
        async with AIInvestmentAgent() as agent:
            print("AI agent initialized successfully")

            # Test a simple scenario that should screen stocks first
            task = "Find the best technology stocks with P/E ratio under 25 and ROE above 15%"
            print(f"\nTesting task: {task}")

            result = await agent.analyze(task)

            print("\nTask execution completed:")
            print(f"Total steps: {result['summary']['total_steps']}")
            print(f"Successful steps: {result['summary']['successful_steps']}")
            print(f"Failed steps: {result['summary']['failed_steps']}")
            print(f"Tools used: {', '.join(result['summary']['tools_used'])}")

            # Show details of each step
            for i, step_result in enumerate(result["execution_results"], 1):
                print(f"\nStep {i}: {step_result['description']}")
                print(f"  Tool: {step_result['tool']}")
                print(f"  Success: {step_result.get('success', False)}")
                if step_result.get("success"):
                    print(f"  Arguments: {step_result.get('arguments', {})}")
                    if step_result["tool"] == "screen_stocks":
                        result_data = step_result.get("result", [])
                        if isinstance(result_data, list) and len(result_data) > 0:
                            print(f"  Found {len(result_data)} stocks")
                            for j, stock in enumerate(result_data[:3]):
                                if isinstance(stock, dict) and "symbol" in stock:
                                    print(
                                        f"    {j + 1}. {stock['symbol']} (score: {stock.get('score', 'N/A')})"
                                    )
                else:
                    error_msg = step_result.get("error", "Unknown error")
                    print(f"  Error: {error_msg[:100]}...")

    except Exception as e:
        print(f"Error in test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_ai_context_passing())
