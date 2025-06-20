#!/usr/bin/env python3
"""
Quick test of the improved AI agent system
"""

import asyncio
from egile_investor.ai_agent import InvestmentAIAgent


async def main():
    agent = InvestmentAIAgent()

    # Test a simple scenario
    result = await agent.analyze("Find dividend-paying stocks with low volatility")

    print("=== Test Results ===")
    print(f"Task: {result['task']}")
    print(f"Plan steps: {len(result['plan'])}")
    print(f"Execution results: {len(result['execution_results'])}")

    # Print step summary
    for i, step_result in enumerate(result["execution_results"], 1):
        success = "✓" if step_result.get("success") else "✗"
        tool = step_result.get("tool", "unknown")
        print(f"  Step {i}: {success} {tool}")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
