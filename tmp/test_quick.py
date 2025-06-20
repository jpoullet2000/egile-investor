#!/usr/bin/env python3

import asyncio
from egile_investor.ai_agent import AIInvestmentAgent
from egile_investor.config import InvestmentAgentConfig


async def test_generate_report():
    """Test the generate_report tool."""
    config = InvestmentAgentConfig()
    agent = AIInvestmentAgent(config)

    try:
        await agent.connect()
        print("Connected to MCP server")

        # Test generate_report with string data
        result = await agent.call_tool(
            "generate_report",
            {
                "analysis_type": "stock_analysis",
                "data": "Test placeholder data from previous analysis steps",
                "format_type": "markdown",
            },
        )

        print("✓ generate_report test successful")
        print(f"  Title: {result.get('title', 'No title')}")
        print(f"  Content length: {len(str(result))}")

        # Test with dict data
        result2 = await agent.call_tool(
            "generate_report",
            {
                "analysis_type": "portfolio_review",
                "data": {"test": "data", "analysis": "results"},
                "format_type": "html",
            },
        )

        print("✓ generate_report with dict data successful")
        print(f"  Title: {result2.get('title', 'No title')}")

    except Exception as e:
        print(f"✗ Test failed: {e}")
    finally:
        if agent.session:
            await agent.disconnect()


async def test_multi_symbol_tool():
    """Test a multi-symbol tool with placeholder input."""
    config = InvestmentAgentConfig()
    agent = AIInvestmentAgent(config)

    try:
        await agent.connect()

        # Test analyze_multiple_stocks with placeholder
        result = await agent.call_tool(
            "analyze_multiple_stocks",
            {"symbols": "<symbols_from_previous_step>", "analysis_type": "brief"},
        )

        print("✓ analyze_multiple_stocks with placeholder successful")
        print(f"  Result type: {type(result)}")

    except Exception as e:
        print(f"✗ Multi-symbol test failed: {e}")
    finally:
        if agent.session:
            await agent.disconnect()


async def main():
    print("Testing improved MCP tools...")
    await test_generate_report()
    print()
    await test_multi_symbol_tool()
    print("Tests complete!")


if __name__ == "__main__":
    asyncio.run(main())
