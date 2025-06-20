#!/usr/bin/env python3
"""Debug script to test MCP server connection."""

import asyncio
import sys
import subprocess
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def test_mcp_connection():
    """Test the MCP server connection directly."""
    print("Testing MCP server connection...")

    try:
        # Test if server can be started as subprocess
        print("Testing server subprocess startup...")
        proc = subprocess.Popen(
            [sys.executable, "-m", "egile_investor.server"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
        )

        # Give it a moment to start
        await asyncio.sleep(1)

        if proc.poll() is None:
            print("Server process started successfully")
            proc.terminate()
            proc.wait()
        else:
            stdout, stderr = proc.communicate()
            print(f"Server failed to start. Exit code: {proc.returncode}")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return

    except Exception as e:
        print(f"Error testing subprocess: {e}")
        return

    try:
        # Test MCP client connection
        print("Testing MCP client connection...")
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "egile_investor.server"],
            env=None,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                print("MCP session created")

                # Initialize the session
                await session.initialize()
                print("MCP session initialized")

                # List available tools
                tools_response = await session.list_tools()
                print(f"Available tools: {len(tools_response.tools)}")

                for tool in tools_response.tools:
                    print(f"  - {tool.name}: {tool.description}")

                # Test calling a simple tool
                if tools_response.tools:
                    tool_name = tools_response.tools[0].name
                    print(f"Testing tool call: {tool_name}")

                    # Try calling the tool with minimal arguments
                    try:
                        if tool_name == "analyze_stock":
                            result = await session.call_tool(
                                tool_name, {"symbol": "AAPL"}
                            )
                            print(f"Tool call successful: {type(result)}")
                            print(f"Result content type: {type(result.content)}")
                            print(f"Result content length: {len(str(result.content))}")
                            if (
                                hasattr(result.content, "__len__")
                                and len(str(result.content)) > 500
                            ):
                                print(
                                    f"Result content (first 500 chars): {str(result.content)[:500]}..."
                                )
                            else:
                                print(f"Result content: {result.content}")

                            # Test another tool
                            print("\nTesting get_market_data...")
                            result2 = await session.call_tool(
                                "get_market_data", {"symbol": "AAPL"}
                            )
                            print(f"get_market_data successful: {type(result2)}")
                            print(
                                f"get_market_data content type: {type(result2.content)}"
                            )

                            # Test the new screening and multi-symbol tools
                            print("\nTesting screen_stocks...")
                            screening_result = await session.call_tool(
                                "screen_stocks",
                                {
                                    "criteria": {
                                        "pe_ratio": {"max": 25},
                                        "roe": {"min": 0.15},
                                    },
                                    "max_results": 3,
                                },
                            )
                            print(f"screen_stocks successful: {type(screening_result)}")

                            # Parse the screening results
                            if (
                                hasattr(screening_result.content, "__len__")
                                and len(screening_result.content) > 0
                            ):
                                screening_data = screening_result.content[0].text
                                print(f"Screening data length: {len(screening_data)}")

                                print("\nTesting get_screening_symbols...")
                                symbols_result = await session.call_tool(
                                    "get_screening_symbols",
                                    {
                                        "screening_results": screening_data,
                                        "max_symbols": 2,
                                    },
                                )
                                print(
                                    f"get_screening_symbols successful: {type(symbols_result)}"
                                )
                                if (
                                    hasattr(symbols_result.content, "__len__")
                                    and len(symbols_result.content) > 0
                                ):
                                    symbols_data = symbols_result.content[0].text
                                    print(f"Extracted symbols: {symbols_data[:200]}")

                                    # Test multi-symbol analysis
                                    print("\nTesting analyze_multiple_stocks...")
                                    multi_result = await session.call_tool(
                                        "analyze_multiple_stocks",
                                        {
                                            "symbols": ["AAPL", "MSFT"],
                                            "analysis_type": "brief",
                                        },
                                    )
                                    print(
                                        f"analyze_multiple_stocks successful: {type(multi_result)}"
                                    )

                        elif tool_name == "get_market_data":
                            result = await session.call_tool(
                                tool_name, {"symbol": "AAPL"}
                            )
                            print(f"Tool call successful: {type(result)}")
                            print(f"Result content: {result.content}")
                        else:
                            print(f"Skipping test for unknown tool: {tool_name}")

                    except Exception as e:
                        print(f"Tool call failed: {e}")
                        print(f"Exception type: {type(e)}")
                        import traceback

                        traceback.print_exc()

    except Exception as e:
        print(f"MCP connection failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_mcp_connection())
