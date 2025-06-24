#!/usr/bin/env python3
"""Debug script to understand the generic MCP agent execution."""

import asyncio
from egile_investor.generic_mcp_agent import GenericMCPAgent
from egile_investor.config import InvestmentAgentConfig, AzureOpenAIConfig


async def debug_mcp_execution():
    """Debug the MCP execution step by step."""

    config = InvestmentAgentConfig(openai_config=AzureOpenAIConfig.from_environment())

    agent = GenericMCPAgent(
        config=config, server_command="python -m egile_investor.server"
    )

    try:
        # Connect to the server
        await agent.connect()

        # Print available tools
        print("=== AVAILABLE TOOLS ===")
        for tool_name, tool in agent.available_tools.items():
            print(f"- {tool_name}: {tool.description}")

        # Create an execution plan
        user_request = "Which stocks should I buy with 1000 euros? Create a summary report of maximum 1000 words."
        print(f"\n=== USER REQUEST ===\n{user_request}")

        plan = await agent._create_execution_plan(user_request)
        print("\n=== EXECUTION PLAN ===")
        for step in plan:
            print(f"Step {step['step']}: {step['tool']} - {step['description']}")
            print(f"  Arguments: {step.get('arguments', {})}")

        # Execute the plan step by step with detailed logging
        print("\n=== EXECUTING PLAN ===")
        execution = await agent._execute_plan(plan)

        print("\n=== EXECUTION RESULTS ===")
        print(f"Success rate: {execution['success_count']}/{execution['total_steps']}")

        for log in execution["execution_log"]:
            print(f"\nStep {log['step']}: {log['tool']}")
            print(f"  Success: {log['success']}")
            if log["success"]:
                print(f"  Result preview: {log.get('result_preview', 'No preview')}")
            else:
                print(f"  Error: {log.get('error', 'Unknown error')}")

        # Print final results
        print("\n=== DETAILED STEP RESULTS ===")
        for step_num, result in execution["step_results"].items():
            print(f"\nStep {step_num}:")
            if isinstance(result, dict):
                if "error" in result:
                    print(f"  ERROR: {result['error']}")
                else:
                    # Print key fields from the result
                    for key, value in result.items():
                        if key in ["recommendations", "stocks", "symbols", "symbol"]:
                            print(f"  {key}: {value}")
                        elif isinstance(value, (list, dict)) and len(str(value)) > 100:
                            print(
                                f"  {key}: {type(value).__name__} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})"
                            )
                        else:
                            print(f"  {key}: {value}")
            else:
                print(f"  Result: {result}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await agent.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(debug_mcp_execution())
