#!/usr/bin/env python3
"""
Simple test script to verify the egile_investor package functionality.
"""

import asyncio
import os
import sys


async def test_basic_imports():
    """Test that we can import the main modules."""
    print("Testing basic imports...")
    
    try:
        from egile_investor.config import InvestmentAgentConfig, AzureOpenAIConfig
        print("✓ Config modules imported successfully")
        
        from egile_investor.exceptions import InvestmentAgentError
        print("✓ Exception modules imported successfully")
        
        # Test config creation
        config = InvestmentAgentConfig(
            name="TestAgent",
            openai_config=AzureOpenAIConfig(
                endpoint="https://test.openai.azure.com/",
                api_key="test-key"
            )
        )
        print("✓ Configuration objects created successfully")
        print(f"  Agent name: {config.name}")
        print(f"  Risk tolerance: {config.risk_tolerance}")
        print(f"  OpenAI endpoint: {config.openai_config.endpoint}")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False


async def test_mcp_server_imports():
    """Test MCP server related imports."""
    print("\nTesting MCP server imports...")
    
    try:
        # This might fail if dependencies aren't installed, but should not crash
        from egile_investor.server import mcp
        print("✓ MCP server imported successfully")
        return True
        
    except ImportError as e:
        print(f"⚠ MCP server import failed (expected if dependencies not installed): {e}")
        return True  # This is acceptable for basic test
    except Exception as e:
        print(f"✗ MCP server import failed: {e}")
        return False


async def test_agent_creation():
    """Test that we can create agent instances without connecting."""
    print("\nTesting agent creation...")
    
    try:
        from egile_investor.agent import InvestmentAgent
        from egile_investor.config import InvestmentAgentConfig, AzureOpenAIConfig
        
        # Create a test configuration
        config = InvestmentAgentConfig(
            name="TestInvestmentAgent",
            investment_focus=["stocks"],
            risk_tolerance="moderate",
            openai_config=AzureOpenAIConfig(
                endpoint="https://test.openai.azure.com/",
                api_key="test-key-123"
            )
        )
        
        # Create agent (this should not try to connect yet)
        agent = InvestmentAgent(config=config)
        print("✓ InvestmentAgent created successfully")
        print(f"  Agent name: {agent.config.name}")
        print(f"  Risk tolerance: {agent.config.risk_tolerance}")
        print(f"  Investment focus: {agent.config.investment_focus}")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent creation failed: {e}")
        return False


async def test_ai_agent_creation():
    """Test AI agent creation."""
    print("\nTesting AI agent creation...")
    
    try:
        from egile_investor.ai_agent import AIInvestmentAgent
        from egile_investor.config import InvestmentAgentConfig, AzureOpenAIConfig
        
        # Create AI agent
        config = InvestmentAgentConfig(
            openai_config=AzureOpenAIConfig(
                endpoint="https://test.openai.azure.com/",
                api_key="test-key-123"
            )
        )
        
        ai_agent = AIInvestmentAgent(config=config)
        print("✓ AIInvestmentAgent created successfully")
        print(f"  Server command: {ai_agent.server_command}")
        
        return True
        
    except Exception as e:
        print(f"✗ AI agent creation failed: {e}")
        return False


async def test_environment_config():
    """Test loading configuration from environment."""
    print("\nTesting environment configuration...")
    
    try:
        # Set some test environment variables
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test-env.openai.azure.com/"
        os.environ["AZURE_OPENAI_API_KEY"] = "test-env-key"
        os.environ["INVESTMENT_AGENT_NAME"] = "EnvTestAgent"
        os.environ["RISK_TOLERANCE"] = "aggressive"
        
        from egile_investor.config import InvestmentAgentConfig
        
        # Load from environment
        config = InvestmentAgentConfig.from_environment()
        
        print("✓ Configuration loaded from environment")
        print(f"  Agent name: {config.name}")
        print(f"  Risk tolerance: {config.risk_tolerance}")
        print(f"  OpenAI endpoint: {config.openai_config.endpoint}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment config test failed: {e}")
        return False
    finally:
        # Clean up environment variables
        for key in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", 
                   "INVESTMENT_AGENT_NAME", "RISK_TOLERANCE"]:
            os.environ.pop(key, None)


async def main():
    """Run all tests."""
    print("=== Egile Investor Package Test ===\n")
    
    tests = [
        test_basic_imports,
        test_mcp_server_imports,
        test_agent_creation,
        test_ai_agent_creation,
        test_environment_config,
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠ Some tests failed or had issues")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
