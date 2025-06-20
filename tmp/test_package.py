"""
Simple test to verify the Egile Investor package structure and imports.
"""

import sys
import os

# Add the package to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_imports():
    """Test that the main package components can be imported."""

    print("Testing package imports...")

    try:
        # Test main package import
        import egile_investor

        print("✓ Main package import successful")

        # Test configuration import
        from egile_investor.config import InvestmentAgentConfig, AzureOpenAIConfig

        print("✓ Configuration classes import successful")

        # Test exceptions import
        from egile_investor.exceptions import InvestmentAgentError, MarketDataError

        print("✓ Exception classes import successful")

        # Test client import
        from egile_investor.client import AzureOpenAIClient

        print("✓ Azure OpenAI client import successful")

        # Test agent import
        from egile_investor.agent import InvestmentAgent

        print("✓ Investment agent import successful")

        # Test AI agent import
        from egile_investor.ai_agent import AIInvestmentAgent

        print("✓ AI investment agent import successful")

        # Test server import
        from egile_investor.server import main

        print("✓ MCP server import successful")

        print("\n✅ All imports successful!")
        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_configuration():
    """Test configuration creation."""

    print("\nTesting configuration...")

    try:
        from egile_investor.config import InvestmentAgentConfig, AzureOpenAIConfig

        # Test Azure OpenAI config
        azure_config = AzureOpenAIConfig(
            endpoint="https://test.openai.azure.com/",
            api_version="2024-12-01-preview",
            default_model="gpt-4",
        )
        print("✓ Azure OpenAI config creation successful")

        # Test investment agent config
        investment_config = InvestmentAgentConfig(
            name="TestAgent",
            investment_focus=["stocks"],
            risk_tolerance="moderate",
            openai_config=azure_config,
        )
        print("✓ Investment agent config creation successful")

        # Test config values
        assert investment_config.name == "TestAgent"
        assert investment_config.risk_tolerance == "moderate"
        assert "stocks" in investment_config.investment_focus
        print("✓ Configuration values correct")

        print("\n✅ Configuration tests passed!")
        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_agent_creation():
    """Test agent creation without actually connecting."""

    print("\nTesting agent creation...")

    try:
        from egile_investor.agent import InvestmentAgent
        from egile_investor.config import InvestmentAgentConfig, AzureOpenAIConfig

        # Create a test configuration
        azure_config = AzureOpenAIConfig(
            endpoint="https://test.openai.azure.com/",
            api_version="2024-12-01-preview",
            default_model="gpt-4",
        )

        config = InvestmentAgentConfig(
            name="TestAgent",
            investment_focus=["stocks"],
            risk_tolerance="moderate",
            openai_config=azure_config,
        )

        # Create agent (without connecting)
        agent = InvestmentAgent(config=config)
        print("✓ Investment agent creation successful")

        # Check agent properties
        assert agent.config.name == "TestAgent"
        assert agent.config.risk_tolerance == "moderate"
        print("✓ Agent properties correct")

        print("\n✅ Agent creation tests passed!")
        return True

    except Exception as e:
        print(f"❌ Agent creation test failed: {e}")
        return False


def main():
    """Run all tests."""

    print("=== Egile Investor Package Tests ===\n")

    tests = [
        test_imports,
        test_configuration,
        test_agent_creation,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Package structure is correct.")
        return 0
    else:
        print("❌ Some tests failed. Check the package structure.")
        return 1


if __name__ == "__main__":
    exit(main())
