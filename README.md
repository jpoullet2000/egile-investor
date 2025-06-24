# Egile Investor

An AI-powered investment analysis and stock research automation package built on Model Context Protocol (MCP) servers.

## Overview

Egile Investor provides intelligent investment research capabilities through an AI agent that can analyze stocks, market trends, financial data, and provide investment recommendations. It uses MCP (Model Context Protocol) servers to expose investment tools and Azure OpenAI for intelligent decision-making.

## Features

- **Intelligent Stock Analysis**: AI-powered analysis of individual stocks with technical and fundamental insights
- **Market Trend Analysis**: Automated analysis of market trends and sector performance
- **Portfolio Optimization**: AI-driven portfolio analysis and optimization recommendations
- **Financial Data Integration**: Integration with multiple financial data sources (Yahoo Finance, Alpha Vantage, Finnhub, etc.)
- **Risk Assessment**: Automated risk analysis and portfolio risk metrics
- **Investment Screening**: AI-powered stock screening based on various criteria
- **Technical Analysis**: Automated technical indicator analysis and pattern recognition
- **Fundamental Analysis**: AI-driven fundamental analysis using financial statements and ratios
- **Market News Analysis**: Sentiment analysis of market news and its impact on investments
- **Automated Reporting**: Generate comprehensive investment reports and summaries
- **Generic MCP Agent**: Universal AI agent that works with any MCP server (like GitHub Copilot)
- **File Report Generation**: Automatically saves reports to files with custom or timestamp-based naming

## Architecture

The package is built on the Model Context Protocol (MCP) architecture:

- **AI Investment Agent**: Core AI agent that plans and executes investment analysis tasks
- **Generic MCP Agent**: Universal AI agent that works with any MCP server using LLM reasoning
- **MCP Server**: Exposes investment tools as MCP tools for integration with other systems
- **Investment Tools**: Specialized tools for different types of investment analysis
- **Data Sources**: Integration with multiple financial data providers
- **Azure OpenAI Integration**: Uses Azure OpenAI for intelligent analysis and decision-making

## Installation

```bash
# Install with pip
pip install egile-investor

# Or install from source
git clone <repository-url>
cd egile-investor
pip install -e .
```

## Configuration

Create a `.env` file with your configuration:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=your-azure-openai-endpoint
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEFAULT_MODEL=gpt-4
AZURE_KEY_VAULT_URL=your-key-vault-url
AZURE_OPENAI_API_KEY_SECRET_NAME=your-api-key-secret-name
AZURE_USE_MANAGED_IDENTITY=true

# Financial Data API Keys
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key
FINNHUB_API_KEY=your-finnhub-key
QUANDL_API_KEY=your-quandl-key
```

## Quick Start

### Basic Investment Analysis

```python
import asyncio
from egile_investor import ai_investment_analysis

async def main():
    # Analyze a specific stock
    result = await ai_investment_analysis("Analyze AAPL stock for investment potential")
    print(result)

asyncio.run(main())
```

### Using the Investment Agent

```python
from egile_investor import AIInvestmentAgent

async def main():
    async with AIInvestmentAgent() as agent:
        # Get stock analysis
        analysis = await agent.analyze("Analyze TSLA stock performance and provide buy/sell recommendation")
        
        # Analyze portfolio
        portfolio_analysis = await agent.analyze("Analyze my portfolio risk and suggest optimizations")
        
        print(analysis)
        print(portfolio_analysis)

asyncio.run(main())
```

### Advanced Stock Screening with S&P 500 Support

```python
from egile_investor import AIInvestmentAgent

async def main():
    async with AIInvestmentAgent() as agent:
        # Screen S&P 500 stocks for value opportunities
        sp500_results = await agent.call_tool("screen_stocks", {
            "criteria": {
                "pe_ratio": {"max": 18},        # P/E ratio under 18
                "dividend_yield": {"min": 0.02}, # Dividend yield above 2%
                "roe": {"min": 0.12}            # ROE above 12%
            },
            "use_sp500": True,  # Screen all S&P 500 stocks
            "max_results": 10
        })
        
        # Alternative: use universe parameter
        results = await agent.call_tool("screen_stocks", {
            "criteria": {"pe_ratio": {"max": 20}},
            "universe": "sp500",  # Options: "sp500", "major", or custom list
            "max_results": 5
        })
        
        # Screen only major stocks (default behavior)
        major_results = await agent.call_tool("screen_stocks", {
            "criteria": {"market_cap": {"min": 100000000000}},
            "universe": "major",
            "max_results": 10
        })

asyncio.run(main())
```

### Generic MCP Agent (Works with Any MCP Server)

The Generic MCP Agent works like GitHub Copilot - it can automatically discover and work with any MCP server:

```python
from egile_investor import GenericMCPAgent, generic_mcp_request

# Simple usage with any MCP server
async def main():
    # Works with investment server
    result = await generic_mcp_request(
        "Find value stocks with PE ratio under 15 and dividend yield above 3%",
        server_command="python -m egile_investor.server"
    )
    
    # Works with any other MCP server
    result = await generic_mcp_request(
        "List all Python files in the project",
        server_command="python -m file_server"  # hypothetical file server
    )
    
    # Manual control for advanced usage
    async with GenericMCPAgent(server_command="python -m any_server") as agent:
        # Agent automatically discovers available tools
        print(f"Found {len(agent.available_tools)} tools")
        
        # Uses LLM reasoning to create execution plans
        result = await agent.execute_request("Do something intelligent with available tools")
        
        print(f"Execution plan: {len(result['execution_plan'])} steps")
        print(f"Success rate: {result['success_rate']}")

asyncio.run(main())
```

### Report Generation with File Output

The `generate_report` tool now automatically saves reports to files:

```python
from egile_investor import AIInvestmentAgent

async def main():
    async with AIInvestmentAgent() as agent:
        # Generate report and save to custom file
        report = await agent.call_tool("generate_report", {
            "topic": "AAPL Stock Analysis",
            "content": {"analysis": "Detailed AAPL analysis..."},
            "filename": "aapl_analysis.md"  # Custom filename
        })
        
        # Generate report with automatic timestamp filename
        report = await agent.call_tool("generate_report", {
            "topic": "Market Overview",
            "content": {"market_data": "Current market conditions..."}
            # Will create: report_20241224-1430.md
        })

asyncio.run(main())
```

### MCP Server Usage

Start the MCP server:

```bash
egile-investor-server
```

Or programmatically:

```python
from egile_investor.server import main

if __name__ == "__main__":
    main()
```

## Available Tools

### Investment-Specific Tools

The MCP server exposes the following investment tools:

- `analyze_stock`: Comprehensive stock analysis with technical and fundamental insights
- `get_market_data`: Retrieve real-time and historical market data
- `screen_stocks`: Screen stocks based on various financial criteria (supports full S&P 500 screening)
- `analyze_portfolio`: Analyze portfolio performance, risk, and optimization
- `get_financial_ratios`: Calculate and analyze financial ratios
- `technical_analysis`: Perform technical analysis with indicators and patterns
- `sentiment_analysis`: Analyze market sentiment from news and social media
- `risk_assessment`: Assess investment risk and calculate risk metrics
- `generate_report`: Generate comprehensive investment analysis reports (now with file output)

### Generic MCP Tools

- `GenericMCPAgent`: Universal AI agent that works with any MCP server
- `generic_mcp_request`: Simple function to execute requests against any MCP server

**Key Features of Generic MCP Agent:**
- **Auto-Discovery**: Automatically discovers available tools from any MCP server
- **LLM Planning**: Uses Azure OpenAI to create intelligent execution plans
- **Error Resilience**: Continues execution even when individual steps fail
- **Type Safety**: Handles MCP validation requirements automatically
- **Server Agnostic**: Works with any MCP server without modification

## Examples

See the `examples/` directory for detailed usage examples:

- `basic_stock_analysis.py`: Basic stock analysis workflow
- `portfolio_optimization.py`: Portfolio analysis and optimization
- `market_screening.py`: Stock screening based on criteria
- `ai_investment_simple.py`: Simple AI-powered investment analysis

### Testing Different Agent Approaches

The package includes several agent implementations for comparison:

- `test_generic_agent.py`: Test the universal Generic MCP Agent
- `test_ultra_simple.py`: Test the ultra-simple investment-specific agent
- `test_validation_fix.py`: Test validation error handling
- `AGENT_COMPARISON.md`: Detailed comparison of different agent approaches

**Agent Comparison:**
- **Generic MCP Agent** ⭐: Works with any MCP server, like GitHub Copilot
- **Ultra-Simple Agent**: Investment-specific, no validation errors
- **Original AI Agent**: Complex validation logic, over-engineered (deprecated)

## Investment Workflows

### Stock Analysis Workflow
1. Fetch stock data from multiple sources
2. Perform technical analysis
3. Analyze fundamental metrics
4. Assess market sentiment
5. Generate investment recommendation

### Portfolio Analysis Workflow
1. Analyze current portfolio composition
2. Calculate risk metrics
3. Identify optimization opportunities
4. Suggest rebalancing strategies
5. Generate performance report

### Market Screening Workflow
1. Define screening criteria
2. Fetch market data for universe of stocks
3. Apply filters and scoring
4. Rank investment opportunities
5. Generate screening report

## Generic MCP Agent Advantages

The Generic MCP Agent provides several key advantages over traditional hardcoded approaches:

### 🔄 **Universal Compatibility**
- Works with **any** MCP server without modification
- Automatically discovers available tools and their schemas
- Adapts to different tool sets dynamically

### 🧠 **Intelligent Planning**
- Uses LLM reasoning to understand tool capabilities
- Creates optimal execution plans based on available tools
- Handles complex multi-step workflows automatically

### 🛡️ **Robust Error Handling**
- Graceful handling of validation errors
- Continues execution even when individual steps fail
- Type-safe argument resolution

### 📈 **Scalable Architecture**
- No hardcoded tool dependencies
- Easy to extend with new MCP servers
- Clean separation of concerns

### 🎯 **GitHub Copilot-like Experience**
- Similar to how Copilot works with MCP servers
- Natural language requests get converted to tool executions
- Intelligent chaining of tool calls

```python
# Example: Generic agent adapts to any MCP server
async def work_with_any_server(server_command, user_request):
    result = await generic_mcp_request(user_request, server_command)
    return result

# Investment server
result = await work_with_any_server("python -m egile_investor.server", "Find tech stocks")

# File server (hypothetical)
result = await work_with_any_server("python -m file_server", "List Python files")

# Database server (hypothetical)  
result = await work_with_any_server("python -m db_server", "Query user data")
```

## Configuration Options

The package supports extensive configuration through `InvestmentAgentConfig`:

```python
from egile_investor.config import InvestmentAgentConfig, AzureOpenAIConfig

config = InvestmentAgentConfig(
    name="MyInvestmentAgent",
    investment_focus=["large_cap", "technology", "growth"],
    risk_tolerance="moderate",
    max_positions=20,
    default_analysis_period=252,  # trading days
    openai_config=AzureOpenAIConfig.from_environment()
)
```

## Data Sources

- **Yahoo Finance**: Real-time and historical stock data
- **Alpha Vantage**: Financial data and indicators
- **Finnhub**: Market data and financial statements
- **Quandl**: Economic and financial data
- **pandas-datareader**: Integration with various data sources

## Dependencies

- Azure OpenAI for intelligent analysis
- FastMCP for MCP server functionality
- Financial data libraries (yfinance, alpha-vantage, etc.)
- Data analysis libraries (pandas, numpy, scipy)
- Technical analysis library (ta)
- Visualization libraries (matplotlib, seaborn, plotly)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests.

## Support

For support and questions, please open an issue on the GitHub repository.
Investor agent
