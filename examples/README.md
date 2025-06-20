# Egile Investor Examples

This directory contains example scripts demonstrating how to use the Egile Investor package for various investment analysis tasks.

## Examples

### 1. Basic Stock Analysis (`basic_stock_analysis.py`)
Demonstrates basic stock analysis functionality:
- Getting stock data from multiple sources
- Performing technical and fundamental analysis
- Comparing multiple stocks
- Understanding analysis results

**Run with:**
```bash
python basic_stock_analysis.py
```

### 2. AI-Powered Investment Analysis (`ai_investment_simple.py`)
Shows how to use the AI Investment Agent:
- Simple AI-powered analysis queries
- Complex multi-step analysis planning
- Different investment scenarios
- Automated tool selection and execution

**Run with:**
```bash
python ai_investment_simple.py
```

### 3. Stock Screening (`market_screening.py`)
Demonstrates stock screening capabilities:
- Basic screening with custom criteria
- Value investing screens
- Growth investing screens
- Dividend stock screening

**Run with:**
```bash
python market_screening.py
```

### 4. Portfolio Optimization (`portfolio_optimization.py`)
Portfolio analysis and optimization examples:
- Portfolio performance analysis
- Risk assessment and metrics
- Optimization recommendations
- Rebalancing strategies

**Run with:**
```bash
python portfolio_optimization.py
```

## Prerequisites

Before running the examples, make sure you have:

1. **Installed the package:**
   ```bash
   pip install -e .
   ```

2. **Set up your environment variables** in a `.env` file:
   ```env
   # Azure OpenAI Configuration
   AZURE_OPENAI_ENDPOINT=your-endpoint
   AZURE_OPENAI_API_VERSION=2024-12-01-preview
   AZURE_OPENAI_DEFAULT_MODEL=gpt-4
   AZURE_KEY_VAULT_URL=your-key-vault-url
   AZURE_OPENAI_API_KEY_SECRET_NAME=your-secret-name
   
   # Optional: Financial Data API Keys
   ALPHA_VANTAGE_API_KEY=your-key
   FINNHUB_API_KEY=your-key
   QUANDL_API_KEY=your-key
   ```

3. **For MCP server examples**, start the server:
   ```bash
   egile-investor-server
   ```

## Example Outputs

### Stock Analysis Output
```
=== Basic Stock Analysis ===

Analyzing AAPL...
AAPL Current Price: $150.25
AAPL Recommendation: BUY
AAPL Confidence: High
AAPL Risk Level: Medium
AAPL RSI: 45.32
AAPL P/E Ratio: 28.5
```

### AI Analysis Output
```
=== Simple AI Investment Analysis ===

Task: Analyze AAPL stock and tell me if I should buy, hold, or sell

Plan created by AI:
  Step 1: Analyze stock for comprehensive investment analysis
  Step 2: Assess investment risk and market conditions

Execution Summary:
  Total steps: 2
  Successful: 2
  Failed: 0
  Tools used: analyze_stock, risk_assessment
```

### Screening Output
```
=== Value Investing Screen ===

Found 3 value opportunities:

AAPL:
  P/E: 15.2
  P/B: 1.8
  ROE: 0.25
  Debt/Equity: 0.3
  Score: 85.0
```

## Customization

You can customize the examples by:

1. **Modifying stock symbols** in the example scripts
2. **Adjusting analysis parameters** (time periods, criteria, etc.)
3. **Changing screening criteria** for different investment strategies
4. **Adding your own analysis logic** and metrics

## Error Handling

The examples include error handling for common scenarios:
- Invalid stock symbols
- API rate limits
- Network connectivity issues
- Missing configuration

## Performance Tips

- **Use caching** for repeated analysis of the same stocks
- **Batch requests** when analyzing multiple stocks
- **Set appropriate time periods** for your analysis needs
- **Monitor API usage** to avoid rate limits

## Support

If you encounter issues with the examples:
1. Check your configuration and API keys
2. Verify internet connectivity
3. Review the error messages for specific issues
4. Check the main README for troubleshooting tips
