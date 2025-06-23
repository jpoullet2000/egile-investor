# Enhanced Stock Screening with YAML-based Universe Support

## Overview

The `screen_stocks` tool has been enhanced to support comprehensive stock universe screening using configurable YAML files. This allows you to screen stocks from predefined universes like S&P 500, NASDAQ 100, or Dow 30, in addition to custom lists.

## New Features

### 1. YAML-based Stock Universes

Stock symbols are now maintained in `data/sp500.yml`, making them easy to update and maintain:

```yaml
sp500_symbols:
  - AAPL    # Apple Inc.
  - MSFT    # Microsoft Corporation
  - GOOGL   # Alphabet Inc. Class A
  # ... 500+ symbols

nasdaq100_symbols:
  - AAPL
  - MSFT
  - AMZN
  # ... 100 symbols

dow30_symbols:
  - AAPL
  - MSFT
  - UNH
  # ... 30 symbols
```

### 2. Enhanced Universe Parameter

The `universe` parameter now supports:

- **Predefined universes**: `"sp500"`, `"nasdaq100"`, `"dow30"`, `"major"`
- **Custom symbol lists**: `["AAPL", "MSFT", "GOOGL"]`
- **Default behavior**: Major stocks if no universe specified

### 3. S&P 500 Toggle

The `use_sp500` parameter provides easy access to full S&P 500 screening:

```python
# Screen all S&P 500 stocks
results = await agent.call_tool("screen_stocks", {
    "criteria": {"pe_ratio": {"max": 25}},
    "use_sp500": True,
    "max_results": 20
})
```

## Usage Examples

### 1. Screen S&P 500 Stocks

```python
from egile_investor.ai_agent import AIInvestmentAgent

async with AIInvestmentAgent() as agent:
    # Method 1: Using use_sp500 parameter
    results = await agent.call_tool("screen_stocks", {
        "criteria": {
            "pe_ratio": {"max": 18},
            "dividend_yield": {"min": 0.02},
            "roe": {"min": 0.12}
        },
        "use_sp500": True,
        "max_results": 10
    })
    
    # Method 2: Using universe parameter
    results = await agent.call_tool("screen_stocks", {
        "criteria": {"pe_ratio": {"max": 20}},
        "universe": "sp500",
        "max_results": 15
    })
```

### 2. Screen NASDAQ 100 or Dow 30

```python
# NASDAQ 100 screening
nasdaq_results = await agent.call_tool("screen_stocks", {
    "criteria": {"market_cap": {"min": 1000000000}},
    "universe": "nasdaq100",
    "max_results": 10
})

# Dow 30 screening
dow_results = await agent.call_tool("screen_stocks", {
    "criteria": {"dividend_yield": {"min": 0.03}},
    "universe": "dow30",
    "max_results": 5
})
```

### 3. AI-Powered Analysis with Different Universes

```python
from egile_investor import ai_investment_analysis

# Screen S&P 500 for value stocks
result = await ai_investment_analysis(
    "Find S&P 500 value stocks with PE ratio under 15 and dividend yield above 3%"
)

# Screen NASDAQ 100 for growth stocks
result = await ai_investment_analysis(
    "Find NASDAQ 100 growth stocks with strong fundamentals"
)
```

## Screening Criteria Examples

The enhanced screening supports various financial criteria:

```python
criteria = {
    # Valuation metrics
    "pe_ratio": {"max": 25, "min": 5},
    "pb_ratio": {"max": 3},
    "peg_ratio": {"max": 1.5},
    
    # Profitability metrics
    "roe": {"min": 0.15},               # ROE >= 15%
    "roa": {"min": 0.05},               # ROA >= 5%
    "gross_margin": {"min": 0.3},       # Gross margin >= 30%
    
    # Financial health
    "debt_to_equity": {"max": 0.5},     # D/E <= 0.5
    "current_ratio": {"min": 1.2},      # Current ratio >= 1.2
    "quick_ratio": {"min": 1.0},        # Quick ratio >= 1.0
    
    # Market metrics
    "market_cap": {"min": 1000000000},  # Market cap >= $1B
    "dividend_yield": {"min": 0.02},    # Dividend yield >= 2%
    "volume": {"min": 1000000},         # Average volume >= 1M
    
    # Growth metrics
    "revenue_growth": {"min": 0.05},    # Revenue growth >= 5%
    "earnings_growth": {"min": 0.1},    # Earnings growth >= 10%
}
```

## Financial Metrics Explained

Understanding the screening criteria is crucial for effective stock analysis. Here's what each metric means:

### Valuation Metrics
- **PE Ratio (Price-to-Earnings)**: Stock price divided by earnings per share. Lower values may indicate undervalued stocks
- **PB Ratio (Price-to-Book)**: Stock price divided by book value per share. Measures market premium over company's net worth
- **PEG Ratio (Price/Earnings to Growth)**: PE ratio divided by earnings growth rate. Values under 1.0 may indicate good value

### Profitability Metrics
- **ROE (Return on Equity)**: Net income divided by shareholder equity. Measures how efficiently a company uses shareholders' money
- **ROA (Return on Assets)**: Net income divided by total assets. Shows how efficiently a company uses its assets to generate profit
- **Gross Margin**: Gross profit divided by revenue. Indicates pricing power and cost control efficiency

### Financial Health Metrics
- **Debt-to-Equity**: Total debt divided by shareholders' equity. Lower ratios indicate less financial risk
- **Current Ratio**: Current assets divided by current liabilities. Values above 1.0 indicate ability to pay short-term debts
- **Quick Ratio**: (Current assets - inventory) divided by current liabilities. More conservative liquidity measure

### Market & Growth Metrics
- **Market Cap**: Total value of all company shares. Indicates company size and stability
- **Dividend Yield**: Annual dividends per share divided by stock price. Shows income return on investment
- **Revenue Growth**: Year-over-year increase in company revenue. Indicates business expansion
- **Earnings Growth**: Year-over-year increase in earnings per share. Shows profitability improvement

### Screening Strategy Tips
- **Value investing**: Focus on low PE, PB ratios with solid fundamentals (ROE, debt levels)
- **Growth investing**: Emphasize revenue growth, earnings growth, and reasonable PEG ratios  
- **Income investing**: Prioritize dividend yield, current ratio, and stable earnings
- **Quality investing**: Balance profitability (ROE, ROA) with financial stability (debt ratios)

## Performance Considerations

- **S&P 500 screening**: Processes up to 500+ stocks (may take 2-5 minutes)
- **NASDAQ 100 screening**: Processes ~100 stocks (faster)
- **Dow 30 screening**: Processes 30 stocks (fastest)
- **Major stocks**: Processes ~20 curated stocks (fastest)

The system automatically limits processing to avoid timeouts while still providing comprehensive results.

## File Structure

```
egile-investor/
├── data/
│   └── sp500.yml           # Stock universe definitions
├── egile_investor/
│   ├── agent.py            # Enhanced screening logic
│   └── server.py           # MCP tool definitions
└── examples/
    └── sp500_screening_example.py  # Usage examples
```

## Configuration

The YAML file (`data/sp500.yml`) can be updated to:

1. **Add new symbols**: Simply add to the appropriate list
2. **Create custom universes**: Add new symbol lists with descriptive names
3. **Update existing lists**: Modify symbols as companies are added/removed from indices

## Fallback Mechanisms

The system includes multiple fallback layers:

1. **Primary**: Load from YAML file
2. **Secondary**: Fetch from Wikipedia (for S&P 500)
3. **Tertiary**: Use hardcoded fallback list of major stocks

This ensures the system continues to work even if external data sources are unavailable.

## Benefits

- **Comprehensive coverage**: Screen all major US stock indices
- **Easy maintenance**: Update symbols via YAML configuration
- **Flexible usage**: Support for both predefined and custom universes
- **AI integration**: Works seamlessly with AI-powered investment analysis
- **Performance optimized**: Configurable limits to balance speed vs. coverage

This enhancement makes the Egile Investor package much more powerful for systematic stock screening and analysis across different market segments.
