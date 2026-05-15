# Tech Sector Portfolio Analysis

End-to-end quantitative analysis of **35 stocks** across **AI, Semiconductor, and New Energy** sectors — from raw data through EDA, volatility, correlation, lead-lag, risk assessment, mean-variance portfolio optimization, and backtesting.

> Data period: May 2025 – May 2026 (251 trading days)

For the full analysis with charts, data tables, and detailed findings, see **[report.ipynb](report.ipynb)**.

---

## Stock Universe

| Sector | Count | Tickers |
|--------|:-----:|---------|
| AI | 12 | NVDA, MSFT, GOOGL, META, AMZN, CRM, PLTR, AI, SNOW, IBM, SMCI, ORCL |
| Chip | 10 | AMD, INTC, TSM, AVGO, QCOM, TXN, MU, MRVL, ARM, ASML |
| New Energy | 13 | TSLA, ENPH, SEDG, FSLR, RUN, PLUG, BE, NEE, RIVN, LCID, NIO, LI, XPEV |
| Index (benchmark) | 6 | S&P 500, NASDAQ, Dow Jones, Russell 2000, VIX, PHLX Semiconductor |

---

## Analysis Pipeline

```
Data ── Part 1: EDA ── Part 2: Volatility ── Part 3: Correlation ── Part 4: Lead-Lag
                                                                       │
                        Part 6: Portfolio ◄── Part 5: Risk ◄───────────┘
```

---

## Scripts

| Script | Steps | What it does |
|--------|:-----:|--------------|
| `download_data.py` | 1 | Downloads 1-year daily OHLCV data from Yahoo Finance for all 41 tickers |
| `EDA.py` | 1 | Runs 9 data quality checks (missing values, duplicates, outliers, etc.) |
| `candlestick.py` | 2 | Generates line charts, sector cumulative return comparison, and performance ranking |
| `returns.py` | 3 | Computes daily/monthly/yearly return distributions and monthly return heatmap |
| `volatility.py` | 4–5 | Calculates annualized volatility, 30-day rolling volatility, and detects volatility spike events |
| `correlation.py` | 6–7 | Builds cross-sector correlation heatmaps, NVDA vs AMD deep dive, and Beta analysis vs S&P 500 |
| `leadlag.py` | 8–9 | Computes lagged cross-correlations and leadership scores to identify market leaders vs followers |
| `risk.py` | 10–11 | Drawdown analysis (underwater charts, recovery time) and risk metrics (VaR, Sharpe, Sortino, Calmar, Treynor) |
| `portfolio.py` | 12–13 | Mean-variance optimization (efficient frontier, Max Sharpe, Min Variance) and portfolio backtesting |

---

## Getting Started

**Prerequisites:** Python 3.10+

```bash
pip install -r requirements.txt
```

Run scripts in order:

```bash
python download_data.py
python EDA.py
python candlestick.py
python returns.py
python volatility.py
python correlation.py
python leadlag.py
python risk.py
python portfolio.py
```

All charts and CSV outputs are saved to the `charts/` directory.

---

## Tech Stack

| Library | Role |
|---------|------|
| pandas / numpy | Data processing and computation |
| matplotlib / seaborn | Visualization |
| scipy | Portfolio optimization (efficient frontier) |
| yfinance | Market data download |

## License

This project is for educational and research purposes.
