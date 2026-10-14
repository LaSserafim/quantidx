# QUANT.IDX — Institutional Equity & Bandarmology Terminal

Institutional-grade quantitative analytics and Bandarmology terminal for the Indonesia Stock Exchange (IDX / Bursa Efek Indonesia), built with **FastAPI**, **TradingView Pro**, **DeepSeek AI**, and high-density **FutureAs brutalism**.

![Architecture](https://img.shields.io/badge/Architecture-FastAPI%20%2B%20TradingView%20%2B%20DeepSeek-00f59b?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)
![Market](https://img.shields.io/badge/Market-IDX%20%2F%20BEI-00d2ff?style=flat-square)

---

## ⚡ Features

1. **FutureAs Brutalist Design System**:
   - Zero border-radius (`border-radius: 0 !important`), pitch-black (`#000000`), stark line borders (`#1c1c1c`), and high-contrast dual theme (Black + Neon Green / Black + Neon Red).
   - Google Fonts `Barlow Condensed` for display typography and `IBM Plex Mono` for tabular numerals.
   - Built-in scanning overlay with animated laser scan line.

2. **Real-Time Market Data Ingestion**:
   - Direct integration with **Yahoo Finance (`.JK`) `fast_info`** for zero-delay live quotes, lot volume, turnover (IDR), and market capitalization.
   - Embedded **TradingView Pro Interactive Candlestick Chart** tracking `IDX:{TICKER}` with technical indicators (1D, 1H, 30M, 1M).

3. **Bandarmology & Microstructure Engine**:
   - **Multi-Broker Cumulative Inventory Chart**: Track cumulative net lot accumulation/distribution curves across top brokers (`TP`, `CC`, `YU` vs `XL`, `ES`, `LG`) plotted against price.
   - **Broker Summary Matrix (BrokSum)**: Ranked buyer & seller ladders with lot sizes, IDR turnover, and Volume-Weighted Average Prices (`BAVG`/`SAVG`).
   - **Herfindahl-Hirschman Concentration Index (HHI)**: Measures institutional monopolization vs. retail churn.
   - **Retail Participation Ratio (RPR)**: Detects retail herd traps and distribution phases.

4. **Adaptive Orderbook & Live Microstructure**:
   - 15-level Depth of Market (DOM) with tick size scaling according to IDX rules.
   - Real-time **Order Flow Imbalance (OFI)** and Depth Pressure Ratio (`BID %` vs `OFFER %`).
   - Official IDX Market Session Governor: Automatically detects whether the exchange is in Regular Session I/II, Midday Break, or Market Closed, freezing the orderbook at the official closing bell when closed.
   - Fact-Only Toggle: Allows restricting view to **Level-1 Real Bid/Ask Only (100% Factual)** to prevent reliance on estimated small-cap queues.
   - Direct Stockbit L2 Token integration for verified 10-level exchange ITCH orderbooks.

5. **Historical Transaction Ledger & Capital Flow Chart**:
   - Page 02 dedicated view with 60-day transaction ledger.
   - Cumulative Capital Flow Chart comparing Smart Money / Bandar vs Foreign vs Retail over time with automated Bullish/Bearish divergence alerts.
   - Monthly Broker Calendar Heatmap matrix tracking daily broker lot activity.

6. **Corporate Governance, Detailed Shareholders & Real News**:
   - Full Board of Directors and Board of Commissioners roster.
   - Detailed Shareholder Structure (>5% and Public Float) with investor names, share counts, percentages, PSP status, and month-over-month changes.
   - Real-time financial media news and regulatory disclosures via Google News RSS for IDX (`CNBC Indonesia`, `Kontan`, `Bisnis.com`, `IDX Channel`).

7. **Quantitative Risk & Monte Carlo Simulation**:
   - Annualized Realized Volatility ($\sigma_{ann}$).
   - 1-Day Value-at-Risk ($VaR_{95\%}, VaR_{99\%}$) and Conditional VaR (Expected Shortfall).
   - Sharpe Ratio benchmarked against Bank Indonesia BI-Rate (6.0%).
   - 60-Day Historical Underwater Drawdown Curve (% peak-to-trough losses).
   - 1,000-simulation 30-Day Geometric Brownian Motion (GBM) Monte Carlo Price Cones (P90 Bullish, P50 Median, P10 Bearish).

8. **DeepSeek AI Institutional Analyst**:
   - Generates structured institutional market structure memorandums (Fase Akumulasi / Pre-Markup / Distribusi / Markdown), tactical execution levels (Support Entry, Resistance Target, Invalidation Stop), and risk warnings.
   - Enforced with strict anti-hallucination prompts.

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/LaSserafim/quantidx.git
cd quantidx
```

### 2. Configuration

Copy the example environment file:
```bash
cp .env.example .env
```
Edit `.env` and insert your DeepSeek API key (and optional Stockbit token):
```env
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
PORT=3000
```

### 3. Install Dependencies & Run

```bash
pip install -r requirements.txt
./run.sh
```
Open **`http://localhost:3000`** in your browser.

---

## 📁 Repository Structure

```
├── backend/
│   ├── main.py          # FastAPI server & REST API routes
│   ├── fetcher.py       # Live IDX data, yfinance fast_info, RSS news, governance
│   ├── quant_engine.py  # Monte Carlo, VaR, Sharpe, Drawdown, Bandarmology HHI
│   └── ai_analyst.py    # DeepSeek AI institutional analyst & quant fallback
├── frontend/
│   └── index.html       # FutureAs brutalist single-page terminal
├── .env.example         # Environment template
├── .gitignore           # Git ignore rules
├── requirements.txt     # Python dependencies
├── run.sh               # Single-command executable launcher
└── README.md            # Terminal documentation
```

---

## ⚖️ License

MIT License. Designed for quantitative researchers, algorithmic traders, and Indonesian capital market analysts.
