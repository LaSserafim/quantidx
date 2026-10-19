# QUANT.IDX: Quantitative Equity Research & Market Microstructure Terminal for the Indonesia Stock Exchange

QUANT.IDX is an open-access quantitative finance terminal designed for Indonesian capital markets (IDX / Bursa Efek Indonesia). The platform integrates empirical financial econometrics, range-based volatility estimators, non-Gaussian tail risk modeling, liquidity proxies, and fundamental accounting ratios into a unified terminal interface.

The backend runs on Python (FastAPI, NumPy, Pandas) and ingests live market data for all `.JK` tickers. The frontend is built on a high-density brutalist user interface with Chart.js canvas rendering and TradingView Pro technical charting.

Live Web Deployment: [https://quantidx.vercel.app](https://quantidx.vercel.app)

---

## 1. System Architecture

The application separates empirical time-series computation from data ingestion and presentation:

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Browser                         │
│  Single-Page Brutalist Terminal (Barlow Condensed + Mono)   │
│  TradingView Pro Widget · Chart.js Engine · GSAP Animations │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / JSON
┌──────────────────────────────┴──────────────────────────────┐
│                    FastAPI Backend Server                   │
│                       (backend/main.py)                     │
├──────────────────────────────┬──────────────────────────────┤
│  Data Ingestion Engine       │  Quantitative Analytics Core │
│  (backend/fetcher.py)        │  (backend/quant_engine.py)   │
│  - yfinance (.JK fast_info)  │  - Distributional Moments    │
│  - Official IDX Rules        │  - Jarque-Bera Test          │
│  - Corporate Governance      │  - Multi-Volatility Models   │
│  - Audited Financials        │  - Cornish-Fisher VaR & CVaR │
│  - Google News RSS Feed      │  - Corwin-Schultz Spread     │
│                              │  - Amihud Illiquidity Ratio  │
│                              │  - Monte Carlo GBM 30D       │
├──────────────────────────────┴──────────────────────────────┤
│                DeepSeek AI Analyst Core                     │
│                (backend/ai_analyst.py)                      │
│  - Structured Econometric Research Memorandums              │
│  - Deterministic Quantitative Synthesizer Fallback          │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Econometric & Mathematical Methodology

All analytical metrics in QUANT.IDX rely on verified, peer-reviewed financial econometrics. The system does not use synthetic order books or simulated broker ladders.

### 2.1 Return Distribution Moments and Normality Testing

Logarithmic daily returns are computed as:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

From the log-return series, the engine derives the sample mean ($\mu$), standard deviation ($\sigma$), sample skewness ($S$), and excess kurtosis ($K$):

$$S = \frac{\frac{1}{N}\sum_{t=1}^N (r_t - \mu)^3}{\sigma^3}$$

$$K = \frac{\frac{1}{N}\sum_{t=1}^N (r_t - \mu)^4}{\sigma^4} - 3$$

To evaluate whether the empirical return series conforms to a normal distribution, the system computes the Jarque-Bera (1987) test statistic:

$$JB = \frac{N}{6}\left(S^2 + \frac{K^2}{4}\right)$$

Because $JB$ asymptotically follows a Chi-Square distribution with two degrees of freedom ($\chi^2_2$), the exact p-value is calculated in closed form:

$$p = \exp\left(-\frac{JB}{2}\right)$$

When $p < 0.05$, the Gaussian normality assumption is rejected, confirming that returns exhibit leptokurtosis (fat tails) and asymmetry common to emerging market equities.

### 2.2 Range-Based Realized Volatility Estimators

Standard close-to-close volatility discards intraday price extremes. QUANT.IDX computes three extreme-value volatility estimators alongside close-to-close volatility:

1. Parkinson (1980) High-Low Volatility:
   Measures price dispersion using daily high ($H_t$) and low ($L_t$) prices, providing approximately five times greater statistical efficiency than the close-to-close estimator:

   $$\sigma_{\text{Parkinson}} = \sqrt{\frac{1}{4 \ln(2) \cdot N} \sum_{t=1}^N \left(\ln\frac{H_t}{L_t}\right)^2 \times 252}$$

2. Garman-Klass (1980) Volatility:
   Incorporates opening ($O_t$), high ($H_t$), low ($L_t$), and closing ($C_t$) prices to capture both intraday range and opening jumps:

   $$\sigma_{\text{GK}} = \sqrt{\frac{1}{N} \sum_{t=1}^N \left[ 0.5 \left(\ln\frac{H_t}{L_t}\right)^2 - (2\ln 2 - 1) \left(\ln\frac{C_t}{O_t}\right)^2 \right] \times 252}$$

3. Rogers-Satchell (1991) Volatility:
   Allows for non-zero drift in stock prices without upward bias:

   $$\sigma_{\text{RS}} = \sqrt{\frac{1}{N} \sum_{t=1}^N \left[ \ln\left(\frac{H_t}{C_t}\right)\ln\left(\frac{H_t}{O_t}\right) + \ln\left(\frac{L_t}{C_t}\right)\ln\left(\frac{L_t}{O_t}\right) \right] \times 252}$$

### 2.3 Downside Risk and Value-at-Risk (VaR)

Standard parametric VaR assumes normal distribution and underestimates tail risk when returns are fat-tailed. The engine implements three complementary risk measures:

1. Historical Simulation VaR (95% and 99%):
   The empirical 5th and 1st percentiles of daily log returns:

   $$\text{VaR}_{95, \text{Hist}} = -\text{Percentile}(r, 5)$$

2. Cornish-Fisher (1937) Modified VaR:
   Adjusts standard Gaussian critical values ($z_\alpha$) for observed skewness ($S$) and excess kurtosis ($K$):

   $$z_{\text{CF}} = z_\alpha + \frac{1}{6}(z_\alpha^2 - 1)S + \frac{1}{24}(z_\alpha^3 - 3z_\alpha)K - \frac{1}{36}(2z_\alpha^3 - 5z_\alpha)S^2$$

   $$\text{VaR}_{95, \text{CF}} = -(\mu - z_{\text{CF}} \sigma)$$

3. Expected Shortfall / Conditional VaR (CVaR 95%):
   Measures average loss conditional on the loss exceeding the 95% VaR threshold:

   $$\text{CVaR}_{95} = E\left[-r_t \mid -r_t \ge \text{VaR}_{95}\right]$$

### 2.4 Risk-Adjusted Performance Ratios

Performance is benchmarked against the Bank Indonesia 7-Day Reverse Repo Rate (BI-Rate, 6.00% p.a.):

1. Sharpe Ratio:

   $$\text{Sharpe} = \frac{R_{\text{annual}} - R_f}{\sigma_{\text{annual}}}$$

2. Sortino Ratio:
   Penalizes only downside volatility below the daily risk-free threshold ($R_{f, \text{daily}} = 6.00\% / 252$):

   $$\text{Sortino} = \frac{R_{\text{annual}} - R_f}{\sqrt{\frac{1}{N}\sum_{t=1}^N \min(0, r_t - R_{f, \text{daily}})^2 \times 252}}$$

3. Calmar Ratio:

   $$\text{Calmar} = \frac{R_{\text{annual}}}{\text{Maximum Drawdown}}$$

### 2.5 Empirical Market Microstructure and Liquidity

Because free public APIs do not provide Level-2 order books for the Indonesia Stock Exchange, QUANT.IDX uses peer-reviewed econometric estimators to assess effective bid-ask spread and liquidity:

1. Corwin & Schultz (2012) High-Low Bid-Ask Spread Estimator:
   Published in the Journal of Finance, this model extracts bid-ask spread from the variance of two-day versus single-day high-to-low price ratios:

   $$\beta = \ln\left(\frac{H_t}{L_t}\right)^2 + \ln\left(\frac{H_{t+1}}{L_{t+1}}\right)^2$$

   $$\gamma = \left[\ln\left(\frac{\max(H_t, H_{t+1})}{\min(L_t, L_{t+1})}\right)\right]^2$$

   $$\alpha = \frac{\sqrt{2\beta} - \sqrt{\beta}}{3 - 2\sqrt{2}} - \sqrt{\frac{\gamma}{3 - 2\sqrt{2}}}$$

   $$\text{Spread}_{\text{CS}} = \frac{2(e^\alpha - 1)}{1 + e^\alpha}$$

2. Roll (1984) Effective Spread Model:
   Estimates effective spread from the first-order serial covariance of price changes:

   $$S_{\text{Roll}} = 2\sqrt{\max\left(0, -\text{Cov}(\Delta P_t, \Delta P_{t-1})\right)} / P_{\text{mean}}$$

3. Amihud (2002) Illiquidity Ratio ($ILLIQ$):
   Measures absolute return per unit of daily rupiah trading value (in billions of IDR), quantifying price impact:

   $$ILLIQ = \frac{1}{N}\sum_{t=1}^N \frac{|R_t|}{\text{Value IDR}_t} \times 10^9$$

4. Volume-Weighted Average Price (VWAP):

   $$\text{VWAP} = \frac{\sum_{t=1}^N \left(\frac{H_t + L_t + C_t}{3} \times V_t\right)}{\sum_{t=1}^N V_t}$$

### 2.6 Monte Carlo Geometric Brownian Motion (GBM) Simulation

The 30-day forward price distribution is modeled via 1,000 Geometric Brownian Motion simulation paths calibrated to historical drift and diffusion:

$$P_{t+\Delta t} = P_t \exp\left(\left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t} Z\right), \quad Z \sim \mathcal{N}(0, 1)$$

Percentile bands at P10 (bearish boundary), P50 (median projection), and P90 (bullish boundary) provide probabilistic targets, alongside empirical probability of profit ($P(P_{30} > P_0)$).

---

## 3. Official Exchange Rules & Trading Sessions

QUANT.IDX enforces official Indonesia Stock Exchange market rules:

1. Tick Size Schedule (Peraturan BEI No. II-A):
   - Price < Rp 200: Tick size Rp 1 (maximum 10 ticks per step)
   - Price Rp 200 to Rp 500: Tick size Rp 2 (maximum 20 ticks per step)
   - Price Rp 500 to Rp 2,000: Tick size Rp 5 (maximum 20 ticks per step)
   - Price Rp 2,000 to Rp 5,000: Tick size Rp 10 (maximum 40 ticks per step)
   - Price >= Rp 5,000: Tick size Rp 25 (maximum 50 ticks per step)

2. Asymmetric Auto-Rejection Limits (ARA / ARB):
   - Price Rp 50 to Rp 200: +35% ARA / -35% ARB
   - Price Rp 200 to Rp 5,000: +25% ARA / -25% ARB
   - Price > Rp 5,000: +20% ARA / -20% ARB
   Limits are calculated from the previous closing price and rounded to the valid tick boundary.

3. Official Trading Session Monitor (WIB / UTC+7):
   - Pre-Opening: 08:45 to 08:59 WIB
   - Session I: 09:00 to 12:00 WIB (Friday: 09:00 to 11:30 WIB)
   - Midday Break: 12:00 to 13:30 WIB (Friday: 11:30 to 14:00 WIB)
   - Session II: 13:30 to 15:50 WIB (Friday: 14:00 to 15:50 WIB)
   - Pre-Closing: 15:50 to 16:00 WIB
   - Post-Trading: 16:05 to 16:15 WIB
   - Market Closed: After 16:15 WIB and weekends

---

## 4. REST API Specification

The backend exposes the following endpoints:

| Endpoint | Method | Response Description |
|---|---|---|
| `/api/health` | GET | System status, version, and data integrity verification tag |
| `/api/stock/{ticker}` | GET | Spot quote, day high/low, lot volume, turnover IDR, market cap, tick size, ARA/ARB limits, and session status |
| `/api/microstructure/{ticker}` | GET | Corwin-Schultz spread, Roll spread, Amihud illiquidity ratio, VWAP, VWAP deviation, and float turnover velocity |
| `/api/quant/{ticker}` | GET | Return moments (mean, volatility, skewness, excess kurtosis), Jarque-Bera statistics, multi-volatility estimators, VaR, CVaR, drawdown series, Sharpe ratio, Monte Carlo paths, and distribution histogram bins |
| `/api/valuation/{ticker}` | GET | Audited financial statement metrics: P/E, P/B, EV/EBITDA, P/S, dividend yield, ROE, ROA, margins, debt-to-equity, current/quick ratios, BVPS, EPS |
| `/api/history/{ticker}` | GET | 60-day factual transaction ledger with daily returns, VWAP, Amihud impact, and rolling 10-day realized volatility series |
| `/api/profile/{ticker}` | GET | Company description, sector, Board of Directors, Board of Commissioners, and shareholder structure (>5% and float) |
| `/api/news/{ticker}` | GET | Filtered real-time Google News RSS articles for the emiten ticker with sentiment classification |
| `/api/analyze/{ticker}` | POST | DeepSeek AI quantitative equity memorandum; falls back to deterministic econometric synthesizer when no key is present |
| `/api/settings/key` | POST | Updates DeepSeek API key dynamically at runtime |

---

## 5. Local Setup and Deployment

### 5.1 Prerequisites

- Python 3.9 or higher
- Modern web browser with JavaScript enabled (Chrome, Firefox, Safari, Edge)

### 5.2 Installation

```bash
# Clone the repository
git clone https://github.com/LaSserafim/quantidx.git
cd quantidx

# Install dependencies
pip install -r requirements.txt
```

### 5.3 Configuration

Create an environment configuration file:

```bash
cp .env.example .env
```

Edit `.env` to configure your settings (optional):

```env
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
HOST=127.0.0.1
PORT=3000
```

If `DEEPSEEK_API_KEY` is omitted, the terminal automatically runs on its built-in deterministic econometric synthesizer, generating analytical memorandums from empirical calculations without external API calls.

### 5.4 Running the Terminal

```bash
./run.sh
```

Or start the server directly using Uvicorn:

```bash
python3 backend/main.py
```

Access the interface in your browser at `http://127.0.0.1:3000`.

---

## 6. Academic References

1. Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects. *Journal of Financial Markets*, 5(1), 31-56.
2. Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. *The Journal of Finance*, 67(2), 719-760.
3. Cornish, E. A., & Fisher, R. A. (1937). Moments and cumulants in the specification of distributions. *Revue de l'Institut International de Statistique*, 307-320.
4. Garman, M. B., & Klass, M. J. (1980). On the estimation of security price volatilities from historical data. *Journal of Business*, 67-78.
5. Jarque, C. M., & Bera, A. K. (1987). A test for normality of observations and regression residuals. *International Statistical Review*, 163-172.
6. Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 61-65.
7. Rogers, L. C., & Satchell, S. E. (1991). Estimating variance from high, low, open, and closing prices. *The Annals of Applied Probability*, 504-512.
8. Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market. *The Journal of Finance*, 39(4), 1127-1139.

---

## 7. License

MIT License. Developed for quantitative finance researchers, students, and capital market analysts.
