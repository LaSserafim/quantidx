"""
quant_engine.py — Quantitative Financial Econometrics, Microstructure & Risk Analytics.
Rigorous mathematical modeling for Indonesia Stock Exchange (IDX) time-series data.
Peer-reviewed econometric estimators: Amihud (2002), Corwin & Schultz (2012),
Roll (1984), Garman-Klass (1980), Parkinson (1980), Rogers-Satchell (1991),
Jarque-Bera (1987), Cornish-Fisher (1937), and calibrated Monte Carlo GBM.
"""
import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

# Bank Indonesia 7-Day Reverse Repo Rate (BI-Rate) benchmark risk-free rate (6.00% p.a.)
BI_RATE_RF = 0.06

def compute_quantitative_metrics(candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes rigorous empirical statistical moments, multi-estimator realized volatilities,
    tail risk metrics (VaR / CVaR / Cornish-Fisher), drawdown dynamics, and Monte Carlo GBM.
    """
    if len(candles) < 5:
        return {
            "sample_size": len(candles),
            "mean_daily_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "daily_volatility_pct": 0.0,
            "annualized_volatility_pct": 0.0,
            "parkinson_volatility_pct": 0.0,
            "garman_klass_volatility_pct": 0.0,
            "rogers_satchell_volatility_pct": 0.0,
            "skewness": 0.0,
            "excess_kurtosis": 0.0,
            "jarque_bera_stat": 0.0,
            "jarque_bera_pvalue": 1.0,
            "is_normal_distribution": True,
            "var_95_daily_pct": 0.0,
            "var_99_daily_pct": 0.0,
            "var_95_parametric_pct": 0.0,
            "var_95_cornish_fisher_pct": 0.0,
            "cvar_95_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "current_drawdown_pct": 0.0,
            "longest_drawdown_days": 0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "monte_carlo": {},
            "underwater_curve": [],
            "distribution_bins": []
        }

    df = pd.DataFrame(candles)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    prices = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    dates = df["time"].tolist() if "time" in df.columns else [f"D{i}" for i in range(len(prices))]

    # Prevent division by zero or negative prices
    valid_mask = (prices > 0) & (highs > 0) & (lows > 0) & (opens > 0) & (highs >= lows)
    if np.sum(valid_mask) < 5:
        prices = np.clip(prices, 1.0, None)
        highs = np.clip(highs, 1.0, None)
        lows = np.clip(lows, 1.0, None)
        opens = np.clip(opens, 1.0, None)

    # 1. Logarithmic Returns: r_t = ln(P_t / P_{t-1})
    log_returns = np.diff(np.log(prices))
    n_returns = len(log_returns)

    # 2. Distribution Moments
    mean_daily = float(np.mean(log_returns))
    ann_return = mean_daily * 252.0
    daily_vol = float(np.std(log_returns, ddof=1)) if n_returns > 1 else 0.01
    ann_vol_cc = daily_vol * np.sqrt(252.0)

    # Skewness: S = (1/N) * sum(((r - mu) / sigma)^3)
    if daily_vol > 1e-8:
        std_returns = (log_returns - mean_daily) / daily_vol
        skewness = float(np.mean(std_returns**3))
        # Excess Kurtosis: K = (1/N) * sum(((r - mu) / sigma)^4) - 3.0
        excess_kurtosis = float(np.mean(std_returns**4) - 3.0)
    else:
        skewness = 0.0
        excess_kurtosis = 0.0

    # Jarque-Bera Test of Normality: JB = (N/6) * (S^2 + (K^2 / 4))
    # Asymptotically distributed as Chi-Square with 2 degrees of freedom.
    # Survival function for Chi-Square(df=2) is exactly exp(-JB / 2).
    jb_stat = float((n_returns / 6.0) * (skewness**2 + ((excess_kurtosis**2) / 4.0)))
    jb_pvalue = float(math.exp(-max(0.0, jb_stat) / 2.0))
    is_normal = bool(jb_pvalue >= 0.05)

    # 3. Multi-Estimator Realized Volatility Comparison
    # A. Parkinson (1980) High-Low Extreme Value Volatility:
    # sigma_P = sqrt((1 / (4 * ln(2) * N)) * sum(ln(H_t / L_t)^2) * 252)
    hl_ratio = np.clip(highs / np.clip(lows, 1e-6, None), 1.0, None)
    parkinson_var = (1.0 / (4.0 * np.log(2.0) * len(hl_ratio))) * np.sum(np.log(hl_ratio)**2)
    parkinson_vol = float(np.sqrt(max(0.0, parkinson_var) * 252.0))

    # B. Garman-Klass (1980) Volatility (incorporates Open, High, Low, Close):
    # sigma_GK = sqrt((1/N) * sum(0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2) * 252)
    co_ratio = np.clip(prices / np.clip(opens, 1e-6, None), 1e-6, None)
    gk_terms = 0.5 * (np.log(hl_ratio)**2) - (2.0 * np.log(2.0) - 1.0) * (np.log(co_ratio)**2)
    gk_vol = float(np.sqrt(max(0.0, np.mean(gk_terms)) * 252.0))

    # C. Rogers-Satchell (1991) Volatility (drift-independent estimator):
    # sigma_RS = sqrt((1/N) * sum(ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)) * 252)
    hc = np.log(np.clip(highs / np.clip(prices, 1e-6, None), 1e-6, None))
    ho = np.log(np.clip(highs / np.clip(opens, 1e-6, None), 1e-6, None))
    lc = np.log(np.clip(lows / np.clip(prices, 1e-6, None), 1e-6, None))
    lo = np.log(np.clip(lows / np.clip(opens, 1e-6, None), 1e-6, None))
    rs_terms = (hc * ho) + (lc * lo)
    rs_vol = float(np.sqrt(max(0.0, np.mean(rs_terms)) * 252.0))

    # 4. Tail Risk: Value-at-Risk (VaR) & Expected Shortfall (CVaR)
    # A. Historical Simulation VaR (1-day)
    var_95_hist = float(-np.percentile(log_returns, 5))
    var_99_hist = float(-np.percentile(log_returns, 1))

    # B. Parametric Gaussian VaR: VaR_alpha = -(mu + z_alpha * sigma)
    z_95 = 1.6448536
    z_99 = 2.3263479
    var_95_param = float(-(mean_daily - (z_95 * daily_vol)))

    # C. Cornish-Fisher Modified VaR (accounts for non-zero skewness and excess kurtosis):
    # z_CF = z + (1/6)*(z^2 - 1)*S + (1/24)*(z^3 - 3z)*K - (1/36)*(2z^3 - 5z)*S^2
    z_cf_95 = (
        z_95
        + (1.0 / 6.0) * (z_95**2 - 1.0) * skewness
        + (1.0 / 24.0) * (z_95**3 - 3.0 * z_95) * excess_kurtosis
        - (1.0 / 36.0) * (2.0 * z_95**3 - 5.0 * z_95) * (skewness**2)
    )
    var_95_cf = float(-(mean_daily - (z_cf_95 * daily_vol)))

    # D. Expected Shortfall (Conditional VaR 95%): E[-r | -r >= VaR_95]
    tail_losses = log_returns[log_returns <= -var_95_hist]
    cvar_95 = float(-np.mean(tail_losses)) if len(tail_losses) > 0 else var_95_hist

    # 5. Drawdown Dynamics & Underwater Curve
    cum_max = np.maximum.accumulate(prices)
    drawdowns = (cum_max - prices) / cum_max
    max_dd = float(np.max(drawdowns))
    current_dd = float(drawdowns[-1])

    # Longest underwater duration (trading days)
    underwater_streaks = []
    current_streak = 0
    for dd in drawdowns:
        if dd > 0.001:
            current_streak += 1
        else:
            if current_streak > 0:
                underwater_streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        underwater_streaks.append(current_streak)
    longest_dd_days = max(underwater_streaks) if underwater_streaks else 0

    # 60-day historical underwater curve points
    underwater_curve = []
    for d_idx, dt in enumerate(dates):
        underwater_curve.append({
            "date": dt,
            "drawdown_pct": round(float(drawdowns[d_idx]) * 100.0, 2),
            "close": float(prices[d_idx])
        })

    # 6. Risk-Adjusted Ratios (Benchmarked against Bank Indonesia BI-Rate 6.00%)
    excess_return = ann_return - BI_RATE_RF
    sharpe = float(excess_return / ann_vol_cc) if ann_vol_cc > 1e-8 else 0.0

    # Sortino Ratio: Excess return / Downside Semi-Deviation
    daily_rf = BI_RATE_RF / 252.0
    downside_diffs = np.minimum(0.0, log_returns - daily_rf)
    downside_dev = float(np.sqrt(np.mean(downside_diffs**2) * 252.0))
    sortino = float(excess_return / downside_dev) if downside_dev > 1e-8 else 0.0

    # Calmar Ratio: Annualized Return / Maximum Drawdown
    calmar = float(ann_return / max_dd) if max_dd > 1e-4 else 0.0

    # 7. Monte Carlo 30-Day Forward Geometric Brownian Motion (GBM)
    current_price = float(prices[-1])
    n_days = 30
    n_sims = 1000
    dt = 1.0 / 252.0

    # Calibration on empirical annualized drift and volatility
    mu_annual = ann_return
    sigma_annual = ann_vol_cc

    drift = (mu_annual - 0.5 * sigma_annual**2) * dt
    shock = sigma_annual * np.sqrt(dt)

    # Use reproducible pseudo-random sequence for consistent analytical verification
    rng = np.random.RandomState(42)
    daily_innovations = rng.normal(drift, shock, size=(n_sims, n_days))
    price_paths = current_price * np.exp(np.cumsum(daily_innovations, axis=1))

    # Path percentiles across days
    p10_path = [round(float(v), 2) for v in np.percentile(price_paths, 10, axis=0)]
    p50_path = [round(float(v), 2) for v in np.percentile(price_paths, 50, axis=0)]
    p90_path = [round(float(v), 2) for v in np.percentile(price_paths, 90, axis=0)]

    final_prices = price_paths[:, -1]
    p_profit = float(np.mean(final_prices > current_price)) * 100.0
    mean_sim_ret = float(np.mean(final_prices / current_price - 1.0)) * 100.0

    monte_carlo = {
        "n_simulations": n_sims,
        "horizon_days": n_days,
        "current_price": current_price,
        "expected_median_price": p50_path[-1],
        "expected_bull_price": p90_path[-1],
        "expected_bear_price": p10_path[-1],
        "p_profit_pct": round(p_profit, 1),
        "mean_simulated_return_pct": round(mean_sim_ret, 2),
        "p10_path": p10_path,
        "p50_path": p50_path,
        "p90_path": p90_path
    }

    # 8. Empirical Return Distribution Bins (for Histogram vs Gaussian Curve)
    counts, bin_edges = np.histogram(log_returns * 100.0, bins=14)
    dist_bins = []
    for b_idx in range(len(counts)):
        b_mid = float((bin_edges[b_idx] + bin_edges[b_idx + 1]) / 2.0)
        # Theoretical normal PDF value at bin midpoint
        z_score = (b_mid / 100.0 - mean_daily) / (daily_vol if daily_vol > 1e-8 else 0.01)
        norm_pdf = (1.0 / (np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z_score**2)
        # Scaled to histogram frequency
        bin_width = float(bin_edges[b_idx + 1] - bin_edges[b_idx])
        norm_expected_count = norm_pdf * (bin_width / 100.0 / daily_vol) * n_returns
        dist_bins.append({
            "bin_min": round(float(bin_edges[b_idx]), 2),
            "bin_max": round(float(bin_edges[b_idx + 1]), 2),
            "midpoint": round(b_mid, 2),
            "observed_count": int(counts[b_idx]),
            "gaussian_fit_count": round(float(norm_expected_count), 1)
        })

    return {
        "sample_size": n_returns,
        "mean_daily_return_pct": round(mean_daily * 100.0, 3),
        "annualized_return_pct": round(ann_return * 100.0, 2),
        "daily_volatility_pct": round(daily_vol * 100.0, 2),
        "annualized_volatility_pct": round(ann_vol_cc * 100.0, 2),
        "parkinson_volatility_pct": round(parkinson_vol * 100.0, 2),
        "garman_klass_volatility_pct": round(gk_vol * 100.0, 2),
        "rogers_satchell_volatility_pct": round(rs_vol * 100.0, 2),
        "skewness": round(skewness, 2),
        "excess_kurtosis": round(excess_kurtosis, 2),
        "jarque_bera_stat": round(jb_stat, 2),
        "jarque_bera_pvalue": float(f"{jb_pvalue:.4e}"),
        "is_normal_distribution": is_normal,
        "var_95_daily_pct": round(var_95_hist * 100.0, 2),
        "var_99_daily_pct": round(var_99_hist * 100.0, 2),
        "var_95_parametric_pct": round(var_95_param * 100.0, 2),
        "var_95_cornish_fisher_pct": round(var_95_cf * 100.0, 2),
        "cvar_95_pct": round(cvar_95 * 100.0, 2),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "current_drawdown_pct": round(current_dd * 100.0, 2),
        "longest_drawdown_days": int(longest_dd_days),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "monte_carlo": monte_carlo,
        "underwater_curve": underwater_curve,
        "distribution_bins": dist_bins
    }

def compute_microstructure_metrics(candles: List[Dict[str, Any]], current_price: float, float_shares: Optional[int] = None) -> Dict[str, Any]:
    """
    Computes authentic empirical market microstructure and liquidity metrics from
    historical trade bars: Corwin-Schultz (2012) spread, Roll (1984) spread,
    Amihud (2002) illiquidity ratio, VWAP, and volume velocity.
    """
    if len(candles) < 3:
        return {
            "corwin_schultz_spread_pct": 0.0,
            "roll_spread_pct": 0.0,
            "amihud_illiq": 0.0,
            "turnover_velocity_pct": 0.0,
            "vwap": current_price,
            "vwap_deviation_pct": 0.0,
            "liquidity_classification": "INSUFFICIENT_DATA"
        }

    df = pd.DataFrame(candles)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    prices = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    vols = df["volume"].values  # In lots (1 lot = 100 shares)

    # 1. Corwin & Schultz (2012) High-Low Bid-Ask Spread Estimator (Journal of Finance)
    # Uses 2-day consecutive high-low price ratios to extract spread from variance
    try:
        beta = np.log(np.maximum(1e-6, highs[1:] / np.maximum(1e-6, lows[1:])))**2 + \
               np.log(np.maximum(1e-6, highs[:-1] / np.maximum(1e-6, lows[:-1])))**2
        h2 = np.maximum(highs[1:], highs[:-1])
        l2 = np.minimum(lows[1:], lows[:-1])
        gamma = np.log(np.maximum(1e-6, h2 / np.maximum(1e-6, l2)))**2
        
        c = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / c - np.sqrt(gamma / c)
        daily_spreads = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
        # Non-negative spread condition
        daily_spreads = np.maximum(0.0, daily_spreads)
        cs_spread = float(np.nanmean(daily_spreads)) * 100.0
    except Exception:
        cs_spread = 0.0

    # 2. Roll (1984) Effective Spread Model
    # S = 2 * sqrt(-Cov(Delta P_t, Delta P_{t-1})) / P_mean
    try:
        delta_p = np.diff(prices)
        if len(delta_p) > 2:
            cov_matrix = np.cov(delta_p[1:], delta_p[:-1])
            cov_p = float(cov_matrix[0, 1])
            if cov_p < 0:
                roll_spread = float(2.0 * np.sqrt(-cov_p) / np.mean(prices)) * 100.0
            else:
                roll_spread = 0.0
        else:
            roll_spread = 0.0
    except Exception:
        roll_spread = 0.0

    # 3. Amihud (2002) Illiquidity Ratio (Price Impact per IDR volume)
    # ILLIQ = (1/N) * sum(|R_t| / (P_t * Vol_shares_t)) * 1e9
    try:
        ret_series = np.abs(np.diff(prices) / np.maximum(1e-6, prices[:-1]))
        val_series = prices[1:] * (vols[1:] * 100.0)  # Value in IDR
        valid_val = val_series > 0
        if np.sum(valid_val) > 0:
            illiq = float(np.mean(ret_series[valid_val] / val_series[valid_val])) * 1e9
        else:
            illiq = 0.0
    except Exception:
        illiq = 0.0

    # 4. Volume-Weighted Average Price (VWAP)
    try:
        typical_prices = (highs + lows + prices) / 3.0
        total_vol = np.sum(vols)
        if total_vol > 0:
            vwap = float(np.sum(typical_prices * vols) / total_vol)
        else:
            vwap = float(prices[-1])
        vwap_dev = float(((current_price - vwap) / vwap) * 100.0) if vwap > 0 else 0.0
    except Exception:
        vwap = current_price
        vwap_dev = 0.0

    # 5. Turnover Velocity (% of float traded on latest day)
    try:
        latest_vol_shares = vols[-1] * 100.0
        if float_shares and float_shares > 0:
            velocity = float((latest_vol_shares / float_shares) * 100.0)
        else:
            velocity = 0.0
    except Exception:
        velocity = 0.0

    # Liquidity Classification based on Amihud illiquidity and spread
    if illiq < 0.0001 and cs_spread < 1.0:
        liq_class = "HIGH_LIQUIDITY (Blue-Chip Tier)"
    elif illiq < 0.005 and cs_spread < 3.0:
        liq_class = "MODERATE_LIQUIDITY (Mid-Cap Tier)"
    else:
        liq_class = "ILLIQUID_TAIL (High Price-Impact Tier)"

    return {
        "corwin_schultz_spread_pct": round(cs_spread, 2),
        "roll_spread_pct": round(roll_spread, 2),
        "amihud_illiq": round(illiq, 6),
        "turnover_velocity_pct": round(velocity, 3),
        "vwap": round(vwap, 2),
        "vwap_deviation_pct": round(vwap_dev, 2),
        "liquidity_classification": liq_class
    }
