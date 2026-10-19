"""
ai_analyst.py — DeepSeek AI Quantitative Analyst & Econometric Research Memorandum.
Synthesizes verified econometric moments, extreme value volatilities, tail risk (VaR/CVaR),
empirical microstructure (Amihud & Corwin-Schultz), and valuation fundamentals.
Zero hallucinated broker codes, zero fabricated order books.
"""
import os
import json
import math
from typing import Dict, Any, Optional
from openai import OpenAI

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

def get_deepseek_client() -> OpenAI:
    key = os.environ.get("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
    return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)

def generate_fallback_memorandum(
    ticker: str,
    overview: Dict[str, Any],
    quant: Dict[str, Any],
    microstructure: Dict[str, Any],
    valuation: Dict[str, Any],
    reason: str
) -> Dict[str, Any]:
    """
    Deterministic quantitative research memorandum generated directly from verified figures.
    Guarantees mathematically consistent, robust output when no external LLM API key is present.
    """
    price = overview.get("price", 0.0)
    change_pct = overview.get("change_pct", 0.0)
    tick = overview.get("tick_size", 5)

    ann_vol = quant.get("annualized_volatility_pct", 0.0)
    gk_vol = quant.get("garman_klass_volatility_pct", 0.0)
    skew = quant.get("skewness", 0.0)
    kurt = quant.get("excess_kurtosis", 0.0)
    jb_stat = quant.get("jarque_bera_stat", 0.0)
    jb_pvalue = quant.get("jarque_bera_pvalue", 1.0)
    is_normal = quant.get("is_normal_distribution", True)

    var_95_cf = quant.get("var_95_cornish_fisher_pct", 0.0)
    cvar_95 = quant.get("cvar_95_pct", 0.0)
    mdd = quant.get("max_drawdown_pct", 0.0)
    sharpe = quant.get("sharpe_ratio", 0.0)

    mc = quant.get("monte_carlo", {})
    mc_median = mc.get("expected_median_price", price)
    mc_profit_prob = mc.get("p_profit_pct", 50.0)

    cs_spread = microstructure.get("corwin_schultz_spread_pct", 0.0)
    illiq = microstructure.get("amihud_illiq", 0.0)
    vwap = microstructure.get("vwap", price)
    vwap_dev = microstructure.get("vwap_deviation_pct", 0.0)

    pe = valuation.get("trailing_pe")
    pb = valuation.get("price_to_book")
    roe = valuation.get("return_on_equity_pct")
    de = valuation.get("debt_to_equity_pct")

    # 1. Determine Econometric Regime
    if kurt > 3.0 or jb_pvalue < 0.01:
        fat_tail_str = "Leptokurtic (Heavy Fat Tails)"
    elif kurt < -0.5:
        fat_tail_str = "Platykurtic (Thin Tails)"
    else:
        fat_tail_str = "Mesokurtic (Near-Gaussian)"

    if ann_vol > 60.0:
        vol_str = "High Volatility Regime"
    elif ann_vol > 30.0:
        vol_str = "Moderate Volatility Regime"
    else:
        vol_str = "Low Volatility Stable Regime"

    regime = f"{vol_str} // {fat_tail_str}"

    # 2. Key Statistical Execution Levels
    # Statistical Entry Support: bounded by VWAP and Cornish-Fisher VaR
    support = round(price * (1.0 - (max(2.0, var_95_cf) / 100.0)), 0)
    support = math.floor(support / tick) * tick

    # Volatility Target Resistance: bounded by 30-day Monte Carlo median
    resistance = round(max(price * 1.04, mc_median), 0)
    resistance = math.ceil(resistance / tick) * tick

    # Invalidation Stop: beyond CVaR 95% tail risk
    invalidation = round(price * (1.0 - (max(4.0, cvar_95 * 1.15) / 100.0)), 0)
    invalidation = math.floor(invalidation / tick) * tick

    # 3. Action Recommendation
    if mc_profit_prob >= 60.0 and sharpe > 1.0 and vwap_dev <= 3.0:
        action = "Scale-in Long (Positive Risk-Adjusted Expectancy)"
        conviction = "High"
    elif mc_profit_prob < 40.0 or sharpe < 0.0:
        action = "Defensive / Capital Preservation (Negative Risk-Adjusted Asymmetry)"
        conviction = "High"
    elif vwap_dev > 10.0:
        action = "Mean-Reversion Caution (Overextended Above VWAP)"
        conviction = "Moderate"
    else:
        action = "Neutral Accumulation on Pullbacks"
        conviction = "Moderate"

    # 4. Formulate Comprehensive Memorandum
    pe_str = f"{pe:.1f}x" if pe else "N/A"
    pb_str = f"{pb:.2f}x" if pb else "N/A"
    roe_str = f"{roe:.1f}%" if roe else "N/A"
    de_str = f"{de:.1f}%" if de else "N/A"

    memo_text = (
        f"1. REGIME & DISTRIBUTION MOMENTS: {ticker} exhibits an annualized realized volatility of {ann_vol:.1f}% "
        f"(Garman-Klass OHLC volatility: {gk_vol:.1f}%). The empirical return distribution displays a skewness of {skew:.2f} "
        f"and excess kurtosis of {kurt:.2f}. The Jarque-Bera test yields a test statistic of {jb_stat:.2f} (p-value: {jb_pvalue:.4e}), "
        f"{'rejecting the Gaussian normality hypothesis at the 1% significance level and confirming leptokurtic fat-tail risks' if not is_normal else 'failing to reject normality, indicating stable symmetric diffusion'}.\n\n"
        f"2. TAIL RISK & DOWNSIDE MODELING: The 1-day Cornish-Fisher modified Value-at-Risk (95% confidence) is {var_95_cf:.2f}%, "
        f"while Expected Shortfall (CVaR 95%) indicates an average tail loss expectation of {cvar_95:.2f}%. "
        f"Maximum historical drawdown over the 60-day window stands at {mdd:.1f}%. Risk-adjusted performance benchmarked against the "
        f"Bank Indonesia BI-Rate (6.00% p.a.) yields a Sharpe Ratio of {sharpe:.2f}.\n\n"
        f"3. MARKET MICROSTRUCTURE & LIQUIDITY: The Corwin-Schultz (2012) high-low effective bid-ask spread is estimated at "
        f"{cs_spread:.2f}%. Amihud (2002) illiquidity ratio measures {illiq:.6f} (scaled price impact per IDR billion turnover). "
        f"Current spot price Rp {price:,.0f} trades at a {vwap_dev:+.2f}% deviation from the 60-day Volume-Weighted Average Price (VWAP Rp {vwap:,.0f}).\n\n"
        f"4. FUNDAMENTAL QUALITY & VALUATION: Trading multiples indicate Trailing P/E of {pe_str}, Price/Book of {pb_str}, "
        f"Return on Equity (ROE) of {roe_str}, and Debt-to-Equity of {de_str}. "
        f"The 1,000-run Monte Carlo Geometric Brownian Motion forward simulation projects a 30-day median price of Rp {mc_median:,.0f} "
        f"with a {mc_profit_prob:.1f}% empirical probability of positive return."
    )

    return {
        "ticker": ticker,
        "econometric_regime": regime,
        "action_recommendation": action,
        "conviction": conviction,
        "support_entry": float(support),
        "resistance_target": float(resistance),
        "invalidation_stop": float(invalidation),
        "memorandum": memo_text,
        "source": "QUANT_DETERMINISTIC_ENGINE",
        "diagnostic_note": reason
    }

def analyze_stock(
    ticker: str,
    overview: Dict[str, Any],
    quant: Dict[str, Any],
    microstructure: Dict[str, Any],
    valuation: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes DeepSeek AI quantitative equity memorandum generation.
    Enforces factual grounding on mathematical moments, empirical microstructure, and audited valuation.
    Falls back to deterministic econometric synthesizer if API key is not configured.
    """
    sym = ticker.upper().strip()
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()

    if not key or key.startswith("sk-your"):
        return generate_fallback_memorandum(
            sym, overview, quant, microstructure, valuation,
            "DeepSeek API key not configured; using deterministic econometric synthesizer."
        )

    client = get_deepseek_client()

    price = overview.get("price", 0.0)
    ann_vol = quant.get("annualized_volatility_pct", 0.0)
    gk_vol = quant.get("garman_klass_volatility_pct", 0.0)
    skew = quant.get("skewness", 0.0)
    kurt = quant.get("excess_kurtosis", 0.0)
    jb_stat = quant.get("jarque_bera_stat", 0.0)
    jb_pvalue = quant.get("jarque_bera_pvalue", 1.0)
    var_95_cf = quant.get("var_95_cornish_fisher_pct", 0.0)
    cvar_95 = quant.get("cvar_95_pct", 0.0)
    mdd = quant.get("max_drawdown_pct", 0.0)
    sharpe = quant.get("sharpe_ratio", 0.0)
    sortino = quant.get("sortino_ratio", 0.0)
    mc = quant.get("monte_carlo", {})
    mc_median = mc.get("expected_median_price", price)
    mc_prob = mc.get("p_profit_pct", 50.0)

    cs_spread = microstructure.get("corwin_schultz_spread_pct", 0.0)
    illiq = microstructure.get("amihud_illiq", 0.0)
    vwap = microstructure.get("vwap", price)
    vwap_dev = microstructure.get("vwap_deviation_pct", 0.0)

    system_prompt = (
        "You are an institutional quantitative equity researcher and financial econometrician evaluating "
        "Indonesia Stock Exchange (IDX / BEI) equities. "
        "Analyze empirical return distributions, extreme-value volatilities, tail risk (Cornish-Fisher VaR & Expected Shortfall), "
        "microstructure liquidity (Corwin-Schultz spread & Amihud illiquidity), and valuation multiples. "
        "STRICT CONSTRAINTS:\n"
        "1. NEVER invent broker codes, orderbook queue counts, or insider rumors.\n"
        "2. Ground every sentence strictly in the provided mathematical moments and accounting ratios.\n"
        "3. Output valid JSON only with keys: econometric_regime, action_recommendation, conviction (High/Moderate/Low), "
        "support_entry (numeric), resistance_target (numeric), invalidation_stop (numeric), memorandum (clean text)."
    )

    user_payload = {
        "ticker": sym,
        "company_name": overview.get("name"),
        "spot_price_idr": price,
        "change_pct": overview.get("change_pct"),
        "volume_lot": overview.get("volume_lot"),
        "turnover_value_idr": overview.get("value_idr"),
        "market_cap_idr": overview.get("market_cap"),
        "statistical_moments": {
            "mean_daily_return_pct": quant.get("mean_daily_return_pct"),
            "annualized_close_volatility_pct": ann_vol,
            "garman_klass_ohlc_volatility_pct": gk_vol,
            "parkinson_high_low_volatility_pct": quant.get("parkinson_volatility_pct"),
            "skewness": skew,
            "excess_kurtosis": kurt,
            "jarque_bera_stat": jb_stat,
            "jarque_bera_pvalue": jb_pvalue,
            "is_normal_distribution": quant.get("is_normal_distribution")
        },
        "risk_and_tail_loss": {
            "var_95_cornish_fisher_pct": var_95_cf,
            "expected_shortfall_cvar_95_pct": cvar_95,
            "max_drawdown_pct": mdd,
            "sharpe_ratio_vs_bi_rate_6pct": sharpe,
            "sortino_ratio": sortino
        },
        "market_microstructure": {
            "corwin_schultz_effective_spread_pct": cs_spread,
            "roll_spread_pct": microstructure.get("roll_spread_pct"),
            "amihud_illiquidity_ratio": illiq,
            "vwap_idr": vwap,
            "vwap_deviation_pct": vwap_dev,
            "liquidity_classification": microstructure.get("liquidity_classification")
        },
        "monte_carlo_gbm_30d": {
            "expected_median_price_idr": mc_median,
            "p_profit_pct": mc_prob,
            "p10_bearish_price": mc.get("expected_bear_price"),
            "p90_bullish_price": mc.get("expected_bull_price")
        },
        "valuation_multiples": {
            "trailing_pe": valuation.get("trailing_pe"),
            "price_to_book": valuation.get("price_to_book"),
            "enterprise_to_ebitda": valuation.get("enterprise_to_ebitda"),
            "return_on_equity_pct": valuation.get("return_on_equity_pct"),
            "debt_to_equity_pct": valuation.get("debt_to_equity_pct")
        }
    }

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate Quantitative Equity Research Memorandum for {sym}:\n{json.dumps(user_payload, indent=2)}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=1000
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        data["ticker"] = sym
        data["source"] = "DEEPSEEK_AI_REASONING"
        return data
    except Exception as e:
        return generate_fallback_memorandum(
            sym, overview, quant, microstructure, valuation,
            f"DeepSeek API query error: {str(e)}; fallback synthesizer engaged."
        )
