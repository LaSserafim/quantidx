"""
ai_analyst.py — DeepSeek AI Institutional Analyst & Quant Heuristic Fallback Engine.
Generates institutional Indonesian equity research and bandarmology execution strategy.
"""
import os
import json
from typing import Dict, Any, Optional
from openai import OpenAI

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

def get_deepseek_client() -> OpenAI:
    key = os.environ.get("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
    return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)

def generate_fallback_analysis(ticker: str, overview: Dict[str, Any], broksum: Dict[str, Any], quant: Dict[str, Any], ob: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """
    Deterministic Quantitative Rule Engine: Produces rigorous institutional analysis
    when DeepSeek API key is invalid (401), rate-limited, or offline.
    """
    price = overview.get("price", 0)
    change = overview.get("change_pct", 0)
    vol_ann = quant.get("annualized_volatility_pct", 0)
    var_95 = quant.get("var_95_daily_pct", 0)
    sharpe = quant.get("sharpe_ratio", 0)
    p_profit = quant.get("monte_carlo", {}).get("p_profit_pct", 50)
    median_target = quant.get("monte_carlo", {}).get("expected_median_price", price)

    bandar = broksum.get("bandar_detector", {})
    bandar_status = bandar.get("status", "Neutral")
    top3_buyers = [f"{b['broker_code']} (avg {b['avg_price']})" for b in broksum.get("buyers", [])[:3]]
    top3_sellers = [f"{s['broker_code']} (avg {s['avg_price']})" for s in broksum.get("sellers", [])[:3]]

    flows = broksum.get("flows", {})
    net_foreign = flows.get("net_foreign_idr", 0)
    net_retail = flows.get("net_retail_idr", 0)

    # Determine Phase
    if "Accumulation" in bandar_status and change >= 0:
        phase = "Akumulasi / Pre-Markup"
        conviction = "Tinggi"
        action = "Buy on Weakness"
    elif "Accumulation" in bandar_status and change < 0:
        phase = "Akumulasi Saat Koreksi (Absorption)"
        conviction = "Sedang"
        action = "Scale-in Entry"
    elif "Distribution" in bandar_status and change > 0:
        phase = "Distribusi Terselubung (Bull Trap)"
        conviction = "Tinggi"
        action = "Take Profit / Sell on Strength"
    elif "Distribution" in bandar_status:
        phase = "Fase Markdown / Distribusi Terbuka"
        conviction = "Tinggi"
        action = "Avoid / Tight Stop"
    else:
        phase = "Konsolidasi / Rotasi Netral"
        conviction = "Rendah"
        action = "Wait and See"

    # Support / Resistance / Invalidation calculations based on tick size and VaR
    support = round(price * (1 - (var_95 / 100)), 0)
    resistance = round(max(price * 1.05, median_target), 0)
    invalidation = round(support * 0.96, 0)

    narrative = (
        f"Struktur pasar {ticker} saat ini menunjukkan status **{bandar_status}** dengan konsentrasi broker pembeli utama: "
        f"{', '.join(top3_buyers)} berhadapan dengan tekanan jual dari {', '.join(top3_sellers)}. "
        f"Net Foreign tercatat {flows.get('net_foreign_fmt', '0 B')} dan Retail Flow sebesar {flows.get('net_retail_fmt', '0 B')}. "
        f"Volatilitas tahunan berada pada level {vol_ann}%, dengan risiko harian Value at Risk (VaR 95%) sebesar {var_95}%. "
        f"Simulasi Monte Carlo 30-hari memproyeksikan probabilitas profitabilitas sebesar {p_profit}% dengan target median IDR {median_target:,.0f}."
    )

    strategic_verdict = (
        f"Secara kuantitatif, saham ini berada dalam {phase}. "
        f"Ketika smart money mendominasi transaksi, level kritis yang perlu diawasi adalah batas bawah IDR {support:,.0f}. "
        f"Apabila harga menembus ke bawah IDR {invalidation:,.0f}, maka tesis akumulasi batal (invalidated)."
    )

    return {
        "is_live_ai": False,
        "ai_status_badge": "QUANT ANALYST ENGINE (DETERMINISTIC)",
        "ai_status_note": f"DeepSeek API Notice: {reason}. Beralih otomatis ke Mesin Kuantitatif Presisi Tinggi.",
        "ticker": ticker,
        "phase": phase,
        "conviction": conviction,
        "action": action,
        "support_level": support,
        "resistance_level": resistance,
        "invalidation_level": invalidation,
        "monte_carlo_p_profit": p_profit,
        "narrative": narrative,
        "strategic_verdict": strategic_verdict,
        "smart_money_summary": f"Akumulator Dominan: {', '.join([b['broker_code'] for b in broksum.get('buyers', [])[:3]])}",
        "retail_risk_alert": "Waspada perangkap ritel!" if net_retail > 0 and "Distribution" in bandar_status else "Tekanan ritel terkendali."
    }

def analyze_stock(ticker: str, overview: Dict[str, Any], broksum: Dict[str, Any], quant: Dict[str, Any], ob: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls DeepSeek API if available; otherwise safely falls back to Quant Rule Engine.
    """
    sym = ticker.upper().strip()
    client = get_deepseek_client()

    bandar = broksum.get("bandar_detector", {})
    flows = broksum.get("flows", {})
    top3_buyers = [f"{b['broker_code']} (lot {b['net_lot']}, avg {b['avg_price']})" for b in broksum.get("buyers", [])[:3]]
    top3_sellers = [f"{s['broker_code']} (lot {s['net_lot']}, avg {s['avg_price']})" for s in broksum.get("sellers", [])[:3]]

    system_prompt = (
        "Anda adalah Kepala Analis Kuantitatif & Spesialis Bandarmology Pasar Modal Indonesia (IDX/BEI) di dana institusi papan atas. "
        "PRINSIP INVIOLABLE: DILARANG KERAS HALUSINASI. DILARANG MENGARANG ANGKA, REKOMENDASI PALSU, ATAU KODE BROKER YANG TIDAK TERCANTUM. "
        "Analisa HANYA fakta angka riil yang diberikan secara objektif, dingin, dan skeptis (anti-slop). "
        "Jika suatu metrik memiliki risiko atau bias, jelaskan secara transparan batas ketidakpastiannya kepada investor institusi."
    )

    user_prompt = f"""ANALISA EMITEN IDX: {sym}
Harga Terakhir: IDR {overview.get('price'):,.0f} ({overview.get('change_pct'):+.2f}%)
Volume: {overview.get('volume_lot'):,} lot | Nilai Transaksi: IDR {overview.get('value_idr'):,.0f}

DATA BANDARMOLOGY (BROKER SUMMARY):
- Status Bandar: {bandar.get('status')}
- Top 1 Konsentrasi: {bandar.get('top1_pct')}% | Top 3: {bandar.get('top3_pct')}%
- Top 3 Akumulator: {', '.join(top3_buyers)}
- Top 3 Distributor: {', '.join(top3_sellers)}
- Net Foreign Flow: {flows.get('net_foreign_fmt')} ({flows.get('foreign_sentiment')})
- Retail Flow: {flows.get('net_retail_fmt')} ({flows.get('retail_sentiment')})

METRIK KUANTITATIF & RISIKO:
- Volatilitas Tahunan: {quant.get('annualized_volatility_pct')}%
- Value at Risk (1D VaR 95%): {quant.get('var_95_daily_pct')}%
- Sharpe Ratio (Risk-free BI Rate 6%): {quant.get('sharpe_ratio')}
- Max Drawdown: {quant.get('max_drawdown_pct')}%
- Probabilitas Profitabilitas Monte Carlo (30H): {quant.get('monte_carlo', {}).get('p_profit_pct')}% (Target Median: IDR {quant.get('monte_carlo', {}).get('expected_median_price'):,.0f})

ORDERBOOK DEPTH & MICROSTRUCTURE:
- Bid Volume Ratio: {ob.get('bid_ratio')}% vs Offer Ratio: {ob.get('offer_ratio')}%
- Order Flow Imbalance (OFI): {ob.get('order_flow_imbalance'):,} lot
- Spoofing Alert: {ob.get('spoofing_alert') or 'Tidak terdeteksi manipulasi fake bid.'}

Outputkan HANYA JSON valid tanpa markdown formatting tambahan dengan struktur:
{{
  "phase": "Akumulasi / Pre-Markup / Distribusi / Markdown / Konsolidasi",
  "conviction": "Tinggi / Sedang / Rendah",
  "action": "Buy on Weakness / Scale-in / Take Profit / Wait and See",
  "support_level": 14200,
  "resistance_level": 15800,
  "invalidation_level": 13900,
  "narrative": "Penjelasan mendalam 2 paragraf mengenai pergerakan smart money vs ritel dan likuiditas",
  "strategic_verdict": "Kesimpulan taktis untuk trader dan investor institusi",
  "smart_money_summary": "Ringkasan manuver broker bandar utama",
  "retail_risk_alert": "Peringatan khusus mengenai risiko ritel"
}}"""

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            timeout=12.0
        )
        content = response.choices[0].message.content
        res_json = json.loads(content)
        res_json["is_live_ai"] = True
        res_json["ai_status_badge"] = "DEEPSEEK LIVE AI ANALYST"
        res_json["ai_status_note"] = "Analisa dihasilkan langsung via model DeepSeek AI secara real-time."
        res_json["ticker"] = sym
        return res_json
    except Exception as e:
        err_msg = str(e)
        return generate_fallback_analysis(sym, overview, broksum, quant, ob, err_msg)
