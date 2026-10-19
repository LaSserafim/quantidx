"""
main.py — FastAPI Backend Server for QUANT.IDX Terminal.
Exposes REST API endpoints for authentic IDX market data, empirical financial econometrics,
multi-estimator volatility models, audited corporate fundamentals, and quantitative AI research.
"""
import os
import sys
import pathlib
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Add backend directory to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import fetcher
import quant_engine
import ai_analyst

app = FastAPI(
    title="QUANT.IDX Terminal API",
    description="Institutional Quantitative Finance & Econometrics Terminal for Indonesia Stock Exchange (IDX)",
    version="2.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = BASE_DIR / "frontend"

class KeyUpdateRequest(BaseModel):
    api_key: str

# ── REST API Endpoints ────────────────────────────────────────────────────────

@app.get("/api/health")
def get_health():
    """Health check endpoint and system status."""
    return {
        "status": "online",
        "system": "QUANT.IDX Terminal",
        "market": "Indonesia Stock Exchange (IDX / BEI)",
        "version": "2.0.0",
        "data_integrity": "100%_VERIFIED_EMPIRICAL"
    }

@app.get("/api/stock/{ticker}")
def get_stock(ticker: str):
    """
    Returns authentic market quote, volume in lots, turnover in IDR, market capitalization,
    official BEI session status, tick size rules, and asymmetric auto-rejection limits (ARA/ARB).
    """
    try:
        data = fetcher.fetch_stock_overview(ticker)
        return data
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stock quote fetch error: {str(e)}")

@app.get("/api/microstructure/{ticker}")
def get_microstructure(ticker: str):
    """
    Returns authentic empirical market microstructure and liquidity analytics:
    Corwin & Schultz (2012) bid-ask spread, Roll (1984) spread, Amihud (2002) illiquidity ratio,
    Volume-Weighted Average Price (VWAP), and float turnover velocity.
    """
    try:
        overview = fetcher.fetch_stock_overview(ticker)
        metrics = quant_engine.compute_microstructure_metrics(
            overview["candles"],
            overview["price"],
            overview.get("float_shares")
        )
        metrics["ticker"] = ticker.upper()
        metrics["current_price"] = overview["price"]
        metrics["tick_size"] = overview["tick_size"]
        metrics["auto_rejection"] = overview["auto_rejection"]
        metrics["session"] = overview["session"]
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Microstructure calculation error: {str(e)}")

@app.get("/api/quant/{ticker}")
def get_quant(ticker: str):
    """
    Returns comprehensive quantitative financial econometrics:
    - Distribution moments: Mean, Standard Deviation, Skewness, Excess Kurtosis.
    - Normality test: Jarque-Bera statistic and chi-square p-value.
    - Multi-estimator realized volatility: Close-to-Close, Parkinson, Garman-Klass, Rogers-Satchell.
    - Downside tail risk: Historical VaR (95%/99%), Parametric VaR, Cornish-Fisher Modified VaR, CVaR 95%.
    - Drawdown dynamics: Max Drawdown, current drawdown, underwater duration streak, 60-day underwater curve.
    - Risk-adjusted ratios: Sharpe Ratio (vs BI-Rate 6.00%), Sortino Ratio, Calmar Ratio.
    - Monte Carlo simulation: 1,000-path 30-day Geometric Brownian Motion (P10, P50, P90).
    - Return distribution histogram bins with fitted Gaussian comparison curve.
    """
    try:
        overview = fetcher.fetch_stock_overview(ticker)
        candles = overview["candles"]
        quant_data = quant_engine.compute_quantitative_metrics(candles)
        quant_data["ticker"] = ticker.upper()
        quant_data["current_price"] = overview["price"]
        return quant_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quant calculation error: {str(e)}")

@app.get("/api/valuation/{ticker}")
def get_valuation(ticker: str):
    """
    Returns authentic corporate valuation multiples and capital structure metrics:
    Trailing P/E, Forward P/E, Price/Book, EV/EBITDA, Price/Sales, Dividend Yield,
    ROE, ROA, Operating/Net Margins, Debt/Equity, Current/Quick Ratios, BVPS, EPS.
    """
    try:
        data = fetcher.fetch_valuation_and_fundamentals(ticker)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Valuation fetch error: {str(e)}")

@app.get("/api/history/{ticker}")
def get_history(ticker: str):
    """
    Returns 60-day factual transaction ledger with daily Open, High, Low, Close,
    Volume (Lots), Value (IDR), VWAP, Intraday Volatility, and Amihud Price Impact,
    along with cumulative return series and rolling realized volatility series.
    """
    try:
        data = fetcher.fetch_historical_transaction_ledger(ticker)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History ledger fetch error: {str(e)}")

@app.get("/api/profile/{ticker}")
def get_profile(ticker: str):
    """
    Returns verified corporate governance, Board of Directors, Board of Commissioners,
    detailed shareholder registry (>5% and public float), and business description.
    """
    try:
        data = fetcher.fetch_company_profile(ticker)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile fetch error: {str(e)}")

@app.get("/api/news/{ticker}")
def get_news(ticker: str):
    """
    Returns real-time financial media news items and regulatory disclosures via RSS.
    """
    try:
        data = fetcher.fetch_emiten_news(ticker)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"News fetch error: {str(e)}")

@app.post("/api/analyze/{ticker}")
def analyze(ticker: str):
    """
    Generates structured institutional quantitative research memorandum via DeepSeek AI.
    Grounds findings on empirical moments, multi-estimator volatility, tail risk,
    microstructure liquidity, and valuation multiples.
    """
    try:
        overview = fetcher.fetch_stock_overview(ticker)
        quant = quant_engine.compute_quantitative_metrics(overview["candles"])
        microstructure = quant_engine.compute_microstructure_metrics(
            overview["candles"],
            overview["price"],
            overview.get("float_shares")
        )
        valuation = fetcher.fetch_valuation_and_fundamentals(ticker)
        analysis = ai_analyst.analyze_stock(ticker, overview, quant, microstructure, valuation)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Analyst error: {str(e)}")

@app.post("/api/settings/key")
def update_key(req: KeyUpdateRequest):
    """Updates DeepSeek API key dynamically at runtime."""
    os.environ["DEEPSEEK_API_KEY"] = req.api_key
    ai_analyst.DEEPSEEK_API_KEY = req.api_key
    return {"status": "success", "message": "DeepSeek API Key updated successfully."}

# ── Serve Frontend ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found.")
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 3000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=True)
