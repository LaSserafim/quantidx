"""
main.py — Indonesian Stock Exchange (IDX) Quantitative & Bandarmology Terminal.
Single-server FastAPI backend serving both REST APIs and the accelerated WebGPU frontend.
Run: python3 backend/main.py
"""
import os
import sys
import pathlib
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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
    title="IDX Quant & Bandarmology Terminal",
    description="Institutional-grade equity analysis for Indonesia Stock Exchange (BEI)",
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
    return {
        "status": "online",
        "market": "IDX / Bursa Efek Indonesia",
        "model": ai_analyst.DEEPSEEK_MODEL,
        "deepseek_connected": bool(os.environ.get("DEEPSEEK_API_KEY", ai_analyst.DEEPSEEK_API_KEY))
    }

@app.get("/api/stock/{ticker}")
def get_stock(ticker: str):
    """Returns price quote, volume lot, change %, candles, and market cap."""
    try:
        overview = fetcher.fetch_stock_overview(ticker)
        return overview
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data fetch error: {str(e)}")

@app.get("/api/broksum/{ticker}")
def get_broksum(
    ticker: str,
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to")
):
    """Returns Broker Summary with Top Buyers, Top Sellers, BAVG, SAVG, and Bandarmology status."""
    try:
        broksum = fetcher.fetch_broker_summary(ticker, from_date, to_date)
        return broksum
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BrokSum error: {str(e)}")

@app.get("/api/inventory/{ticker}")
def get_inventory(ticker: str):
    """Returns multi-broker cumulative inventory curves (Net Akum vs Net Dist) over time."""
    try:
        overview = fetcher.fetch_stock_overview(ticker)
        broksum = fetcher.fetch_broker_summary(ticker)
        inv = quant_engine.compute_inventory_series(overview["candles"], broksum)
        return inv
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inventory calculation error: {str(e)}")

@app.get("/api/orderbook/{ticker}")
def get_orderbook(ticker: str):
    """Returns 10-level Depth of Market (DOM) with Order Flow Imbalance and spoofing detection."""
    try:
        ob = fetcher.fetch_orderbook(ticker)
        return ob
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orderbook error: {str(e)}")

@app.get("/api/profile/{ticker}")
def get_profile(ticker: str):
    """Returns sector classification, Board of Commissioners, Directors, Subsidiaries, and Shareholders."""
    try:
        prof = fetcher.fetch_company_profile(ticker)
        return prof
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile error: {str(e)}")

@app.get("/api/news/{ticker}")
def get_news(ticker: str):
    """Returns real-time disclosures, KSEI reports, and financial news items."""
    try:
        news = fetcher.fetch_emiten_news(ticker)
        return news
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"News fetch error: {str(e)}")

@app.get("/api/history/{ticker}")
def get_history(ticker: str):
    """Returns comprehensive transaction history ledger, flow chart data, and broker calendar."""
    try:
        history = fetcher.fetch_transaction_flow_history(ticker)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History fetch error: {str(e)}")

@app.get("/api/quant/{ticker}")
def get_quant(ticker: str):
    """Returns annualized volatility, VaR 95%, Sharpe ratio, Monte Carlo 30-day forecast, and HHI score."""
    try:
        overview = fetcher.fetch_stock_overview(ticker)
        broksum = fetcher.fetch_broker_summary(ticker)
        ob = fetcher.fetch_orderbook(ticker)
        
        q_metrics = quant_engine.compute_quantitative_metrics(overview["candles"])
        bandar_score = quant_engine.compute_bandarmology_quant_score(broksum, ob)
        
        return {
            "ticker": ticker.upper(),
            "metrics": q_metrics,
            "bandarmology_quant": bandar_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quant calculation error: {str(e)}")

@app.post("/api/analyze/{ticker}")
def analyze(ticker: str):
    """Executes DeepSeek AI institutional narrative analysis or quant fallback."""
    try:
        overview = fetcher.fetch_stock_overview(ticker)
        broksum = fetcher.fetch_broker_summary(ticker)
        ob = fetcher.fetch_orderbook(ticker)
        quant = quant_engine.compute_quantitative_metrics(overview["candles"])
        
        analysis = ai_analyst.analyze_stock(ticker, overview, broksum, quant, ob)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Analyst error: {str(e)}")

@app.post("/api/settings/key")
def update_key(req: KeyUpdateRequest):
    """Updates DeepSeek API key dynamically at runtime."""
    if not req.api_key.startswith("sk-"):
        raise HTTPException(status_code=400, detail="Invalid API key format. Key must start with 'sk-'")
    os.environ["DEEPSEEK_API_KEY"] = req.api_key
    return {"status": "success", "message": "DeepSeek API Key updated successfully."}

class StockbitTokenRequest(BaseModel):
    token: str

@app.post("/api/settings/stockbit-token")
def update_stockbit_token(req: StockbitTokenRequest):
    os.environ["STOCKBIT_TOKEN"] = req.token
    fetcher.STOCKBIT_TOKEN = req.token
    return {"status": "success", "message": "Stockbit Token tersimpan! Live Orderbook ITCH aktif."}

# ── Serve Frontend ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "3000"))
    print(f"🚀 Starting IDX Quant & Bandarmology Terminal at http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)
