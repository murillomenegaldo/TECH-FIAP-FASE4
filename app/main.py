"""
LSTM Stock Price Prediction API
Tech Challenge Fase 4 — FastAPI + Monitoring
"""

from __future__ import annotations

import json
import time
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional

import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:  # pragma: no cover — TF not installed in test env
    tf = None  # type: ignore
    _TF_AVAILABLE = False

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("monitoring/api.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# IN-MEMORY MONITORING STORE
# ─────────────────────────────────────────────
monitoring_data = {
    "total_requests": 0,
    "successful_predictions": 0,
    "failed_requests": 0,
    "response_times_ms": [],     # last 1000
    "predictions_history": [],   # last 100
}

MODEL_DIR = os.getenv("MODEL_DIR", "model/artifacts")
os.makedirs("monitoring", exist_ok=True)

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
state: dict = {}


def load_artifacts():
    """Load model, scaler and metadata."""
    try:
        model_path = os.path.join(MODEL_DIR, "lstm_best.keras")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        meta_path = os.path.join(MODEL_DIR, "meta.json")

        if not all(os.path.exists(p) for p in [model_path, scaler_path, meta_path]):
            logger.warning("Artefatos do modelo não encontrados. Use /train para treinar primeiro.")
            return False

        if not _TF_AVAILABLE:
            logger.warning("TensorFlow não disponível.")
            return False
        state["model"] = tf.keras.models.load_model(model_path)
        state["scaler"] = joblib.load(scaler_path)
        with open(meta_path) as f:
            state["meta"] = json.load(f)

        metrics_path = os.path.join(MODEL_DIR, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                state["metrics"] = json.load(f)

        logger.info("✅ Modelo carregado com sucesso.")
        return True
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("monitoring", exist_ok=True)
    load_artifacts()
    yield
    logger.info("API encerrada.")


# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="LSTM Stock Price Prediction API",
    description=(
        "API para previsão de preços de ações usando redes neurais LSTM. "
        "Tech Challenge Fase 4 — Pos Tech Machine Learning Engineering."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# MIDDLEWARE — request timing
# ─────────────────────────────────────────────
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000  # ms
    monitoring_data["total_requests"] += 1
    monitoring_data["response_times_ms"].append(round(elapsed, 2))
    if len(monitoring_data["response_times_ms"]) > 1000:
        monitoring_data["response_times_ms"].pop(0)
    response.headers["X-Response-Time-ms"] = str(round(elapsed, 2))
    return response


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    symbol: Optional[str] = Field(None, description="Ticker (ex: PETR4.SA). Se omitido, usa o símbolo do modelo.")
    days_ahead: int = Field(5, ge=1, le=30, description="Número de dias a prever (1-30)")
    use_latest: bool = Field(True, description="Buscar dados mais recentes automaticamente via yfinance")
    prices: Optional[List[float]] = Field(
        None,
        description="Lista manual de preços de fechamento (mín. sequence_length valores). Usado quando use_latest=False.",
    )


class PredictResponse(BaseModel):
    symbol: str
    predictions: List[dict]
    model_metrics: Optional[dict]
    generated_at: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    symbol: Optional[str]
    uptime_requests: int


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def fetch_recent_data(symbol: str, seq_len: int, features: List[str]) -> np.ndarray:
    """Download last (seq_len + buffer) days of data."""
    buffer = 30
    end = datetime.today()
    start = end - timedelta(days=seq_len * 2 + buffer)
    df = yf.download(symbol, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Nenhum dado encontrado para {symbol}.")

    # Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[features].dropna()
    if len(df) < seq_len:
        raise HTTPException(
            status_code=422,
            detail=f"Dados insuficientes para {symbol}. Necessário: {seq_len}, obtido: {len(df)}."
        )
    return df.values[-seq_len:]   # shape (seq_len, n_features)


def predict_n_days(model, scaler, meta: dict, seed_window: np.ndarray, n: int) -> List[float]:
    """
    Iterative multi-step prediction:
    Usa a janela atual → prediz 1 dia → desliza a janela → repete.
    """
    features = meta["features"]
    target_idx = meta["target_idx"]
    seq_len = meta["sequence_length"]
    n_features = len(features)

    window = seed_window.copy()   # (seq_len, n_features)
    preds = []

    for _ in range(n):
        scaled_window = scaler.transform(window)
        X = scaled_window[np.newaxis, ...]             # (1, seq_len, n_features)
        pred_scaled = model.predict(X, verbose=0)[0, 0]

        # Inverse transform
        dummy = np.zeros((1, n_features))
        dummy[0, target_idx] = pred_scaled
        price = scaler.inverse_transform(dummy)[0, target_idx]
        preds.append(round(float(price), 4))

        # Slide window: shift left, append new row (repeat last row & update Close)
        new_row = window[-1].copy()
        new_row[target_idx] = price
        window = np.vstack([window[1:], new_row])

    return preds


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    return """
    <html><head><title>LSTM Stock API</title></head><body style="font-family:monospace;padding:2rem;background:#0d1117;color:#58a6ff">
    <h1>📈 LSTM Stock Price Prediction API</h1>
    <p>Tech Challenge Fase 4 — Machine Learning Engineering</p>
    <ul>
      <li><a href="/docs" style="color:#3fb950">📖 Swagger UI</a></li>
      <li><a href="/redoc" style="color:#3fb950">📄 ReDoc</a></li>
      <li><a href="/health" style="color:#3fb950">❤️  Health Check</a></li>
      <li><a href="/metrics" style="color:#3fb950">📊 Monitoring Metrics</a></li>
    </ul>
    </body></html>
    """


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    return HealthResponse(
        status="ok" if "model" in state else "model_not_loaded",
        model_loaded="model" in state,
        symbol=state.get("meta", {}).get("symbol"),
        uptime_requests=monitoring_data["total_requests"],
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Returns monitoring metrics for the API."""
    times = monitoring_data["response_times_ms"]
    return {
        "total_requests": monitoring_data["total_requests"],
        "successful_predictions": monitoring_data["successful_predictions"],
        "failed_requests": monitoring_data["failed_requests"],
        "response_time_ms": {
            "avg": round(np.mean(times), 2) if times else 0,
            "p95": round(np.percentile(times, 95), 2) if times else 0,
            "p99": round(np.percentile(times, 99), 2) if times else 0,
            "min": round(min(times), 2) if times else 0,
            "max": round(max(times), 2) if times else 0,
        },
        "model_metrics": state.get("metrics", {}),
        "recent_predictions": monitoring_data["predictions_history"][-10:],
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Returns metadata about the loaded model."""
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    return {
        "meta": state["meta"],
        "metrics": state.get("metrics", {}),
        "model_summary": f"LSTM stacked — {state['meta']['sequence_length']} day window",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(req: PredictRequest):
    """
    Predicts future closing prices using the trained LSTM model.

    - **symbol**: Stock ticker (e.g. PETR4.SA, AAPL). Defaults to model's training symbol.
    - **days_ahead**: How many trading days to predict (1–30).
    - **use_latest**: If true, fetches recent data from Yahoo Finance automatically.
    - **prices**: Provide your own closing prices (requires ≥ sequence_length values).
    """
    if "model" not in state:
        monitoring_data["failed_requests"] += 1
        raise HTTPException(status_code=503, detail="Modelo não carregado. Execute o treinamento primeiro.")

    model = state["model"]
    scaler = state["scaler"]
    meta = state["meta"]
    seq_len = meta["sequence_length"]
    features = meta["features"]
    target_idx = meta["target_idx"]
    symbol = req.symbol or meta["symbol"]

    try:
        if req.use_latest:
            seed_window = fetch_recent_data(symbol, seq_len, features)
        else:
            if not req.prices or len(req.prices) < seq_len:
                raise HTTPException(
                    status_code=422,
                    detail=f"Forneça pelo menos {seq_len} preços quando use_latest=False."
                )
            # Build window: pad other features with last values if only Close given
            close_prices = np.array(req.prices[-seq_len:])
            seed_window = np.zeros((seq_len, len(features)))
            seed_window[:, target_idx] = close_prices

        preds = predict_n_days(model, scaler, meta, seed_window, req.days_ahead)

        today = datetime.today()
        predictions = []
        day_offset = 1
        for p in preds:
            future_date = today + timedelta(days=day_offset)
            # Skip weekends
            while future_date.weekday() >= 5:
                future_date += timedelta(days=1)
                day_offset += 1
            predictions.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "predicted_close": p,
            })
            day_offset += 1

        monitoring_data["successful_predictions"] += 1
        monitoring_data["predictions_history"].append({
            "symbol": symbol,
            "days_ahead": req.days_ahead,
            "first_prediction": predictions[0] if predictions else None,
            "timestamp": datetime.utcnow().isoformat(),
        })
        if len(monitoring_data["predictions_history"]) > 100:
            monitoring_data["predictions_history"].pop(0)

        logger.info(f"Previsão gerada | {symbol} | {req.days_ahead} dia(s)")

        return PredictResponse(
            symbol=symbol,
            predictions=predictions,
            model_metrics=state.get("metrics"),
            generated_at=datetime.utcnow().isoformat(),
        )

    except HTTPException:
        monitoring_data["failed_requests"] += 1
        raise
    except Exception as e:
        monitoring_data["failed_requests"] += 1
        logger.exception(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """Reload model artifacts from disk (useful after retraining)."""
    ok = load_artifacts()
    if ok:
        return {"status": "reloaded", "symbol": state["meta"]["symbol"]}
    raise HTTPException(status_code=500, detail="Falha ao recarregar o modelo.")


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
