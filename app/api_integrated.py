"""
ForexChautari — FastAPI Backend  (auth-integrated)
app/api.py

All endpoints now require a JWT Bearer token except:
  GET  /health     — public (liveness/monitoring)
  POST /auth/login — public (token issuance)
  POST /auth/refresh
  POST /auth/logout
  GET  /auth/me

New plan-gating:
  free/basic : /predict/latest, /predict/all  (plan limits which pairs)
  basic+     : /walk-forward
  pro+       : /portfolio/signals
  admin only : /retrain, /fetch-data
"""

import os
import subprocess
import sys
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from config.settings import (
    APP_BRAND, APP_NAME, APP_VERSION,
    ACTIVE_PAIRS, data_path, meta_path,
    DEFAULT_SIGNAL_THRESHOLD,
)
from src.data_loader import load_forex_data
from src.features import add_features
from src.logger import get_logger
from src.schemas import (
    HealthResponse, PredictionResponse,
    ModelInfoResponse, FetchResponse,
)

# ── Auth imports ──────────────────────────────────────────────────────────────
from app.api_auth import (
    auth_router,
    require_user,
    require_admin,
    require_plan,
    pair_allowed_for_plan,
    rate_limit,
    CurrentUser,
)

logger = get_logger("api")

app = FastAPI(
    title=f"{APP_BRAND} API",
    description=f"{APP_NAME} — Multi-pair ML signal API",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount auth routes  (/auth/login, /auth/refresh, /auth/logout, /auth/me) ──
app.include_router(auth_router)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_pair(pair: str):
    """Load model + data for a given pair. Raises HTTPException on failure."""
    import joblib, json
    from config.settings import model_path, MODEL_PATH, METADATA_PATH

    mp  = model_path(pair)
    mep = meta_path(pair)
    if not os.path.exists(mp):
        mp, mep = MODEL_PATH, METADATA_PATH
    if not os.path.exists(mp):
        raise HTTPException(
            status_code=503,
            detail=f"No model for {pair}. Run: python train_all.py --fetch",
        )

    model = joblib.load(mp)
    with open(mep) as f:
        metadata = json.load(f)

    csv = data_path(pair)
    if not os.path.exists(csv):
        from config.settings import DATA_PATH
        csv = DATA_PATH
    if not os.path.exists(csv):
        raise HTTPException(status_code=503, detail=f"No data for {pair}")

    df = add_features(load_forex_data(csv))
    return model, metadata, df


def _confidence_label(prob_up: float) -> str:
    gap = abs(prob_up - 0.5)
    if gap >= 0.15: return "high"
    if gap >= 0.08: return "medium"
    return "low"


def _valid_pair(pair: str) -> str:
    pair = pair.upper()
    if pair not in ACTIVE_PAIRS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pair '{pair}'. Valid: {ACTIVE_PAIRS}",
        )
    return pair


# ── System ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """
    Public liveness check — no auth required.
    Reports model and data status for all pairs.
    """
    result = {
        "status":  "ok",
        "brand":   APP_BRAND,
        "version": APP_VERSION,
        "pairs":   {},
    }
    any_ok = False
    for pair in ACTIVE_PAIRS:
        mp  = meta_path(pair)
        csv = data_path(pair)
        model_ok = os.path.exists(mp)
        data_ok  = os.path.exists(csv)
        info: dict = {"model": model_ok, "data": data_ok}
        if model_ok:
            try:
                import json
                with open(mp) as f: m = json.load(f)
                info["trained_at"]    = m.get("trained_at", "?")[:10]
                info["wf_accuracy"]   = m.get("walk_forward_mean_accuracy")
                info["test_accuracy"] = m.get("accuracy_test")
            except Exception:
                pass
        if data_ok:
            try:
                df = load_forex_data(csv)
                info["rows"]        = len(df)
                info["latest_date"] = str(df["Date"].max().date())
            except Exception:
                pass
        result["pairs"][pair] = info
        if model_ok and data_ok:
            any_ok = True
    if not any_ok:
        result["status"] = "degraded"
    return result


# ── Prediction ────────────────────────────────────────────────────────────────

@app.get("/predict/latest", tags=["Prediction"])
def predict_latest(
    pair:      str   = Query("EUR_USD", description="Pair e.g. EUR_USD"),
    threshold: float = Query(DEFAULT_SIGNAL_THRESHOLD, ge=0.5, le=0.95),
    user:      CurrentUser = Depends(require_user),
):
    """
    Return the latest directional signal for one pair.
    Requires auth. Pair access is gated by subscription plan.
    """
    pair = _valid_pair(pair)
    pair_allowed_for_plan(pair, user)     # raises 403 if the pair is above the user's plan

    model, metadata, df = _load_pair(pair)
    feat_cols = metadata["feature_columns"]
    latest    = df[feat_cols].iloc[-1:]
    pred      = int(model.predict(latest)[0])
    prob_up   = float(model.predict_proba(latest)[0][1])
    signal    = "BUY / UP" if pred == 1 else "SELL / DOWN"
    row       = df.iloc[-1]
    return {
        "pair":             pair,
        "signal":           signal,
        "prediction":       pred,
        "probability_up":   round(prob_up, 4),
        "probability_down": round(1 - prob_up, 4),
        "confidence":       _confidence_label(prob_up),
        "latest_close":     float(row["Close"]),
        "latest_date":      str(row["Date"]),
        "threshold_used":   threshold,
        "model_version":    metadata.get("model_version"),
    }


@app.get("/predict/all", tags=["Prediction"])
def predict_all(
    threshold: float = Query(DEFAULT_SIGNAL_THRESHOLD, ge=0.5, le=0.95),
    user:      CurrentUser = Depends(require_user),
):
    """
    Return signals for all pairs allowed under the user's plan.
    Admins see all pairs regardless of plan.
    """
    allowed = ACTIVE_PAIRS if user.is_admin() else user.allowed_pairs()
    results = []
    for pair in allowed:
        try:
            model, metadata, df = _load_pair(pair)
            feat_cols = metadata["feature_columns"]
            latest    = df[feat_cols].iloc[-1:]
            pred      = int(model.predict(latest)[0])
            prob_up   = float(model.predict_proba(latest)[0][1])
            results.append({
                "pair":           pair,
                "signal":         "BUY" if pred == 1 else "SELL",
                "probability_up": round(prob_up, 4),
                "confidence":     _confidence_label(prob_up),
                "latest_date":    str(df["Date"].iloc[-1]),
                "ok":             True,
            })
        except Exception as e:
            results.append({"pair": pair, "ok": False, "error": str(e)})
    return {"pairs": results, "threshold": threshold}


# ── Portfolio ─────────────────────────────────────────────────────────────────

@app.get("/portfolio/signals", tags=["Portfolio"])
def portfolio_signals(
    threshold: float = Query(DEFAULT_SIGNAL_THRESHOLD),
    user:      CurrentUser = Depends(require_plan("pro", "enterprise")),
):
    """
    Get ranked portfolio signals.  Pro and Enterprise plans only.
    Non-admin users only see pairs within their plan allowance.
    """
    try:
        from src.multi_pair_manager import get_portfolio_signals
        df = get_portfolio_signals(threshold)
        if not user.is_admin():
            allowed = user.allowed_pairs()
            df = df[df["pair"].isin(allowed)]
        return {"signals": df.to_dict(orient="records"), "count": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio/health", tags=["Portfolio"])
def portfolio_health(
    user: CurrentUser = Depends(require_user),
):
    """Return model health metrics for all pairs. Requires auth."""
    import json
    result = []
    for pair in ACTIVE_PAIRS:
        mp = meta_path(pair)
        if os.path.exists(mp):
            with open(mp) as f:
                m = json.load(f)
            result.append({
                "pair":          pair,
                "ok":            True,
                "test_accuracy": m.get("accuracy_test"),
                "wf_accuracy":   m.get("walk_forward_mean_accuracy"),
                "trained_at":    m.get("trained_at", "?")[:10],
                "rows":          m.get("rows_total"),
            })
        else:
            result.append({"pair": pair, "ok": False})
    return {"pairs": result}


# ── Model ─────────────────────────────────────────────────────────────────────

@app.get("/model-info", tags=["Model"])
def model_info(
    pair: str = Query("EUR_USD"),
    user: CurrentUser = Depends(require_user),
):
    """Return full metadata for a pair's model. Pair access is plan-gated."""
    pair = _valid_pair(pair)
    pair_allowed_for_plan(pair, user)
    _, metadata, _ = _load_pair(pair)
    return {
        "pair":            pair,
        "feature_columns": metadata["feature_columns"],
        "feature_count":   len(metadata["feature_columns"]),
        "metadata":        metadata,
    }


@app.post(
    "/retrain",
    tags=["Model"],
    dependencies=[Depends(rate_limit("retrain"))],
)
def retrain(
    pair: str = Query("all", description="Pair to retrain, or 'all'"),
    user: CurrentUser = Depends(require_admin),
):
    """
    Retrain model(s).  Admin only.  Rate-limited to 3 calls/hour.
    pair='all' retrains all active pairs.
    """
    try:
        args = [sys.executable, "train_all.py"]
        if pair.lower() != "all":
            _valid_pair(pair)
            args.append(pair)
        result = subprocess.run(args, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Training failed:\n{result.stderr}")
        return {"success": True, "pair": pair, "output": result.stdout[-800:]}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Training timed out (300s).")


# ── Data ──────────────────────────────────────────────────────────────────────

@app.post(
    "/fetch-data",
    tags=["Data"],
    dependencies=[Depends(rate_limit("retrain"))],
)
def fetch_data(
    pair:       str = Query("all"),
    outputsize: str = Query("compact", pattern="^(compact|full)$"),
    user:       CurrentUser = Depends(require_admin),
):
    """Fetch latest candles for one or all pairs from Oanda.  Admin only."""
    try:
        from src.multi_pair_manager import fetch_all_pairs, fetch_pair_data
        if pair.lower() == "all":
            results = fetch_all_pairs(count=100 if outputsize == "compact" else 500)
            return {"results": results}
        else:
            _valid_pair(pair)
            df = fetch_pair_data(pair, count=100 if outputsize == "compact" else 500)
            return {
                "pair":   pair,
                "rows":   len(df),
                "latest": str(df["Date"].max().date()),
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", tags=["Data"])
def history(
    pair: str = Query("EUR_USD"),
    n:    int = Query(200, ge=10, le=2000),
    user: CurrentUser = Depends(require_user),
):
    """Return last N OHLC bars for a pair. Pair access is plan-gated."""
    pair = _valid_pair(pair)
    pair_allowed_for_plan(pair, user)
    csv  = data_path(pair)
    if not os.path.exists(csv):
        raise HTTPException(status_code=404, detail=f"No data for {pair}")
    df = load_forex_data(csv).tail(n)[["Date", "Open", "High", "Low", "Close"]]
    df["Date"] = df["Date"].astype(str)
    return {"pair": pair, "rows": df.to_dict(orient="records"), "count": len(df)}


# ── Walk-forward ──────────────────────────────────────────────────────────────

@app.get("/walk-forward", tags=["Model"])
def walk_forward(
    pair: str = Query("EUR_USD"),
    user: CurrentUser = Depends(require_plan("basic", "pro", "enterprise")),
):
    """
    Return walk-forward validation results for a pair.
    Available from Basic plan upward.
    """
    import pandas as pd
    from config.settings import wf_path
    pair = _valid_pair(pair)
    pair_allowed_for_plan(pair, user)
    wf   = wf_path(pair)
    if not os.path.exists(wf):
        raise HTTPException(status_code=404, detail=f"No WF results for {pair}.")
    df = pd.read_csv(wf)
    return {
        "pair":   pair,
        "splits": df.to_dict(orient="records"),
        "summary": {
            "total_splits":        len(df),
            "mean_accuracy":       round(float(df["accuracy"].mean()), 4),
            "mean_strategy_return":round(float(df["strategy_return"].mean()), 4),
            "profitable_splits":   int((df["strategy_return"] > 0).sum()),
        },
    }
