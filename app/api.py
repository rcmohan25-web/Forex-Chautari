"""
ForexChautari — FastAPI Backend
Forex Market Prediction and Analysis ML Model

Endpoints:
  System:     GET  /health
  Prediction: GET  /predict/latest?pair=EUR_USD
              GET  /predict/all
  Model:      GET  /model-info?pair=EUR_USD
              POST /retrain?pair=EUR_USD  (or /retrain/all)
  Data:       POST /fetch-data
              GET  /history?pair=EUR_USD&n=200
  Portfolio:  GET  /portfolio/signals
              GET  /portfolio/health
  WF:         GET  /walk-forward?pair=EUR_USD
"""

import os
import subprocess
import sys
from typing import Optional as Opt

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse

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
from config.startup_checks import validate_env, warn_if_debug_settings_in_production, verify_database_connectivity

logger = get_logger("api")

# Fail fast — check all required env vars and database connectivity before starting
validate_env()
warn_if_debug_settings_in_production()
verify_database_connectivity()

from app.api_auth import (
    auth_router,
    require_user,
    require_admin,
    require_plan,
    pair_allowed_for_plan,
    rate_limit,
    CurrentUser,
)
from src.database import init_db, is_admin_password_default

init_db()


# ── Trading Request/Response Models ───────────────────────────────────────────

class PlaceTradeRequest(BaseModel):
    instrument: str
    direction: str
    units: int
    sl_pips: float = 20.0
    tp_pips: float = 40.0
    order_type: str = "Market"
    limit_price: Opt[float] = None
    account_db_id: Opt[int] = None


class CloseTradeRequest(BaseModel):
    broker_trade_id: str
    db_trade_id: int
    account_db_id: Opt[int] = None


class ModifyTradeRequest(BaseModel):
    trade_id: str
    account_db_id: Opt[int] = None
    stop_loss: Opt[float] = None
    take_profit: Opt[float] = None


class TradingSettingsUpdate(BaseModel):
    mode: Opt[str] = None
    auto_trade_enabled: Opt[bool] = None
    trading_account_id: Opt[int] = None
    threshold: Opt[float] = None
    risk_pct: Opt[float] = None
    sl_pips: Opt[float] = None
    tp_pips: Opt[float] = None
    units: Opt[int] = None
    max_positions: Opt[int] = None
    use_regime_filter: Opt[bool] = None


# ── Profile & Admin Management Models ──────────────────────────────────────────

class ProfileUpdate(BaseModel):
    full_name: Opt[str] = None
    phone: Opt[str] = None
    email: Opt[str] = None


class PasswordUpdate(BaseModel):
    current_value: str
    new_value: str


class CreateUserRequest(BaseModel):
    username: str
    email: str
    password: str
    full_name: str = ""
    role: str = "user"
    plan: str = "free"


class PlanUpdateRequest(BaseModel):
    plan: str


class AddTradingAccountRequest(BaseModel):
    account_name: str
    api_key: str
    account_id: str
    environment: str = "practice"


class MarkReadRequest(BaseModel):
    pass


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
app.include_router(auth_router)


# ── Default-password firewall ─────────────────────────────────────────────────
# All endpoints except /health and /auth/* are blocked until the admin
# changes the seeded password. This cannot be bypassed by the client.

_ALWAYS_ALLOWED = {"/health", "/auth", "/docs", "/openapi.json", "/redoc"}


@app.middleware("http")
async def require_password_changed(request: Request, call_next):
    path = request.url.path
    # Allow health, auth, and API docs unconditionally
    if any(path == p or path.startswith(f"{p}/") for p in _ALWAYS_ALLOWED):
        return await call_next(request)

    if is_admin_password_default():
        return JSONResponse(
            status_code=503,
            content={
                "type":   "setup_required",
                "title":  "Service Unavailable — First-Run Setup Incomplete",
                "detail": (
                    "The admin account is still using the default password 'admin123'. "
                    "Log in to the admin panel and change it before using any endpoint. "
                    "Only GET /health is available until setup is complete."
                ),
            },
        )

    return await call_next(request)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_pair(pair: str):
    """Load model + data for a given pair. Raises HTTPException on failure."""
    import joblib, json
    from config.settings import model_path, MODEL_PATH, METADATA_PATH

    mp  = model_path(pair)
    mep = meta_path(pair)
    if not os.path.exists(mp):
        # Fall back to legacy single-pair model
        mp, mep = MODEL_PATH, METADATA_PATH
    if not os.path.exists(mp):
        raise HTTPException(
            status_code=503,
            detail=f"No model for {pair}. Run: python train_all.py --fetch"
        )

    model = joblib.load(mp)
    with open(mep) as f:
        metadata = json.load(f)

    csv = data_path(pair)
    if not os.path.exists(csv):
        # Try legacy path
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
            detail=f"Unknown pair '{pair}'. Valid: {ACTIVE_PAIRS}"
        )
    return pair


# ── System ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Liveness check — reports model and data status for all pairs."""
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
                info["trained_at"]   = m.get("trained_at", "?")[:10]
                info["wf_accuracy"]  = m.get("walk_forward_mean_accuracy")
                info["test_accuracy"]= m.get("accuracy_test")
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
    """Return the latest directional signal for one pair."""
    pair = _valid_pair(pair)
    pair_allowed_for_plan(pair, user)
    model, metadata, df = _load_pair(pair)
    feat_cols = metadata["feature_columns"]
    latest    = df[feat_cols].iloc[-1:]
    pred      = int(model.predict(latest)[0])
    prob_up   = float(model.predict_proba(latest)[0][1])
    signal    = "BUY / UP" if pred == 1 else "SELL / DOWN"
    row       = df.iloc[-1]
    return {
        "pair":            pair,
        "signal":          signal,
        "prediction":      pred,
        "probability_up":  round(prob_up, 4),
        "probability_down":round(1 - prob_up, 4),
        "confidence":      _confidence_label(prob_up),
        "latest_close":    float(row["Close"]),
        "latest_date":     str(row["Date"]),
        "threshold_used":  threshold,
        "model_version":   metadata.get("model_version"),
    }


@app.get("/predict/all", tags=["Prediction"])
def predict_all(
    threshold: float = Query(DEFAULT_SIGNAL_THRESHOLD, ge=0.5, le=0.95),
    user:      CurrentUser = Depends(require_user),
):
    """Return signals for all active pairs."""
    allowed = user.allowed_pairs() if not user.is_admin() else ACTIVE_PAIRS
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
    """Get ranked portfolio signals from multi-pair manager."""
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
    """Return model health metrics for all pairs."""
    import json
    result = []
    for pair in ACTIVE_PAIRS:
        mp = meta_path(pair)
        if os.path.exists(mp):
            with open(mp) as f:
                m = json.load(f)
            result.append({
                "pair":         pair,
                "ok":           True,
                "test_accuracy":m.get("accuracy_test"),
                "wf_accuracy":  m.get("walk_forward_mean_accuracy"),
                "trained_at":   m.get("trained_at", "?")[:10],
                "rows":         m.get("rows_total"),
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
    """Return full metadata for a pair's model."""
    pair = _valid_pair(pair)
    pair_allowed_for_plan(pair, user)
    _, metadata, _ = _load_pair(pair)
    return {
        "pair":           pair,
        "feature_columns":metadata["feature_columns"],
        "feature_count":  len(metadata["feature_columns"]),
        "metadata":       metadata,
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
    """Retrain model(s). pair='all' retrains all active pairs."""
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
    """Fetch latest candles for one or all pairs from Oanda."""
    try:
        from src.multi_pair_manager import fetch_all_pairs, fetch_pair_data
        if pair.lower() == "all":
            results = fetch_all_pairs(count=100 if outputsize == "compact" else 500)
            return {"results": results}
        else:
            _valid_pair(pair)
            df = fetch_pair_data(pair, count=100 if outputsize == "compact" else 500)
            return {
                "pair":      pair,
                "rows":      len(df),
                "latest":    str(df["Date"].max().date()),
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", tags=["Data"])
def history(
    pair: str = Query("EUR_USD"),
    n:    int = Query(200, ge=10, le=2000),
    user: CurrentUser = Depends(require_user),
):
    """Return last N OHLC bars for a pair."""
    pair = _valid_pair(pair)
    pair_allowed_for_plan(pair, user)
    csv  = data_path(pair)
    if not os.path.exists(csv):
        raise HTTPException(status_code=404, detail=f"No data for {pair}")
    df = load_forex_data(csv).tail(n)[["Date","Open","High","Low","Close"]]
    df["Date"] = df["Date"].astype(str)
    return {"pair": pair, "rows": df.to_dict(orient="records"), "count": len(df)}


# ── Walk-forward ──────────────────────────────────────────────────────────────

@app.get("/walk-forward", tags=["Model"])
def walk_forward(
    pair: str = Query("EUR_USD"),
    user: CurrentUser = Depends(require_plan("basic", "pro", "enterprise")),
):
    """Return walk-forward validation results for a pair."""
    import pandas as pd
    from config.settings import wf_path
    pair = _valid_pair(pair)
    pair_allowed_for_plan(pair, user)
    wf   = wf_path(pair)
    if not os.path.exists(wf):
        raise HTTPException(status_code=404, detail=f"No WF results for {pair}. Run retrain first.")
    df = pd.read_csv(wf)
    return {
        "pair":   pair,
        "splits": df.to_dict(orient="records"),
        "summary": {
            "total_splits":       len(df),
            "mean_accuracy":      round(float(df["accuracy"].mean()), 4),
            "mean_strategy_return":round(float(df["strategy_return"].mean()), 4),
            "profitable_splits":  int((df["strategy_return"] > 0).sum()),
        },
    }


# ── Trading (writes — replaces direct dashboard calls) ────────────────────────

@app.post("/trading/place", tags=["Trading"])
def trading_place(body: PlaceTradeRequest, user: CurrentUser = Depends(require_user)):
    if not user.is_admin() and not user.can_trade():
        raise HTTPException(403, "Trading requires Pro or Enterprise plan.")
    try:
        if body.order_type == "Market":
            from src.trading_engine import place_trade
            result = place_trade(
                user_id=user.id, instrument=body.instrument, direction=body.direction,
                units=body.units, sl_pips=body.sl_pips, tp_pips=body.tp_pips,
                trade_type="manual", account_db_id=body.account_db_id,
            )
            return {"success": True, "result": result}
        else:
            from src.trading_engine import get_client_for_user, calculate_sl_tp, enforce_hard_risk_limits
            from src.database import log_trade
            client = get_client_for_user(user.id, account_db_id=body.account_db_id)
            enforce_hard_risk_limits(client=client, user_id=user.id, units=body.units,
                                      instrument=body.instrument, sl_pips=body.sl_pips)
            entry = body.limit_price or client.get_live_price(body.instrument)["mid"]
            sl, tp = calculate_sl_tp(body.instrument, body.direction, entry, body.sl_pips, body.tp_pips)
            actual_units = body.units if body.direction == "BUY" else -body.units
            order = client.place_limit_order(body.instrument, actual_units, entry, sl, tp)
            log_trade(user.id, body.instrument, body.direction, entry, body.units, "manual_limit")
            return {"success": True, "result": order, "entry": entry, "sl": sl, "tp": tp}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/trading/close", tags=["Trading"])
def trading_close(body: CloseTradeRequest, user: CurrentUser = Depends(require_user)):
    if not user.is_admin() and not user.can_trade():
        raise HTTPException(403, "Trading requires Pro or Enterprise plan.")
    try:
        from src.trading_engine import close_user_trade
        result = close_user_trade(user.id, body.broker_trade_id, body.db_trade_id,
                                    account_db_id=body.account_db_id)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/trading/modify", tags=["Trading"])
def trading_modify(body: ModifyTradeRequest, user: CurrentUser = Depends(require_user)):
    if not user.is_admin() and not user.can_trade():
        raise HTTPException(403, "Trading requires Pro or Enterprise plan.")
    try:
        from src.trading_engine import get_client_for_user
        client = get_client_for_user(user.id, account_db_id=body.account_db_id)
        result = client.modify_trade_sl_tp(body.trade_id, stop_loss=body.stop_loss,
                                            take_profit=body.take_profit)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/trading/close-all", tags=["Trading"])
def trading_close_all(account_db_id: Opt[int] = Query(None),
                       user: CurrentUser = Depends(require_user)):
    if not user.is_admin() and not user.can_trade():
        raise HTTPException(403, "Trading requires Pro or Enterprise plan.")
    try:
        from src.trading_engine import get_client_for_user
        client = get_client_for_user(user.id, account_db_id=account_db_id)
        results = client.close_all_positions()
        return {"success": True, "results": results}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/trading/signal-trade", tags=["Trading"])
def trading_signal_trade(
    pair: str = Query("EUR_USD"),
    threshold: float = Query(DEFAULT_SIGNAL_THRESHOLD),
    sl_pips: float = Query(20.0),
    tp_pips: float = Query(40.0),
    units: Opt[int] = Query(None),
    account_db_id: Opt[int] = Query(None),
    user: CurrentUser = Depends(require_user),
):
    if not user.is_admin() and not user.can_trade():
        raise HTTPException(403, "Trading requires Pro or Enterprise plan.")
    try:
        from src.paper_trader import PaperTrader
        from src.trading_engine import get_client_for_user
        client = get_client_for_user(user.id, account_db_id=account_db_id)
        pt = PaperTrader(instrument=pair, threshold=threshold,
                          units=units or 1000, use_regime_filter=True,
                          oanda_client=client, user_id=user.id,
                          account_db_id=account_db_id, sl_pips=sl_pips, tp_pips=tp_pips)
        return pt.run_signal_check()
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Account settings (writes) ────────────────────────────────────────────────

@app.post("/account/settings", tags=["Account"])
def account_settings(body: TradingSettingsUpdate, user: CurrentUser = Depends(require_user)):
    from src.database import update_user_trading_settings
    fields = body.dict(exclude_unset=True)
    if "mode" in fields and fields["mode"] in ("manual", "auto") and not user.can_trade():
        raise HTTPException(403, "Manual/auto trading requires Pro or Enterprise.")
    update_user_trading_settings(user.id, **fields)
    return {"success": True}


# ── Account profile / password / accounts ───────────────────────────────────

@app.post("/account/profile", tags=["Account"])
def account_profile(body: ProfileUpdate, user: CurrentUser = Depends(require_user)):
    from src.database import update_user_profile
    fields = body.dict(exclude_unset=True)
    update_user_profile(user.id, **fields)
    return {"success": True}


@app.post("/account/password", tags=["Account"])
def account_password(body: PasswordUpdate, user: CurrentUser = Depends(require_user)):
    from src.database import get_db, verify_password, update_user_password
    if len(body.new_value) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user.id,)).fetchone()
    if not row or not verify_password(body.current_value, row["salt"], row["password_hash"]):
        raise HTTPException(400, "Current password is incorrect.")
    update_user_password(user.id, body.new_value)
    return {"success": True}


@app.post("/account/trading-accounts", tags=["Account"])
def account_add_trading_account(body: AddTradingAccountRequest, user: CurrentUser = Depends(require_user)):
    if body.environment == "live" and not user.is_admin():
        raise HTTPException(403, "Live accounts require admin.")
    try:
        from src.oanda_client import OandaClient
        from src.database import add_trading_account, get_user_trading_settings, update_user_trading_settings
        client = OandaClient(api_key=body.api_key, account_id=body.account_id, environment=body.environment)
        result = client.validate_credentials()
        if not result["valid"]:
            raise HTTPException(400, f"Connection failed: {result.get('error')}")
        new_id = add_trading_account(user.id, body.account_name, body.api_key,
                                       body.account_id, body.environment, is_admin=user.is_admin())
        settings = get_user_trading_settings(user.id)
        if not settings.get("trading_account_id"):
            update_user_trading_settings(user.id, trading_account_id=new_id)
        return {"success": True, "account_db_id": new_id, "balance": result["balance"], "currency": result["currency"]}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/account/trading-accounts/{account_id}", tags=["Account"])
def account_remove_trading_account(account_id: int, user: CurrentUser = Depends(require_user)):
    from src.database import remove_trading_account
    remove_trading_account(account_id, user.id)
    return {"success": True}


@app.post("/account/notifications/mark-read", tags=["Account"])
def account_mark_notifications_read(user: CurrentUser = Depends(require_user)):
    from src.database import mark_notifications_read
    mark_notifications_read(user.id)
    return {"success": True}


# ── Admin user management ────────────────────────────────────────────────────

@app.post("/admin/users", tags=["Admin"])
def admin_create_user(body: CreateUserRequest, user: CurrentUser = Depends(require_admin)):
    from sqlalchemy.exc import IntegrityError
    from src.database import create_user
    try:
        result = create_user(body.username, body.email, body.password,
                              body.full_name, role=body.role, plan=body.plan)
        return {"success": True, "user": result}
    except IntegrityError:
        raise HTTPException(400, "Username or email already taken.")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/admin/users/{target_id}/plan", tags=["Admin"])
def admin_update_plan(target_id: int, body: PlanUpdateRequest, user: CurrentUser = Depends(require_admin)):
    from src.database import update_user_plan
    update_user_plan(target_id, body.plan, user.id)
    return {"success": True}


@app.post("/admin/users/{target_id}/deactivate", tags=["Admin"])
def admin_deactivate(target_id: int, user: CurrentUser = Depends(require_admin)):
    from src.database import deactivate_user
    deactivate_user(target_id, user.id)
    return {"success": True}


@app.post("/admin/users/{target_id}/reactivate", tags=["Admin"])
def admin_reactivate(target_id: int, user: CurrentUser = Depends(require_admin)):
    from src.database import reactivate_user
    reactivate_user(target_id, user.id)
    return {"success": True}


@app.post("/admin/users/{target_id}/disable-auto-trade", tags=["Admin"])
def admin_disable_auto_trade(target_id: int, user: CurrentUser = Depends(require_admin)):
    from src.database import update_user_trading_settings
    update_user_trading_settings(target_id, mode="signals_only", auto_trade_enabled=False)
    return {"success": True}
