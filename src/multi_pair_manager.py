"""
Multi-pair manager — EUR/USD, GBP/USD, USD/JPY, AUD/USD.

Each pair gets its own:
  data/{PAIR}.csv
  models/{PAIR}_model.pkl
  models/{PAIR}_metadata.json
  models/{PAIR}_wf_results.csv

Key fix over previous version: model save/load now pass file paths
directly (joblib + json) instead of patching global settings, which
was unreliable and caused models to save to the wrong path.
"""

import os
import json
import joblib
import pandas as pd
from datetime import datetime
from typing import Optional

from src.oanda_client import OandaClient
from src.data_loader import load_forex_data
from src.features import add_features, FEATURE_COLUMNS_V2
from src.model import train_random_forest, evaluate_model
from src.train_pipeline import walk_forward_validation
from src.regime_detector import RegimeDetector
from src.alerter import Alerter
from src.logger import get_logger
from config.settings import (
    PAIRS, ACTIVE_PAIRS,
    data_path, model_path, meta_path, wf_path,
    DEFAULT_SIGNAL_THRESHOLD, DEFAULT_WF_TRAIN_SIZE,
    DEFAULT_WF_TEST_SIZE, DEFAULT_WF_STEP_SIZE,
    DEFAULT_SPREAD_COST, DEFAULT_MAX_POSITIONS,
)

logger = get_logger("multi_pair")


# ── Direct save/load (no global path patching) ────────────────────────────────

def _save_model(model, feature_columns, metadata, pair):
    """Save model + metadata directly to pair-specific paths."""
    os.makedirs("models", exist_ok=True)
    mp  = model_path(pair)
    mep = meta_path(pair)
    joblib.dump(model, mp)
    full_meta = {"feature_columns": feature_columns, **metadata}
    with open(mep, "w") as f:
        json.dump(full_meta, f, indent=2)
    logger.info(f"{pair}: model saved → {mp}")


def _load_model(pair):
    """Load model + metadata directly from pair-specific paths."""
    mp  = model_path(pair)
    mep = meta_path(pair)
    if not os.path.exists(mp):
        raise FileNotFoundError(f"No model for {pair} at {mp}. Run: python train_all.py --fetch")
    if not os.path.exists(mep):
        raise FileNotFoundError(f"No metadata for {pair} at {mep}. Run: python train_all.py --fetch")
    model = joblib.load(mp)
    with open(mep) as f:
        metadata = json.load(f)
    return model, metadata


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_pair_data(pair: str, count: int = 500, granularity: str = "D") -> pd.DataFrame:
    """Fetch candles from Oanda and merge with existing CSV for this pair."""
    oanda  = OandaClient()
    new_df = oanda.get_candles(instrument=pair, granularity=granularity, count=count)

    save = data_path(pair)
    os.makedirs("data", exist_ok=True)

    if os.path.exists(save):
        try:
            existing = pd.read_csv(save, parse_dates=["Date"])
            merged = (
                pd.concat([existing, new_df], ignore_index=True)
                .drop_duplicates(subset=["Date"])
                .sort_values("Date")
                .reset_index(drop=True)
            )
        except Exception:
            merged = new_df
    else:
        merged = new_df

    merged.to_csv(save, index=False)
    logger.info(f"{pair}: {len(merged)} rows saved to {save}")
    return merged


def fetch_all_pairs(count: int = 500) -> dict:
    """Fetch and save data for all active pairs. Returns {pair: result}."""
    results = {}
    for pair in ACTIVE_PAIRS:
        try:
            df = fetch_pair_data(pair, count=count)
            results[pair] = {"ok": True, "rows": len(df)}
        except Exception as e:
            results[pair] = {"ok": False, "error": str(e)}
            logger.error(f"{pair}: fetch failed — {e}")
    return results


# ── Training ──────────────────────────────────────────────────────────────────

def train_pair(
    pair: str,
    threshold: float = DEFAULT_SIGNAL_THRESHOLD,
    spread_cost: float = DEFAULT_SPREAD_COST,
) -> dict:
    """Train a model for one pair and save directly to models/{PAIR}_model.pkl."""
    csv = data_path(pair)
    if not os.path.exists(csv):
        raise FileNotFoundError(
            f"No data for {pair} at {csv}. Run: python train_all.py --fetch"
        )

    df = load_forex_data(csv)
    df = add_features(df)

    if len(df) < 400:
        raise ValueError(f"{pair}: only {len(df)} rows after feature engineering — need 400+.")

    # Walk-forward validation
    wf_df = walk_forward_validation(
        df=df,
        feature_columns=FEATURE_COLUMNS_V2,
        train_size=DEFAULT_WF_TRAIN_SIZE,
        test_size=DEFAULT_WF_TEST_SIZE,
        step_size=DEFAULT_WF_STEP_SIZE,
        threshold=threshold,
        spread_cost=spread_cost,
    )
    os.makedirs("models", exist_ok=True)
    wf_df.to_csv(wf_path(pair), index=False)

    # Final model on 80% of data, test on held-out 20%
    split = int(len(df) * 0.8)
    tr, te = df.iloc[:split], df.iloc[split:]
    model  = train_random_forest(tr[FEATURE_COLUMNS_V2], tr["target"])
    _, _, tr_m = evaluate_model(model, tr[FEATURE_COLUMNS_V2], tr["target"])
    _, _, te_m = evaluate_model(model, te[FEATURE_COLUMNS_V2], te["target"])

    metadata = {
        "pair":                          pair,
        "model_version":                 "v2",
        "feature_set":                   "FEATURE_COLUMNS_V2",
        "rows_total":                    len(df),
        "rows_train":                    len(tr),
        "rows_test":                     len(te),
        "accuracy_train":                tr_m["accuracy"],
        "accuracy_test":                 te_m["accuracy"],
        "walk_forward_mean_accuracy":    float(wf_df["accuracy"].mean()),
        "walk_forward_mean_strategy_return": float(wf_df["strategy_return"].mean()),
        "walk_forward_mean_profit_factor": float(wf_df["profit_factor"].replace([float("inf")], 999).mean())
                                            if "profit_factor" in wf_df else None,
        "walk_forward_mean_expectancy":  float(wf_df["expectancy"].mean()) if "expectancy" in wf_df else None,
        "walk_forward_mean_sharpe":      float(wf_df["sharpe"].mean()) if "sharpe" in wf_df else None,
        "walk_forward_mean_exposure":    float(wf_df["exposure"].mean()) if "exposure" in wf_df else None,
        "walk_forward_profitable_splits":    int((wf_df["strategy_return"] > 0).sum()),
        "walk_forward_total_splits":         len(wf_df),
        "rf_max_depth":                  5,
        "rf_min_samples_leaf":           20,
        "rf_n_estimators":               200,
        "signal_threshold":              threshold,
        "trained_at":                    datetime.utcnow().isoformat(),
    }

    # Save directly — no global path patching
    _save_model(model, FEATURE_COLUMNS_V2, metadata, pair)

    logger.info(
        f"{pair}: trained — train={tr_m['accuracy']:.4f} "
        f"test={te_m['accuracy']:.4f} "
        f"wf={wf_df['accuracy'].mean():.4f}"
    )
    return metadata


def train_all_pairs() -> dict:
    """Train models for all active pairs. Returns {pair: result dict}."""
    results = {}
    for pair in ACTIVE_PAIRS:
        try:
            meta = train_pair(pair)
            results[pair] = {"ok": True, "metadata": meta}
        except Exception as e:
            results[pair] = {"ok": False, "error": str(e)}
            logger.error(f"{pair}: training failed — {e}")
    return results


# ── Signal generation ─────────────────────────────────────────────────────────

def get_pair_signal(pair: str, threshold: float = DEFAULT_SIGNAL_THRESHOLD) -> dict:
    """
    Generate a signal for a single pair using that pair's own model.
    Loads model directly from models/{PAIR}_model.pkl — no global path patching.
    """
    csv = data_path(pair)

    if not os.path.exists(csv):
        return {"pair": pair, "ok": False, "reason": f"No data — run: python train_all.py --fetch"}

    try:
        model, metadata = _load_model(pair)
    except FileNotFoundError as e:
        return {"pair": pair, "ok": False, "reason": str(e)}

    try:
        df         = add_features(load_forex_data(csv))
        feat_cols  = metadata["feature_columns"]
        latest_row = df[feat_cols].iloc[-1:]
        pred       = int(model.predict(latest_row)[0])
        prob_up    = float(model.predict_proba(latest_row)[0][1])
    except Exception as e:
        return {"pair": pair, "ok": False, "reason": f"Prediction error: {e}"}

    is_buy     = pred == 1
    signal     = "BUY" if is_buy else "SELL"
    gap        = abs(prob_up - 0.5)
    confidence = "HIGH" if gap >= 0.15 else ("MEDIUM" if gap >= 0.08 else "LOW")
    above_thresh = prob_up >= threshold if is_buy else (1 - prob_up) >= threshold

    try:
        regime    = RegimeDetector().detect(df)
        tradeable = RegimeDetector().is_tradeable(regime, signal)
    except Exception as e:
        logger.warning(f"{pair}: regime detection failed — {e}")
        regime    = {"adx_regime": "unknown", "trend_direction": "unknown",
                     "vol_regime": "unknown"}
        tradeable = False

    return {
        "pair":            pair,
        "ok":              True,
        "signal":          signal,
        "prob_up":         round(prob_up, 4),
        "prob_down":       round(1 - prob_up, 4),
        "confidence":      confidence,
        "above_threshold": above_thresh,
        "regime":          regime["adx_regime"],
        "trend":           regime["trend_direction"],
        "vol_regime":      regime["vol_regime"],
        "regime_ok":       tradeable,
        "tradeable":       above_thresh and tradeable,
        "wf_accuracy":     metadata.get("walk_forward_mean_accuracy"),
        "test_accuracy":   metadata.get("accuracy_test"),
        "latest_date":     str(df["Date"].iloc[-1].date()),
    }


def get_portfolio_signals(threshold: float = DEFAULT_SIGNAL_THRESHOLD) -> pd.DataFrame:
    """
    Get signals for all active pairs, ranked by confidence.
    Failed pairs (no model/data) sort to the bottom — never crash.
    """
    signals = [get_pair_signal(pair, threshold) for pair in ACTIVE_PAIRS]
    df      = pd.DataFrame(signals)

    if df.empty:
        return df

    ok_mask      = df["ok"] == True
    df["prob_gap"] = 0.0

    if ok_mask.any():
        df.loc[ok_mask, "prob_gap"] = abs(df.loc[ok_mask, "prob_up"] - 0.5)
        df = df.sort_values(
            ["ok", "tradeable", "prob_gap"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    else:
        logger.warning(
            "No trained models found. Run: python train_all.py --fetch"
        )

    return df


# ── Portfolio signal check (used by scheduler) ────────────────────────────────

def run_portfolio_signal_check(
    threshold: float = DEFAULT_SIGNAL_THRESHOLD,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    user_id: Optional[int] = None,
    account_db_id: Optional[int] = None,
    units_by_pair: Optional[dict] = None,
    default_units: Optional[int] = None,
    sl_pips: float = 20.0,
    tp_pips: float = 40.0,
    use_regime_filter: bool = True,
) -> list:
    """
    Run signal checks for all pairs and place orders for the best ones.
    Respects the global position limit across all pairs combined.
    """
    from src.paper_trader import PaperTrader

    results = []
    units_by_pair = units_by_pair or {}
    if user_id is not None:
        from src.trading_engine import get_client_for_user
        oanda = get_client_for_user(user_id, account_db_id=account_db_id)
    else:
        oanda = OandaClient()
    alerter   = Alerter()
    open_pos  = len(oanda.get_open_trades())
    signals_df = get_portfolio_signals(threshold)

    for _, row in signals_df.iterrows():
        pair = row["pair"]

        if not row["ok"]:
            logger.warning(f"{pair}: skipped — {row.get('reason','no model')}")
            continue

        if open_pos >= max_positions:
            logger.info(f"Max positions ({max_positions}) reached — stopping")
            alerter._send(
                f"ℹ️ <b>Portfolio</b>: max positions ({max_positions}) reached. "
                f"Remaining pairs skipped."
            )
            break

        is_tradeable = bool(row["tradeable"]) if use_regime_filter else bool(row.get("above_threshold"))
        if not is_tradeable:
            reason = (
                f"Regime blocked ({row['regime']})"
                if use_regime_filter and not row.get("regime_ok")
                else f"Below threshold ({threshold})"
            )
            alerter.send_signal(
                signal=row["signal"], prob_up=row["prob_up"],
                confidence=row["confidence"], price=0,
                instrument=pair, traded=False, reason=reason,
            )
            results.append({**row.to_dict(), "action": "skipped", "reason": reason})
            continue

        try:
            pt     = PaperTrader(
                instrument=pair,
                threshold=threshold,
                units=int(units_by_pair.get(pair) or default_units or PAIRS[pair]["units"]),
                use_regime_filter=False,  # already filtered above
                oanda_client=oanda,
                user_id=user_id,
                account_db_id=account_db_id,
                sl_pips=sl_pips,
                tp_pips=tp_pips,
            )
            result = pt.run_signal_check()
            if result["action"] == "order_placed":
                open_pos += 1
            results.append({**row.to_dict(), **result})
        except Exception as e:
            logger.error(f"{pair}: trade error — {e}")
            results.append({**row.to_dict(), "action": "error", "reason": str(e)})

    return results
