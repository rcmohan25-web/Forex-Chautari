"""
ForexChautari — Trading Engine
Central place for all trading logic used by both user dashboard and admin panel.
Handles: build client from user's stored account, risk checks, order placement,
position sizing, SL/TP calculation.
"""

import os
from typing import Optional
from src.oanda_client import OandaClient
from src.database import (
    get_trading_accounts, log_trade, close_trade as db_close_trade,
    get_user_trading_settings,
)
from src.logger import get_logger

logger = get_logger("trading_engine")


def _select_account(accounts: list, account_idx: int = 0, account_db_id: int | None = None) -> dict:
    if account_db_id:
        match = next((acc for acc in accounts if int(acc["id"]) == int(account_db_id)), None)
        if match:
            return match
    if account_idx >= len(accounts):
        account_idx = 0
    return accounts[account_idx]


def get_client_for_user(
    user_id: int,
    account_idx: int = 0,
    account_db_id: int | None = None,
) -> OandaClient:
    """
    Build an OandaClient using the user's stored trading account credentials.
    Raises ValueError if no account is linked.
    """
    accounts = get_trading_accounts(user_id)
    if not accounts:
        raise ValueError(
            "No trading account connected. "
            "Go to Auto-Trade → Connect Account to add your Oanda credentials."
        )
    acc = _select_account(accounts, account_idx, account_db_id)
    return OandaClient(
        api_key=acc["api_key_enc"],
        account_id=acc["account_id"],
        environment=acc["environment"],
    )


def calculate_position_size(
    balance:        float,
    risk_pct:       float,
    stop_loss_pips: float,
    pip_value:      float = 0.0001,
    min_units:      int   = 100,
    max_units:      int   = 100000,
) -> int:
    """
    Calculate units based on risk percentage of balance.
    risk_pct: e.g. 0.01 = 1%
    stop_loss_pips: distance to stop in pips
    """
    if stop_loss_pips <= 0:
        return min_units
    risk_amount = balance * risk_pct
    units = int(risk_amount / (stop_loss_pips * pip_value))
    return max(min_units, min(units, max_units))


def calculate_sl_tp(
    instrument:   str,
    direction:    str,   # "BUY" or "SELL"
    entry_price:  float,
    sl_pips:      float = 20.0,
    tp_pips:      float = 40.0,
) -> tuple:
    """
    Calculate stop loss and take profit prices.
    Returns (sl_price, tp_price)
    """
    pip = 0.01 if "JPY" in instrument else 0.0001
    sl_dist = sl_pips * pip
    tp_dist = tp_pips * pip

    if direction == "BUY":
        sl = round(entry_price - sl_dist, 5)
        tp = round(entry_price + tp_dist, 5)
    else:
        sl = round(entry_price + sl_dist, 5)
        tp = round(entry_price - tp_dist, 5)
    return sl, tp


def place_trade(
    user_id:    int,
    instrument: str,
    direction:  str,
    units:      int,
    sl_pips:    float = 20.0,
    tp_pips:    float = 40.0,
    trade_type: str   = "manual",
    account_idx: int  = 0,
    account_db_id: Optional[int] = None,
) -> dict:
    """
    Full trade placement: get client → get live price → calculate SL/TP → place → log to DB.
    Returns result dict with fill details.
    """
    client = get_client_for_user(user_id, account_idx, account_db_id)
    price_data = client.get_live_price(instrument)
    entry = price_data["mid"]

    sl, tp = calculate_sl_tp(instrument, direction, entry, sl_pips, tp_pips)
    actual_units = units if direction == "BUY" else -units

    fill = client.place_market_order(
        instrument=instrument,
        units=actual_units,
        stop_loss_price=sl,
        take_profit_price=tp,
    )

    # Log to DB
    trade_id = log_trade(
        user_id=user_id,
        pair=instrument,
        signal=direction,
        entry_price=fill["fill_price"] or entry,
        units=units,
        trade_type=trade_type,
        broker_trade_id=fill.get("trade_id", ""),
    )

    logger.info(f"Trade placed: user={user_id} {instrument} {direction} units={units} fill={fill['fill_price']}")
    return {
        "success":    True,
        "db_trade_id": trade_id,
        "fill":        fill,
        "sl":          sl,
        "tp":          tp,
        "entry":       fill["fill_price"] or entry,
    }


def close_user_trade(
    user_id: int,
    broker_trade_id: str,
    db_trade_id: int,
    account_idx: int = 0,
    account_db_id: Optional[int] = None,
) -> dict:
    """Close a specific trade and update the DB record."""
    client = get_client_for_user(user_id, account_idx, account_db_id)
    result = client.close_trade(broker_trade_id)
    db_close_trade(db_trade_id, result["fill_price"], result["pl"])
    return result


def get_risk_metrics(
    user_id: int,
    account_idx: int = 0,
    account_db_id: Optional[int] = None,
) -> dict:
    """Return live risk metrics for a user's account."""
    try:
        client  = get_client_for_user(user_id, account_idx, account_db_id)
        summary = client.get_account_summary()
        trades  = client.get_open_trades()
        total_unrealized = sum(t["unrealized_pl"] for t in trades)
        return {
            "balance":           summary["balance"],
            "nav":               summary["nav"],
            "unrealized_pl":     summary["unrealized_pl"],
            "realized_pl":       summary["realized_pl"],
            "margin_used":       summary["margin_used"],
            "margin_available":  summary["margin_avail"],
            "open_trades":       len(trades),
            "daily_pnl":         client.get_daily_pnl(),
            "risk_pct":          round(abs(total_unrealized) / summary["balance"] * 100, 2)
                                 if summary["balance"] > 0 else 0,
        }
    except Exception as e:
        return {"error": str(e)}


def run_user_auto_trade(user_id: int, settings: Optional[dict] = None) -> list:
    """Run one portfolio auto-trade cycle for a user's own linked account."""
    if settings is None:
        settings = get_user_trading_settings(user_id)

    if settings.get("mode") != "auto" or not settings.get("auto_trade_enabled"):
        return [{"action": "skipped", "reason": "Auto-trade is not enabled"}]

    from src.multi_pair_manager import run_portfolio_signal_check

    units = int(settings.get("units") or 1000)
    return run_portfolio_signal_check(
        threshold=float(settings.get("threshold") or 0.55),
        max_positions=int(settings.get("max_positions") or 3),
        user_id=user_id,
        account_db_id=settings.get("trading_account_id"),
        units_by_pair={},
        default_units=units,
        sl_pips=float(settings.get("sl_pips") or 20),
        tp_pips=float(settings.get("tp_pips") or 40),
        use_regime_filter=bool(settings.get("use_regime_filter", True)),
    )
