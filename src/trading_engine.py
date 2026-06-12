"""
ForexChautari — Trading Engine
Central place for all trading logic used by both user dashboard and admin panel.
Handles: build client from user's stored account, risk checks, order placement,
position sizing, SL/TP calculation.

Security (task 1.5):
  enforce_hard_risk_limits() is called before every order placement.
  It enforces three ceilings that no user setting can override:
    • HARD_MAX_POSITIONS      — absolute open-position cap
    • HARD_MAX_RISK_PCT       — max balance fraction at risk per trade
    • HARD_MAX_DAILY_LOSS_PCT — daily kill-switch (disables auto-trading for
                                the rest of the UTC day and sends a Telegram alert)
"""

import os
from datetime import date as _date
from typing import Optional, TYPE_CHECKING

from src.oanda_client import OandaClient
from src.database import (
    get_trading_accounts, log_trade, close_trade as db_close_trade,
    get_user_trading_settings,
    get_platform_settings, update_platform_settings,
)
from src.logger import get_logger
from config.settings import (
    HARD_MAX_POSITIONS,
    HARD_MAX_RISK_PCT,
    HARD_MAX_DAILY_LOSS_PCT,
)

logger = get_logger("trading_engine")


# ── Kill-switch helpers ───────────────────────────────────────────────────────

def _killswitch_key(user_id: int) -> str:
    """platform_settings key for a user's daily kill-switch date."""
    return f"killswitch_user_{user_id}"


def is_user_killed_today(user_id: int) -> bool:
    """
    Return True if the daily kill switch fired for this user today (UTC).
    The switch resets automatically at midnight UTC — no cleanup needed.
    """
    settings = get_platform_settings()
    stored = settings.get(_killswitch_key(user_id), "")
    return stored == str(_date.today())


def _activate_killswitch(user_id: int, drawdown_pct: float) -> None:
    """
    Persist today's date as the kill-switch value for this user and send
    a Telegram alert.  Idempotent — safe to call multiple times in the
    same UTC day.
    """
    update_platform_settings({_killswitch_key(user_id): str(_date.today())})

    logger.critical(
        f"KILL SWITCH ACTIVATED — user_id={user_id} "
        f"daily drawdown={drawdown_pct * 100:.2f}% >= "
        f"{HARD_MAX_DAILY_LOSS_PCT * 100:.0f}% hard limit. "
        "Auto-trading disabled for the rest of the UTC day."
    )

    try:
        from src.alerter import Alerter
        Alerter()._send(
            f"🚨 <b>DAILY KILL SWITCH ACTIVATED</b>\n\n"
            f"User ID: <code>{user_id}</code>\n"
            f"Unrealised drawdown: <b>{drawdown_pct * 100:.2f}%</b> has breached the\n"
            f"hard limit of <b>{HARD_MAX_DAILY_LOSS_PCT * 100:.0f}%</b>.\n\n"
            f"Auto-trading is <b>disabled for the rest of today (UTC)</b>.\n"
            f"It will resume automatically tomorrow.\n\n"
            f"No further orders will be placed for this account today."
        )
    except Exception as exc:
        logger.warning(f"Kill-switch Telegram alert failed: {exc}")


# ── Hard risk enforcement ─────────────────────────────────────────────────────

def enforce_hard_risk_limits(
    client: OandaClient,
    user_id: Optional[int],
    units: int,
    instrument: str,
    sl_pips: float,
) -> None:
    """
    Enforce platform-level hard risk ceilings before any order is placed.

    This function is intentionally separate from the user-configurable
    _check_risk() in PaperTrader so that it cannot be disabled, bypassed,
    or misconfigured through any user-facing setting.

    Checks (in order):

    1. Daily kill switch — was this user shut down earlier today?
    2. Hard position ceiling — HARD_MAX_POSITIONS open at once.
    3. Daily drawdown kill switch — if unrealised P&L / balance is worse
       than -HARD_MAX_DAILY_LOSS_PCT, activate the kill switch and block.
    4. Per-trade risk ceiling — units × sl_pips × pip_value / balance
       must not exceed HARD_MAX_RISK_PCT.

    Raises ValueError with a clear message on any breach.
    Logs a WARNING and proceeds if the account summary is temporarily
    unavailable (checks 3 & 4 only — kill switch and position cap always run).
    """

    # ── 1. Kill switch ────────────────────────────────────────────────────────
    if user_id is not None and is_user_killed_today(user_id):
        raise ValueError(
            f"Daily kill switch is active for user {user_id}. "
            f"Auto-trading is disabled for the rest of today (UTC). "
            f"It will resume tomorrow automatically."
        )

    # ── 2. Hard position ceiling ──────────────────────────────────────────────
    open_trades = client.get_open_trades()
    n_open = len(open_trades)
    if n_open >= HARD_MAX_POSITIONS:
        raise ValueError(
            f"Hard position ceiling reached: {n_open} open "
            f"(hard limit {HARD_MAX_POSITIONS}). "
            f"Close an existing position before placing a new order."
        )

    # ── 3 & 4. Account-level checks ───────────────────────────────────────────
    # These require an API call; if Oanda is temporarily unreachable we log a
    # warning rather than blocking trades — the position cap (check 2) and kill
    # switch (check 1) are the primary safety net.
    try:
        summary = client.get_account_summary()
        balance = float(summary["balance"])

        if balance > 0:
            # ── 3. Daily drawdown → kill switch ───────────────────────────────
            unrealised_pl  = float(summary["unrealized_pl"])
            drawdown_pct   = unrealised_pl / balance  # negative when losing

            if drawdown_pct <= -HARD_MAX_DAILY_LOSS_PCT:
                if user_id is not None:
                    _activate_killswitch(user_id, abs(drawdown_pct))
                raise ValueError(
                    f"Daily loss hard limit breached: "
                    f"{abs(drawdown_pct) * 100:.2f}% unrealised drawdown "
                    f"(hard limit {HARD_MAX_DAILY_LOSS_PCT * 100:.0f}%). "
                    f"Auto-trading disabled for the rest of today."
                )

            # ── 4. Per-trade risk ceiling ──────────────────────────────────────
            pip_value      = 0.01 if "JPY" in instrument else 0.0001
            trade_risk_pct = (units * sl_pips * pip_value) / balance

            if trade_risk_pct > HARD_MAX_RISK_PCT:
                raise ValueError(
                    f"Trade risk {trade_risk_pct * 100:.2f}% exceeds hard cap "
                    f"{HARD_MAX_RISK_PCT * 100:.0f}%. "
                    f"Reduce units ({units:,}) or stop-loss ({sl_pips} pips) "
                    f"so the position risks at most "
                    f"${balance * HARD_MAX_RISK_PCT:,.2f} "
                    f"({HARD_MAX_RISK_PCT * 100:.0f}% of ${balance:,.2f})."
                )

    except ValueError:
        raise  # re-raise our own limit-breach errors untouched
    except Exception as exc:
        # Oanda API temporarily down — position cap and kill switch still ran
        logger.warning(
            f"enforce_hard_risk_limits: account summary unavailable "
            f"({exc}); checks 3 & 4 skipped for this order."
        )


# ── Internal account helpers ──────────────────────────────────────────────────

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

    Note: risk_pct is silently clamped to HARD_MAX_RISK_PCT so that
    risk-sizing helpers can never produce a position that would be
    rejected by enforce_hard_risk_limits().
    """
    # Clamp the requested risk to the hard ceiling
    effective_risk_pct = min(risk_pct, HARD_MAX_RISK_PCT)
    if effective_risk_pct < risk_pct:
        logger.warning(
            f"calculate_position_size: requested risk_pct {risk_pct:.3f} "
            f"exceeds hard cap {HARD_MAX_RISK_PCT:.3f} — clamped."
        )

    if stop_loss_pips <= 0:
        return min_units
    risk_amount = balance * effective_risk_pct
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
    Full trade placement: get client → enforce hard risk limits →
    get live price → calculate SL/TP → place → log to DB.

    Hard risk limits are enforced here unconditionally, before any
    order reaches Oanda.  This is the primary enforcement point for
    trades placed through the Trading tab (manual and limit orders).

    Raises ValueError if any hard limit is breached.
    """
    client = get_client_for_user(user_id, account_idx, account_db_id)

    # ── Hard risk limits (cannot be bypassed) ─────────────────────────────────
    enforce_hard_risk_limits(
        client=client,
        user_id=user_id,
        units=units,
        instrument=instrument,
        sl_pips=sl_pips,
    )

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

    logger.info(
        f"Trade placed: user={user_id} {instrument} {direction} "
        f"units={units} fill={fill['fill_price']}"
    )
    return {
        "success":     True,
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
            # Expose kill-switch state so dashboards can surface it
            "kill_switch_active": is_user_killed_today(user_id) if user_id else False,
            "hard_max_positions": HARD_MAX_POSITIONS,
            "hard_max_risk_pct":  HARD_MAX_RISK_PCT,
            "hard_max_daily_loss_pct": HARD_MAX_DAILY_LOSS_PCT,
        }
    except Exception as e:
        return {"error": str(e)}


def run_user_auto_trade(user_id: int, settings: Optional[dict] = None) -> list:
    """Run one portfolio auto-trade cycle for a user's own linked account."""
    if settings is None:
        settings = get_user_trading_settings(user_id)

    if settings.get("mode") != "auto" or not settings.get("auto_trade_enabled"):
        return [{"action": "skipped", "reason": "Auto-trade is not enabled"}]

    # Kill switch is also checked inside enforce_hard_risk_limits() for each
    # individual pair, but do a fast early-exit here to skip the whole cycle.
    if is_user_killed_today(user_id):
        return [{
            "action": "skipped",
            "reason": (
                "Daily kill switch is active — auto-trading is disabled "
                "for the rest of today (UTC)."
            ),
        }]

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
