"""
Paper trading engine — multi-pair aware.

Task 3.3 additions
──────────────────
• paper_trading_only=True  — signals are logged and outcomes resolved,
  but no order is placed.  Set by run_portfolio_signal_check() when the
  pair's model_status is "paper_only".
• _resolve_previous_signal_outcome(price) — at the start of every signal
  check, resolves the previous tradeable signal's outcome (win/loss) by
  comparing the current mid price to the entry price recorded in the DB.
  This feeds paper_validator.check_and_promote_model() which auto-promotes
  a model once it clears the 30-signal / 50%-win-rate threshold.

Security (task 1.5):
  _check_risk() calls enforce_hard_risk_limits() before the user-configurable
  checks.  This ensures platform ceilings apply to every automated order even
  when PaperTrader is invoked directly (e.g. from the portfolio scheduler).
"""

import os
import json
from datetime import datetime, date
from typing import Optional
import pandas as pd

from src.oanda_client import OandaClient
from src.features import add_features
from src.regime_detector import RegimeDetector
from src.alerter import Alerter
from src.logger import get_logger
from config.settings import (
    PAPER_TRADES_PATH, SIGNALS_LOG_PATH,
    DEFAULT_SIGNAL_THRESHOLD, DEFAULT_RISK_PER_TRADE,
    DEFAULT_MAX_DAILY_LOSS, DEFAULT_MAX_POSITIONS, DEFAULT_UNITS,
)

logger = get_logger("paper_trader")


def _load_pair_model(pair: str):
    """Load the correct model for a given pair. Falls back to legacy model."""
    import joblib, json as _json
    from config.settings import model_path, meta_path, MODEL_PATH, METADATA_PATH

    mp  = model_path(pair)
    mep = meta_path(pair)

    # Prefer pair-specific model; fall back to legacy
    if not os.path.exists(mp):
        mp  = MODEL_PATH
        mep = METADATA_PATH

    if not os.path.exists(mp):
        raise FileNotFoundError(
            f"No model found for {pair}. Run: python train_all.py --fetch"
        )

    model = joblib.load(mp)
    with open(mep) as f:
        metadata = _json.load(f)
    return model, metadata


class PaperTrader:
    def __init__(
        self,
        instrument: str = "EUR_USD",
        granularity: str = "D",
        threshold: float = DEFAULT_SIGNAL_THRESHOLD,
        units: int = DEFAULT_UNITS,
        max_daily_loss: float = DEFAULT_MAX_DAILY_LOSS,
        max_positions: int = DEFAULT_MAX_POSITIONS,
        use_regime_filter: bool = True,
        oanda_client: Optional[OandaClient] = None,
        user_id: Optional[int] = None,
        account_db_id: Optional[int] = None,
        sl_pips: float = 20.0,
        tp_pips: float = 40.0,
        # ── Task 3.3 ──────────────────────────────────────────────────────────
        paper_trading_only: bool = False,
        # ─────────────────────────────────────────────────────────────────────
    ):
        self.instrument        = instrument
        self.granularity       = granularity
        self.threshold         = threshold
        self.units             = units
        self.max_daily_loss    = max_daily_loss
        self.max_positions     = max_positions
        self.use_regime_filter = use_regime_filter
        self.user_id           = user_id
        self.account_db_id     = account_db_id
        self.sl_pips           = sl_pips
        self.tp_pips           = tp_pips
        self.paper_trading_only = paper_trading_only   # ← Task 3.3

        if oanda_client is not None:
            self.oanda = oanda_client
        elif user_id is not None:
            from src.trading_engine import get_client_for_user
            self.oanda = get_client_for_user(user_id, account_db_id=account_db_id)
        else:
            self.oanda = OandaClient()
        self.alerter = Alerter()
        self.regime  = RegimeDetector()

        os.makedirs(os.path.dirname(PAPER_TRADES_PATH) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(SIGNALS_LOG_PATH)  or ".", exist_ok=True)

    # ── Task 3.3: outcome resolution ──────────────────────────────────────────

    def _resolve_previous_signal_outcome(self, current_price: float) -> None:
        """
        Resolve the outcome of the most recent unresolved tradeable signal for
        this instrument.

        Logic:
          - Fetch the latest DB signal with tradeable=1 and outcome IS NULL.
          - Skip if it is less than 4 hours old (same-bar protection for
            intraday granularities).
          - Compare current_price to the signal's entry price:
              BUY signal + price rose  → WIN  (1)
              BUY signal + price fell  → LOSS (0)
              SELL signal + price fell → WIN  (1)
              SELL signal + price rose → LOSS (0)
          - Write the outcome back to signals_log.
          - Call check_and_promote_model() so the model auto-promotes if the
            threshold is now met.

        Failures are logged as warnings and never propagate to the caller.
        """
        try:
            from src.database import (
                get_latest_unresolved_signal,
                resolve_signal_outcome,
            )
            from src.paper_validator import check_and_promote_model

            prev = get_latest_unresolved_signal(self.instrument)
            if not prev:
                return

            # Age check — wait at least 4 h before resolving (daily candles
            # are available ~1 h after UTC midnight; H4 candles close every 4 h)
            try:
                signal_time = datetime.fromisoformat(str(prev["created_at"]))
            except Exception:
                signal_time = datetime.utcnow()

            age_hours = (datetime.utcnow() - signal_time).total_seconds() / 3600
            if age_hours < 4:
                return

            entry_price = float(prev["price"])
            if entry_price <= 0:
                return

            price_moved_up = current_price > entry_price
            if prev["signal"] == "BUY":
                outcome = 1 if price_moved_up else 0
            else:  # SELL
                outcome = 1 if not price_moved_up else 0

            resolve_signal_outcome(int(prev["id"]), outcome, current_price)

            logger.debug(
                f"{self.instrument}: resolved signal id={prev['id']} "
                f"({prev['signal']} @ {entry_price:.5f}) → "
                f"{'WIN' if outcome else 'LOSS'} (exit @ {current_price:.5f})"
            )

            # Check if accumulated paper stats now meet the promotion threshold
            check_and_promote_model(self.instrument)

        except Exception as e:
            logger.warning(
                f"{self.instrument}: could not resolve previous signal outcome — {e}"
            )

    # ── Main signal check loop ────────────────────────────────────────────────

    def run_signal_check(self) -> dict:
        timestamp = datetime.utcnow().isoformat()
        status    = {"timestamp": timestamp, "action": "none", "reason": ""}

        try:
            # 1. Load pair-specific model
            model, metadata = _load_pair_model(self.instrument)
            feature_columns = metadata["feature_columns"]

            # 2. Fetch candles from Oanda
            df_raw = self.oanda.get_candles(
                instrument=self.instrument,
                granularity=self.granularity,
                count=300,
            )
            df = add_features(df_raw)

            # 3. Get live price (moved up so we can pass it to outcome resolution)
            live_price = self.oanda.get_live_price(self.instrument)

            # 3a. Task 3.3 — resolve previous signal outcome and check promotion
            self._resolve_previous_signal_outcome(live_price["mid"])

            # 4. Predict
            latest      = df[feature_columns].iloc[-1:]
            prediction  = int(model.predict(latest)[0])
            prob_up     = float(model.predict_proba(latest)[0][1])
            is_buy      = prediction == 1
            signal      = "BUY" if is_buy else "SELL"
            gap         = abs(prob_up - 0.5)
            confidence  = "HIGH" if gap >= 0.15 else ("MEDIUM" if gap >= 0.08 else "LOW")

            status.update({
                "signal": signal, "prob_up": round(prob_up, 4),
                "confidence": confidence, "price": live_price["mid"],
            })

            # 5. Log signal to CSV
            self._log_signal(timestamp, signal, prob_up, confidence, live_price["mid"])

            # 6. Threshold check
            above = prob_up >= self.threshold if is_buy else (1 - prob_up) >= self.threshold
            if not above:
                reason = f"Confidence below threshold ({self.threshold})"
                status["reason"] = reason
                try:
                    from src.database import log_signal
                    log_signal(self.instrument, signal, prob_up, confidence,
                               "unknown", False, live_price["mid"])
                except Exception:
                    pass
                self.alerter.send_signal(
                    signal=signal, prob_up=prob_up, confidence=confidence,
                    price=live_price["mid"], instrument=self.instrument,
                    traded=False, reason=reason,
                )
                return status

            # 7. Regime filter
            if self.use_regime_filter:
                regime = self.regime.detect(df)
                tradeable = self.regime.is_tradeable(regime, signal)
                try:
                    from src.database import log_signal
                    log_signal(self.instrument, signal, prob_up, confidence,
                               regime["adx_regime"], tradeable, live_price["mid"])
                except Exception:
                    pass
                if not tradeable:
                    reason = f"Regime filter blocked: {regime['adx_regime']}"
                    status["reason"] = reason
                    self.alerter.send_signal(
                        signal=signal, prob_up=prob_up, confidence=confidence,
                        price=live_price["mid"], instrument=self.instrument,
                        traded=False, reason=reason,
                    )
                    return status
            else:
                try:
                    from src.database import log_signal
                    log_signal(self.instrument, signal, prob_up, confidence,
                               "not_applied", True, live_price["mid"])
                except Exception:
                    pass

            # 7b. Task 3.3 — paper-only gate ──────────────────────────────────
            # At this point the signal is above threshold and passes the regime
            # filter (tradeable=True was logged above).  If the model is still
            # in paper_only mode we stop here — the signal is already persisted
            # so it will count toward the promotion threshold once resolved.
            if self.paper_trading_only:
                reason = (
                    "Model in paper-only validation mode — "
                    f"accumulating signals ({metadata.get('paper_signals_count', 0) or 0}"
                    f"/{30} needed). No order placed."
                )
                status["reason"] = reason
                status["action"] = "paper_signal"
                self.alerter.send_signal(
                    signal=signal, prob_up=prob_up, confidence=confidence,
                    price=live_price["mid"], instrument=self.instrument,
                    traded=False, reason=reason,
                )
                return status
            # ─────────────────────────────────────────────────────────────────

            # 8. Risk check (hard limits + user-configurable limits)
            risk_ok, risk_reason = self._check_risk()
            if not risk_ok:
                status["reason"] = risk_reason
                self.alerter.send_risk_alert(risk_reason)
                return status

            # 9. Place order
            units = self.units if is_buy else -self.units
            mid   = live_price["mid"]
            from src.trading_engine import calculate_sl_tp
            sl_price, tp_price = calculate_sl_tp(
                self.instrument, signal, mid, self.sl_pips, self.tp_pips
            )

            fill = self.oanda.place_market_order(
                instrument=self.instrument,
                units=units,
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
            )

            status.update({"action": "order_placed", "fill": fill})
            self._log_trade(timestamp, signal, prob_up, confidence, fill)
            if self.user_id is not None:
                try:
                    from src.database import log_trade
                    log_trade(
                        user_id=self.user_id,
                        pair=self.instrument,
                        signal=signal,
                        entry_price=fill.get("fill_price") or mid,
                        units=abs(int(fill.get("units") or self.units)),
                        trade_type="auto",
                        broker_trade_id=fill.get("trade_id", ""),
                    )
                except Exception as e:
                    logger.warning(f"DB trade log failed: {e}")
            self.alerter.send_signal(
                signal=signal, prob_up=prob_up, confidence=confidence,
                price=fill["fill_price"], instrument=self.instrument,
                traded=True, reason="",
                trade_id=fill.get("trade_id"), sl=sl_price, tp=tp_price,
            )
            logger.info(f"Order placed: {fill}")

        except Exception as e:
            status["reason"] = str(e)
            status["action"]  = "error"
            logger.error(f"Signal check failed: {e}")
            self.alerter.send_error(str(e))

        return status

    def _check_risk(self) -> tuple:
        """
        Two-layer risk check:

        Layer 1 — Hard platform ceilings (enforce_hard_risk_limits).
          These are non-negotiable and cannot be overridden by any user
          setting.  If they fire, the trade is blocked and auto-trading
          may be halted for the day (kill switch).

        Layer 2 — User-configurable per-account limits.
          max_positions, max_daily_loss, duplicate-position guard.
          These are soft limits that users and admins can adjust within
          the bounds of the hard ceilings set by Layer 1.
        """
        # ── Layer 1: Hard limits ──────────────────────────────────────────────
        try:
            from src.trading_engine import enforce_hard_risk_limits
            enforce_hard_risk_limits(
                client=self.oanda,
                user_id=self.user_id,
                units=self.units,
                instrument=self.instrument,
                sl_pips=self.sl_pips,
            )
        except ValueError as exc:
            return False, str(exc)

        # ── Layer 2: User-configurable limits ─────────────────────────────────
        try:
            summary = self.oanda.get_account_summary()
            if summary["open_trades"] >= self.max_positions:
                return False, f"Max positions ({self.max_positions}) reached"
            if summary["balance"] > 0:
                pnl_pct = summary["unrealized_pl"] / summary["balance"]
                if pnl_pct <= -self.max_daily_loss:
                    return False, f"Daily loss limit hit ({pnl_pct*100:.2f}%)"
            open_trades = self.oanda.get_open_trades()
            if any(t["instrument"] == self.instrument for t in open_trades):
                return False, f"Already have open {self.instrument} position"
        except Exception as e:
            return False, f"Risk check error: {e}"
        return True, ""

    def _log_signal(self, timestamp, signal, prob_up, confidence, price):
        row = pd.DataFrame([{
            "timestamp": timestamp, "instrument": self.instrument,
            "signal": signal, "prob_up": prob_up,
            "confidence": confidence, "price": price,
        }])
        if os.path.exists(SIGNALS_LOG_PATH):
            row.to_csv(SIGNALS_LOG_PATH, mode="a", header=False, index=False)
        else:
            row.to_csv(SIGNALS_LOG_PATH, index=False)

    def _log_trade(self, timestamp, signal, prob_up, confidence, fill):
        trades = self._load_trades()
        trades.append({
            "timestamp": timestamp, "instrument": self.instrument,
            "signal": signal, "prob_up": prob_up,
            "confidence": confidence,
            "fill_price": fill.get("fill_price"),
            "units": fill.get("units"),
            "trade_id": fill.get("trade_id"),
        })
        with open(PAPER_TRADES_PATH, "w") as f:
            json.dump(trades, f, indent=2)

    def _load_trades(self) -> list:
        if os.path.exists(PAPER_TRADES_PATH):
            try:
                with open(PAPER_TRADES_PATH) as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def get_paper_trade_summary(self) -> dict:
        trades = self._load_trades()
        if not trades:
            return {"total_trades": 0, "message": "No paper trades yet"}
        df = pd.DataFrame(trades)
        return {
            "total_trades": len(df),
            "buy_trades":   int((df["signal"] == "BUY").sum()),
            "sell_trades":  int((df["signal"] == "SELL").sum()),
            "first_trade":  df["timestamp"].iloc[0],
            "last_trade":   df["timestamp"].iloc[-1],
            "high_conf":    int((df["confidence"] == "HIGH").sum()),
        }

    def get_signal_log(self, n: int = 50) -> pd.DataFrame:
        if not os.path.exists(SIGNALS_LOG_PATH):
            return pd.DataFrame()
        return pd.read_csv(SIGNALS_LOG_PATH).tail(n)
