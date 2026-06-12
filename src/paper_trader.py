"""
Paper trading engine — multi-pair aware.

Fix: now loads the correct per-pair model from models/{PAIR}_model.pkl
instead of the legacy single-pair models/model.pkl.

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
    ):
        self.instrument     = instrument
        self.granularity    = granularity
        self.threshold      = threshold
        self.units          = units
        self.max_daily_loss = max_daily_loss
        self.max_positions  = max_positions
        self.use_regime_filter = use_regime_filter
        self.user_id        = user_id
        self.account_db_id  = account_db_id
        self.sl_pips        = sl_pips
        self.tp_pips        = tp_pips

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

            # 3. Predict
            latest      = df[feature_columns].iloc[-1:]
            prediction  = int(model.predict(latest)[0])
            prob_up     = float(model.predict_proba(latest)[0][1])
            is_buy      = prediction == 1
            signal      = "BUY" if is_buy else "SELL"
            gap         = abs(prob_up - 0.5)
            confidence  = "HIGH" if gap >= 0.15 else ("MEDIUM" if gap >= 0.08 else "LOW")
            live_price  = self.oanda.get_live_price(self.instrument)

            status.update({
                "signal": signal, "prob_up": round(prob_up, 4),
                "confidence": confidence, "price": live_price["mid"],
            })

            # 4. Log signal to CSV. DB logging happens once with final context.
            self._log_signal(timestamp, signal, prob_up, confidence, live_price["mid"])

            # 5. Threshold check
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

            # 6. Regime filter
            if self.use_regime_filter:
                regime = self.regime.detect(df)
                tradeable = self.regime.is_tradeable(regime, signal)
                # Update DB signal log with regime info
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

            # 7. Risk check (hard limits + user-configurable limits)
            risk_ok, risk_reason = self._check_risk()
            if not risk_ok:
                status["reason"] = risk_reason
                self.alerter.send_risk_alert(risk_reason)
                return status

            # 8. Place order
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
