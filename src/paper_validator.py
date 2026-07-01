"""
src/paper_validator.py — Paper Trading Validation Gate  (Task 3.3)

State machine for model trust:
  "paper_only"  → freshly trained; signals are logged but auto-trades are blocked.
  "validated"   → 30+ resolved tradeable signals with win rate ≥ 50%; auto-trades
                  are now permitted.
  "live_eligible" → reserved for future live-account gating (not yet used in v1).

Auto-promotion runs after every resolved signal outcome in PaperTrader.
Admin overrides (promote / reset) are available without waiting for the threshold.

Win-rate definition
-------------------
A signal is "win" if the price moved in the predicted direction between the
signal bar and the next bar.  Only signals flagged tradeable=1 in the DB count
toward the threshold, so regime-blocked or below-threshold signals are excluded.
"""

import json
import os
from datetime import datetime

from config.settings import meta_path
from src.logger import get_logger

logger = get_logger("paper_validator")

# ── Thresholds ─────────────────────────────────────────────────────────────────
PAPER_SIGNALS_NEEDED  = 30    # minimum resolved tradeable signals required
PAPER_WIN_RATE_NEEDED = 0.50  # minimum win rate required for auto-promotion


# ── Internal metadata helpers ─────────────────────────────────────────────────

def _load_meta(pair: str) -> dict | None:
    mep = meta_path(pair)
    if not os.path.exists(mep):
        return None
    try:
        with open(mep) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"{pair}: could not load metadata — {e}")
        return None


def _save_meta(pair: str, metadata: dict) -> None:
    mep = meta_path(pair)
    with open(mep, "w") as f:
        json.dump(metadata, f, indent=2)


# ── Public query helpers ──────────────────────────────────────────────────────

def get_model_status(pair: str) -> str:
    """
    Return the current model_status for a pair.

    Legacy metadata (trained before Task 3.3) has no model_status field.
    Those models default to "validated" so they are not silently blocked.
    Returns "no_model" when no metadata file exists.
    """
    m = _load_meta(pair)
    if m is None:
        return "no_model"
    return m.get("model_status", "validated")


def get_paper_stats(pair: str) -> dict:
    """
    Return paper validation progress for a pair.

    Returns a dict with:
      model_status      — current status string
      paper_count       — resolved signals so far
      paper_win_rate    — float 0–1 (None if no resolved signals)
      signals_needed    — PAPER_SIGNALS_NEEDED constant
      win_rate_needed   — PAPER_WIN_RATE_NEEDED constant
      validated_at      — ISO timestamp or None
      validation_override — bool
    """
    m = _load_meta(pair)
    if m is None:
        return {
            "model_status": "no_model",
            "paper_count": 0,
            "paper_win_rate": None,
            "signals_needed": PAPER_SIGNALS_NEEDED,
            "win_rate_needed": PAPER_WIN_RATE_NEEDED,
            "validated_at": None,
            "validation_override": False,
        }
    return {
        "model_status":      m.get("model_status", "validated"),
        "paper_count":       m.get("paper_signals_count", 0) or 0,
        "paper_win_rate":    m.get("paper_win_rate"),
        "signals_needed":    PAPER_SIGNALS_NEEDED,
        "win_rate_needed":   PAPER_WIN_RATE_NEEDED,
        "validated_at":      m.get("validated_at"),
        "validation_override": bool(m.get("validation_override", False)),
    }


# ── Core validation logic ─────────────────────────────────────────────────────

def check_and_promote_model(pair: str) -> bool:
    """
    Check whether a paper_only model has met the automatic promotion threshold
    and, if so, promote it to 'validated'.

    Called after every resolved signal outcome in PaperTrader.run_signal_check().

    Returns True if the model was promoted during this call, False otherwise.
    Silently no-ops for already-validated models or missing metadata.
    """
    metadata = _load_meta(pair)
    if metadata is None:
        return False

    if metadata.get("model_status") != "paper_only":
        return False  # nothing to do

    trained_at = metadata.get("trained_at", "2000-01-01T00:00:00")

    try:
        from src.database import get_paper_signal_stats
        stats = get_paper_signal_stats(pair, since_date=trained_at)
    except Exception as e:
        logger.warning(f"{pair}: could not fetch paper signal stats — {e}")
        return False

    # Always persist the latest counts so the dashboard is current
    metadata["paper_signals_count"] = stats["resolved"]
    metadata["paper_win_rate"]      = stats["win_rate"] if stats["resolved"] else None

    if stats["resolved"] < PAPER_SIGNALS_NEEDED:
        _save_meta(pair, metadata)
        logger.debug(
            f"{pair}: paper progress {stats['resolved']}/{PAPER_SIGNALS_NEEDED} signals "
            f"(win_rate={stats['win_rate']:.3f})"
        )
        return False

    if stats["win_rate"] < PAPER_WIN_RATE_NEEDED:
        _save_meta(pair, metadata)
        logger.info(
            f"{pair}: paper signal count met ({stats['resolved']}) but win rate "
            f"{stats['win_rate']:.3f} < {PAPER_WIN_RATE_NEEDED:.2f} — not promoting"
        )
        return False

    # ── Promote ────────────────────────────────────────────────────────────────
    metadata["model_status"]       = "validated"
    metadata["validated_at"]       = datetime.utcnow().isoformat()
    metadata["validation_override"] = False
    metadata["paper_signals_count"] = stats["resolved"]
    metadata["paper_win_rate"]      = stats["win_rate"]

    _save_meta(pair, metadata)

    logger.info(
        f"{pair}: ✅ Model auto-promoted to 'validated' — "
        f"{stats['resolved']} signals, win_rate={stats['win_rate']:.3f}"
    )

    try:
        from src.alerter import Alerter
        Alerter()._send(
            f"✅ <b>Model Validated — {pair.replace('_', '/')}</b>\n\n"
            f"The paper trading gate has been cleared:\n"
            f"  Signals evaluated : <b>{stats['resolved']}</b>\n"
            f"  Win rate          : <b>{stats['win_rate'] * 100:.1f}%</b>\n\n"
            f"Auto-trading is now <b>enabled</b> for this pair."
        )
    except Exception:
        pass

    return True


def promote_model_to_validated(pair: str) -> dict:
    """
    Admin override: immediately promote a model to 'validated' status,
    bypassing the automatic signal-count threshold.

    Returns a summary dict; raises FileNotFoundError if no metadata exists.
    """
    metadata = _load_meta(pair)
    if metadata is None:
        raise FileNotFoundError(
            f"No metadata found for {pair} at {meta_path(pair)}. "
            "Run training first."
        )

    previous = metadata.get("model_status", "validated")
    metadata["model_status"]        = "validated"
    metadata["validated_at"]        = datetime.utcnow().isoformat()
    metadata["validation_override"] = True

    _save_meta(pair, metadata)

    logger.warning(
        f"{pair}: model promoted to 'validated' by ADMIN OVERRIDE "
        f"(previous status: {previous})"
    )

    return {
        "pair":            pair,
        "previous_status": previous,
        "new_status":      "validated",
        "override":        True,
    }


def reset_model_to_paper_only(pair: str) -> dict:
    """
    Admin action: reset a model's validation status back to 'paper_only'.
    Clears accumulated paper counts so the gate restarts from zero.

    Returns a summary dict; raises FileNotFoundError if no metadata exists.
    """
    metadata = _load_meta(pair)
    if metadata is None:
        raise FileNotFoundError(
            f"No metadata found for {pair} at {meta_path(pair)}. "
            "Run training first."
        )

    previous = metadata.get("model_status", "validated")
    metadata["model_status"]        = "paper_only"
    metadata["validated_at"]        = None
    metadata["validation_override"] = False
    metadata["paper_signals_count"] = 0
    metadata["paper_win_rate"]      = None

    _save_meta(pair, metadata)

    logger.info(
        f"{pair}: model reset to 'paper_only' by admin "
        f"(previous status: {previous})"
    )

    return {
        "pair":            pair,
        "previous_status": previous,
        "new_status":      "paper_only",
    }
