"""
tests/test_hard_risk_limits.py
==============================
Tests for task 1.5 — position-level hard risk limits.

Covers:
  • enforce_hard_risk_limits() — all four checks
  • is_user_killed_today() / _activate_killswitch()
  • calculate_position_size() risk_pct clamping
  • run_user_auto_trade() early-exit on active kill switch
  • place_trade() enforcement via mock client
  • PaperTrader._check_risk() Layer 1 enforcement

Run:
    python tests/test_hard_risk_limits.py
"""

import os
import sys
import tempfile
import unittest
import unittest.mock as mock
from datetime import date
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_oanda(
    balance=10000.0,
    unrealized_pl=0.0,
    open_trade_count=0,
    open_trades=None,
):
    """Return a MagicMock OandaClient with predictable responses."""
    client = MagicMock()
    client.get_account_summary.return_value = {
        "balance":       balance,
        "nav":           balance + unrealized_pl,
        "unrealized_pl": unrealized_pl,
        "realized_pl":   0.0,
        "open_trades":   open_trade_count,
        "margin_used":   0.0,
        "margin_avail":  balance,
        "currency":      "USD",
        "leverage":      "50",
    }
    client.get_open_trades.return_value = open_trades or []
    return client


def _temp_db():
    """Set DATABASE_URL to a fresh temp file and init it."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    
    # Save original DATABASE_URL
    orig = os.environ.get("DATABASE_URL")
    
    # Set to temp database
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    
    # Reset engine to use new DATABASE_URL
    from src.db_engine import reset_engine
    reset_engine()
    
    from src.database import init_db
    init_db()
    
    return orig


def _restore_db(orig):
    # Restore original DATABASE_URL
    if orig is not None:
        os.environ["DATABASE_URL"] = orig
    else:
        os.environ.pop("DATABASE_URL", None)
    
    # Reset engine to restore original connection
    from src.db_engine import reset_engine
    reset_engine()


# ══════════════════════════════════════════════════════════════════════════════
# TestHardRiskConstants
# ══════════════════════════════════════════════════════════════════════════════

class TestHardRiskConstants(unittest.TestCase):
    """Verify the three hard ceilings exist and have sensible values."""

    def test_constants_exist(self):
        from config.settings import (
            HARD_MAX_POSITIONS,
            HARD_MAX_RISK_PCT,
            HARD_MAX_DAILY_LOSS_PCT,
        )
        self.assertIsInstance(HARD_MAX_POSITIONS,      int)
        self.assertIsInstance(HARD_MAX_RISK_PCT,       float)
        self.assertIsInstance(HARD_MAX_DAILY_LOSS_PCT, float)

    def test_constants_reasonable(self):
        from config.settings import (
            HARD_MAX_POSITIONS,
            HARD_MAX_RISK_PCT,
            HARD_MAX_DAILY_LOSS_PCT,
        )
        self.assertGreater(HARD_MAX_POSITIONS,      0)
        self.assertLessEqual(HARD_MAX_POSITIONS,    20)
        self.assertGreater(HARD_MAX_RISK_PCT,       0)
        self.assertLessEqual(HARD_MAX_RISK_PCT,     0.10)  # ≤ 10%
        self.assertGreater(HARD_MAX_DAILY_LOSS_PCT, 0)
        self.assertLessEqual(HARD_MAX_DAILY_LOSS_PCT, 0.20)  # ≤ 20%

    def test_hard_limits_tighter_than_defaults(self):
        """Hard ceilings must be at least as tight as the user-configurable defaults."""
        from config.settings import (
            HARD_MAX_POSITIONS,
            HARD_MAX_RISK_PCT,
            HARD_MAX_DAILY_LOSS_PCT,
            DEFAULT_MAX_POSITIONS,
            DEFAULT_RISK_PER_TRADE,
            DEFAULT_MAX_DAILY_LOSS,
        )
        self.assertGreaterEqual(HARD_MAX_POSITIONS,      DEFAULT_MAX_POSITIONS)
        self.assertGreaterEqual(HARD_MAX_RISK_PCT,       DEFAULT_RISK_PER_TRADE)
        self.assertGreaterEqual(HARD_MAX_DAILY_LOSS_PCT, DEFAULT_MAX_DAILY_LOSS)


# ══════════════════════════════════════════════════════════════════════════════
# TestKillSwitch
# ══════════════════════════════════════════════════════════════════════════════

class TestKillSwitch(unittest.TestCase):

    def setUp(self):
        self._orig = _temp_db()

    def tearDown(self):
        _restore_db(self._orig)

    def test_not_killed_by_default(self):
        from src.trading_engine import is_user_killed_today
        self.assertFalse(is_user_killed_today(1))

    def test_activate_sets_flag(self):
        from src.trading_engine import _activate_killswitch, is_user_killed_today
        with patch("src.alerter.Alerter"):
            _activate_killswitch(user_id=1, drawdown_pct=0.06)
        self.assertTrue(is_user_killed_today(1))

    def test_different_user_not_affected(self):
        from src.trading_engine import _activate_killswitch, is_user_killed_today
        with patch("src.alerter.Alerter"):
            _activate_killswitch(user_id=1, drawdown_pct=0.06)
        self.assertFalse(is_user_killed_today(2), "Kill switch must be per-user")

    def test_activate_is_idempotent(self):
        """Calling activate twice in the same day should not raise."""
        from src.trading_engine import _activate_killswitch, is_user_killed_today
        with patch("src.alerter.Alerter"):
            _activate_killswitch(user_id=99, drawdown_pct=0.06)
            _activate_killswitch(user_id=99, drawdown_pct=0.07)
        self.assertTrue(is_user_killed_today(99))

    def test_kill_switch_resets_tomorrow(self):
        """Kill switch stored for yesterday should not block today."""
        from src.database import update_platform_settings
        from src.trading_engine import is_user_killed_today, _killswitch_key
        yesterday = str(date.fromordinal(date.today().toordinal() - 1))
        update_platform_settings({_killswitch_key(5): yesterday})
        self.assertFalse(is_user_killed_today(5), "Yesterday's kill switch must not block today")

    def test_telegram_alert_sent_on_activation(self):
        from src.trading_engine import _activate_killswitch
        with patch("src.alerter.Alerter") as MockAlerter:
            instance = MockAlerter.return_value
            _activate_killswitch(user_id=7, drawdown_pct=0.07)
        instance._send.assert_called_once()
        call_args = instance._send.call_args[0][0]
        self.assertIn("KILL SWITCH", call_args)
        self.assertIn("7", call_args)

    def test_telegram_failure_does_not_raise(self):
        """A Telegram outage must not propagate as an exception."""
        from src.trading_engine import _activate_killswitch
        with patch("src.alerter.Alerter") as MockAlerter:
            MockAlerter.return_value._send.side_effect = RuntimeError("network")
            # Should not raise
            _activate_killswitch(user_id=8, drawdown_pct=0.06)


# ══════════════════════════════════════════════════════════════════════════════
# TestEnforceHardRiskLimits
# ══════════════════════════════════════════════════════════════════════════════

class TestEnforceHardRiskLimits(unittest.TestCase):

    def setUp(self):
        self._orig = _temp_db()

    def tearDown(self):
        _restore_db(self._orig)

    def _call(self, client, user_id=None, units=1000,
              instrument="EUR_USD", sl_pips=20.0):
        from src.trading_engine import enforce_hard_risk_limits
        enforce_hard_risk_limits(
            client=client,
            user_id=user_id,
            units=units,
            instrument=instrument,
            sl_pips=sl_pips,
        )

    # ── Check 1: kill switch ──────────────────────────────────────────────────

    def test_kill_switch_blocks_order(self):
        from src.trading_engine import _activate_killswitch
        with patch("src.alerter.Alerter"):
            _activate_killswitch(user_id=1, drawdown_pct=0.06)
        client = _mock_oanda()
        with self.assertRaises(ValueError) as ctx:
            self._call(client, user_id=1)
        self.assertIn("kill switch", str(ctx.exception).lower())

    def test_kill_switch_only_blocks_affected_user(self):
        """A different user's kill switch must not block user 2."""
        from src.trading_engine import _activate_killswitch
        with patch("src.alerter.Alerter"):
            _activate_killswitch(user_id=1, drawdown_pct=0.06)
        client = _mock_oanda()
        # Should not raise for user 2
        self._call(client, user_id=2)

    def test_no_user_id_bypasses_kill_switch(self):
        """user_id=None (anonymous / admin path) skips the kill-switch check."""
        client = _mock_oanda()
        self._call(client, user_id=None)  # must not raise

    # ── Check 2: position ceiling ─────────────────────────────────────────────

    def test_position_ceiling_blocks_at_limit(self):
        from config.settings import HARD_MAX_POSITIONS
        open_trades = [{"instrument": "EUR_USD"}] * HARD_MAX_POSITIONS
        client = _mock_oanda(open_trade_count=HARD_MAX_POSITIONS,
                             open_trades=open_trades)
        with self.assertRaises(ValueError) as ctx:
            self._call(client)
        self.assertIn("ceiling", str(ctx.exception).lower())

    def test_position_ceiling_allows_one_below_limit(self):
        from config.settings import HARD_MAX_POSITIONS
        n = HARD_MAX_POSITIONS - 1
        client = _mock_oanda(open_trade_count=n,
                             open_trades=[{"instrument": "X"}] * n)
        self._call(client)  # must not raise

    def test_position_ceiling_blocks_at_zero_free_slots(self):
        """5 open = at ceiling → block."""
        from config.settings import HARD_MAX_POSITIONS
        client = _mock_oanda(
            open_trade_count=HARD_MAX_POSITIONS,
            open_trades=[{"instrument": "X"}] * HARD_MAX_POSITIONS,
        )
        with self.assertRaises(ValueError):
            self._call(client)

    # ── Check 3: daily drawdown → kill switch ─────────────────────────────────

    def test_daily_loss_triggers_kill_switch(self):
        from config.settings import HARD_MAX_DAILY_LOSS_PCT
        loss = -(HARD_MAX_DAILY_LOSS_PCT + 0.01) * 10000  # e.g. -600 on 10k
        client = _mock_oanda(balance=10000, unrealized_pl=loss)
        with patch("src.alerter.Alerter"):
            with self.assertRaises(ValueError) as ctx:
                self._call(client, user_id=1)
        self.assertIn("daily loss", str(ctx.exception).lower())

    def test_daily_loss_exactly_at_limit_blocks(self):
        from config.settings import HARD_MAX_DAILY_LOSS_PCT
        loss = -HARD_MAX_DAILY_LOSS_PCT * 10000  # exactly at limit
        client = _mock_oanda(balance=10000, unrealized_pl=loss)
        with patch("src.alerter.Alerter"):
            with self.assertRaises(ValueError):
                self._call(client, user_id=1)

    def test_daily_loss_just_below_limit_allows(self):
        from config.settings import HARD_MAX_DAILY_LOSS_PCT
        loss = -(HARD_MAX_DAILY_LOSS_PCT - 0.005) * 10000  # just under
        client = _mock_oanda(balance=10000, unrealized_pl=loss)
        self._call(client, user_id=1)  # must not raise

    def test_daily_loss_activates_kill_switch_for_user(self):
        from config.settings import HARD_MAX_DAILY_LOSS_PCT
        from src.trading_engine import is_user_killed_today
        loss = -(HARD_MAX_DAILY_LOSS_PCT + 0.01) * 10000
        client = _mock_oanda(balance=10000, unrealized_pl=loss)
        with patch("src.alerter.Alerter"):
            try:
                self._call(client, user_id=42)
            except ValueError:
                pass
        self.assertTrue(is_user_killed_today(42))

    # ── Check 4: per-trade risk ceiling ───────────────────────────────────────

    def test_per_trade_risk_too_high_blocked(self):
        # 1000 units × 200-pip SL × 0.0001 / $500 balance = 4% > 2% hard cap
        # Using a small balance to produce an oversized risk fraction.
        client = _mock_oanda(balance=500)
        with self.assertRaises(ValueError) as ctx:
            self._call(client, units=1000, sl_pips=200)
        self.assertIn("risk", str(ctx.exception).lower())

    def test_per_trade_risk_within_limit_allowed(self):
        # 1000 units × 20-pip SL × 0.0001 / $10,000 balance = 0.02% < 2%
        client = _mock_oanda(balance=10000)
        self._call(client, units=1000, sl_pips=20)  # must not raise

    def test_per_trade_risk_jpy_pip_value(self):
        """JPY pairs use pip_value=0.01, not 0.0001."""
        # 1001 × 20 × 0.01 / 10000 = 0.02002 → strictly exceeds 2% → block
        client = _mock_oanda(balance=10000)
        with self.assertRaises(ValueError):
            self._call(client, units=1001, sl_pips=20, instrument="USD_JPY")

    def test_per_trade_risk_jpy_small_units_allowed(self):
        """100 × 20 × 0.01 / 10000 = 0.002 → within limit for JPY."""
        client = _mock_oanda(balance=10000)
        self._call(client, units=100, sl_pips=20, instrument="USD_JPY")

    def test_account_summary_failure_does_not_block_if_position_ok(self):
        """If Oanda summary is unreachable, checks 3 & 4 are skipped gracefully."""
        client = _mock_oanda()
        client.get_account_summary.side_effect = RuntimeError("Oanda down")
        # Should not raise — kill switch and position cap passed
        self._call(client)


# ══════════════════════════════════════════════════════════════════════════════
# TestCalculatePositionSizeClamping
# ══════════════════════════════════════════════════════════════════════════════

class TestCalculatePositionSizeClamping(unittest.TestCase):

    def test_risk_pct_clamped_to_hard_ceiling(self):
        from config.settings import HARD_MAX_RISK_PCT
        from src.trading_engine import calculate_position_size
        # Requesting 10% risk should produce the same units as requesting the
        # 2% hard cap, because calculate_position_size clamps internally.
        unclamped = calculate_position_size(
            balance=10000, risk_pct=0.10, stop_loss_pips=20
        )
        at_cap = calculate_position_size(
            balance=10000, risk_pct=HARD_MAX_RISK_PCT, stop_loss_pips=20
        )
        self.assertEqual(
            unclamped, at_cap,
            "Requesting 10% risk should yield same units as the 2% hard cap",
        )

    def test_normal_risk_not_clamped(self):
        from config.settings import HARD_MAX_RISK_PCT
        from src.trading_engine import calculate_position_size
        # 0.5% < 2% → units at 0.5% must be less than units at the hard cap
        low_risk_units = calculate_position_size(10000, 0.005, 20)
        cap_units      = calculate_position_size(10000, HARD_MAX_RISK_PCT, 20)
        self.assertLess(low_risk_units, cap_units)

    def test_clamped_result_satisfies_hard_limit(self):
        """Units returned by calculate_position_size must never breach the hard cap."""
        from config.settings import HARD_MAX_RISK_PCT
        from src.trading_engine import calculate_position_size
        units = calculate_position_size(
            balance=10000, risk_pct=0.50, stop_loss_pips=20  # absurdly high input
        )
        actual_risk = units * 20 * 0.0001 / 10000
        self.assertLessEqual(
            actual_risk, HARD_MAX_RISK_PCT + 1e-6,  # tiny float tolerance
            f"Returned {units} units risks {actual_risk:.4f} which exceeds the hard cap",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TestRunUserAutoTradeKillSwitch
# ══════════════════════════════════════════════════════════════════════════════

class TestRunUserAutoTradeKillSwitch(unittest.TestCase):

    def setUp(self):
        self._orig = _temp_db()

    def tearDown(self):
        _restore_db(self._orig)

    def test_auto_trade_skips_entire_cycle_when_killed(self):
        from src.trading_engine import _activate_killswitch, run_user_auto_trade
        with patch("src.alerter.Alerter"):
            _activate_killswitch(user_id=1, drawdown_pct=0.06)

        settings = {
            "mode": "auto",
            "auto_trade_enabled": True,
            "threshold": 0.55,
            "max_positions": 3,
            "trading_account_id": None,
            "units": 1000,
            "sl_pips": 20,
            "tp_pips": 40,
            "use_regime_filter": True,
        }
        results = run_user_auto_trade(user_id=1, settings=settings)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["action"], "skipped")
        self.assertIn("kill switch", results[0]["reason"].lower())

    def test_auto_trade_not_skipped_for_different_user(self):
        from src.trading_engine import _activate_killswitch
        with patch("src.alerter.Alerter"):
            _activate_killswitch(user_id=1, drawdown_pct=0.06)

        settings = {
            "mode": "auto",
            "auto_trade_enabled": True,
        }
        # User 2 should not be blocked — the function reaches run_portfolio_signal_check.
        # patch with create=True injects the function into the stub module so the
        # lazy `from src.multi_pair_manager import run_portfolio_signal_check` inside
        # run_user_auto_trade picks it up.
        with patch("src.multi_pair_manager.run_portfolio_signal_check",
                   return_value=[], create=True):
            from src.trading_engine import run_user_auto_trade
            results = run_user_auto_trade(user_id=2, settings=settings)
        # The key assertion: the early-exit kill-switch path was not taken.
        # Kill switch would return [{"action":"skipped","reason":"...kill switch..."}]
        self.assertFalse(
            any("kill switch" in str(r.get("reason","")).lower() for r in results),
            "User 2 must not be blocked by user 1's kill switch",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TestPlaceTradeEnforcement
# ══════════════════════════════════════════════════════════════════════════════

class TestPlaceTradeEnforcement(unittest.TestCase):
    """place_trade() must call enforce_hard_risk_limits() before placing."""

    def setUp(self):
        self._orig = _temp_db()

    def tearDown(self):
        _restore_db(self._orig)

    def test_place_trade_calls_enforce(self):
        from src.trading_engine import place_trade
        with patch("src.trading_engine.get_client_for_user") as mock_get, \
             patch("src.trading_engine.enforce_hard_risk_limits") as mock_enforce, \
             patch("src.trading_engine.log_trade", return_value=1):

            mock_client = _mock_oanda()
            mock_client.get_live_price.return_value = {
                "mid": 1.085, "bid": 1.084, "ask": 1.086,
                "spread": 0.2, "spread_raw": 0.00002, "tradeable": True,
                "timestamp": "2026-01-01T08:00:00Z",
            }
            mock_client.place_market_order.return_value = {
                "order_id": "1", "trade_id": "2", "instrument": "EUR_USD",
                "units": 1000, "fill_price": 1.085, "time": "2026-01-01", "pl": 0,
            }
            mock_get.return_value = mock_client

            place_trade(
                user_id=1, instrument="EUR_USD", direction="BUY",
                units=1000, sl_pips=20, tp_pips=40,
            )

        mock_enforce.assert_called_once()
        call_kwargs = mock_enforce.call_args[1]
        self.assertEqual(call_kwargs["user_id"],    1)
        self.assertEqual(call_kwargs["units"],      1000)
        self.assertEqual(call_kwargs["instrument"], "EUR_USD")
        self.assertEqual(call_kwargs["sl_pips"],    20)

    def test_place_trade_blocked_by_hard_limit(self):
        """If enforce raises, place_trade must not reach Oanda."""
        from src.trading_engine import place_trade
        with patch("src.trading_engine.get_client_for_user") as mock_get, \
             patch("src.trading_engine.enforce_hard_risk_limits",
                   side_effect=ValueError("hard limit breach")):

            mock_client = _mock_oanda()
            mock_get.return_value = mock_client

            with self.assertRaises(ValueError) as ctx:
                place_trade(user_id=1, instrument="EUR_USD",
                            direction="BUY", units=1000)
            self.assertIn("hard limit breach", str(ctx.exception))

        mock_client.place_market_order.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# TestPaperTraderCheckRisk
# ══════════════════════════════════════════════════════════════════════════════

class TestPaperTraderCheckRisk(unittest.TestCase):
    """PaperTrader._check_risk() must enforce hard limits as Layer 1."""

    def setUp(self):
        self._orig = _temp_db()

    def tearDown(self):
        _restore_db(self._orig)

    def _make_trader(self, client=None):
        from src.paper_trader import PaperTrader
        trader = PaperTrader.__new__(PaperTrader)
        trader.instrument       = "EUR_USD"
        trader.units            = 1000
        trader.sl_pips          = 20.0
        trader.tp_pips          = 40.0
        trader.max_positions    = 3
        trader.max_daily_loss   = 0.02
        trader.use_regime_filter = False
        trader.user_id          = 1
        trader.oanda            = client or _mock_oanda()
        trader.alerter          = MagicMock()
        trader.regime           = MagicMock()
        return trader

    def test_hard_limit_breach_returns_false(self):
        trader = self._make_trader()
        with patch("src.trading_engine.enforce_hard_risk_limits",
                   side_effect=ValueError("position ceiling")):
            ok, reason = trader._check_risk()
        self.assertFalse(ok)
        self.assertIn("position ceiling", reason)

    def test_hard_limit_message_surfaces_to_caller(self):
        trader = self._make_trader()
        with patch("src.trading_engine.enforce_hard_risk_limits",
                   side_effect=ValueError("kill switch is active")):
            ok, reason = trader._check_risk()
        self.assertFalse(ok)
        self.assertIn("kill switch", reason.lower())

    def test_layer2_still_runs_when_layer1_passes(self):
        """After hard limits pass, the max_positions check must still run."""
        client = _mock_oanda(open_trade_count=5,
                             open_trades=[{"instrument": "X"}] * 5)
        trader = self._make_trader(client=client)
        trader.max_positions = 3  # user limit lower than hard ceiling
        with patch("src.trading_engine.enforce_hard_risk_limits"):  # Layer 1 passes
            ok, reason = trader._check_risk()
        self.assertFalse(ok)
        self.assertIn("max positions", reason.lower())

    def test_all_checks_pass_returns_true(self):
        client = _mock_oanda(balance=10000, unrealized_pl=0)
        trader = self._make_trader(client=client)
        with patch("src.trading_engine.enforce_hard_risk_limits"):
            ok, reason = trader._check_risk()
        self.assertTrue(ok)
        self.assertEqual(reason, "")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestHardRiskConstants,
        TestKillSwitch,
        TestEnforceHardRiskLimits,
        TestCalculatePositionSizeClamping,
        TestRunUserAutoTradeKillSwitch,
        TestPlaceTradeEnforcement,
        TestPaperTraderCheckRisk,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
