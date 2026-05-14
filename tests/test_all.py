"""
ForexChautari — Full Test Suite
Run with: python tests/test_all.py

Covers:
  TestDatabase       — users, auth, trades, signals, notifications, stats
  TestAuth           — sessions, password hashing, plan limits
  TestDataLoader     — load, validate, deduplicate
  TestFeatures       — indicators, V2 columns, no look-ahead
  TestModel          — train, evaluate, save/load, overfit gap
  TestBacktest       — returns, spread, edge cases
  TestApiFetcher     — response parsing, merge, rate-limit
  TestTrainPipeline  — walk-forward splits and schema
  TestMultiPair      — portfolio signals, path helpers, ok/fail handling
"""

import os
import sys
import json
import tempfile
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_ohlc(n=300, start_price=1.10, seed=42) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    closes = start_price + np.cumsum(rng.normal(0, 0.001, n))
    opens  = closes + rng.normal(0, 0.0005, n)
    highs  = np.maximum(closes, opens) + rng.uniform(0, 0.002, n)
    lows   = np.minimum(closes, opens) - rng.uniform(0, 0.002, n)
    dates  = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Date": dates, "Open": opens, "High": highs,
        "Low": lows, "Close": closes, "Volume": 0,
    })


# ── TestDatabase ──────────────────────────────────────────────────────────────

class TestDatabase(unittest.TestCase):

    def setUp(self):
        """Use a temporary database for each test."""
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        import src.database as db_mod
        self._orig_db_path = db_mod.DB_PATH
        db_mod.DB_PATH = self.db_path
        from src.database import init_db
        init_db()

    def tearDown(self):
        import src.database as db_mod
        db_mod.DB_PATH = self._orig_db_path

    def test_admin_seeded_on_init(self):
        from src.database import authenticate_user
        u = authenticate_user("admin", "admin123")
        self.assertIsNotNone(u)
        self.assertEqual(u["role"], "admin")
        self.assertEqual(u["plan"], "enterprise")

    def test_create_and_authenticate_user(self):
        from src.database import create_user, authenticate_user
        create_user("tuser", "t@test.com", "pass1234", "Test User")
        u = authenticate_user("tuser", "pass1234")
        self.assertIsNotNone(u)
        self.assertEqual(u["username"], "tuser")
        self.assertEqual(u["role"], "user")
        self.assertEqual(u["plan"], "free")
        self.assertFalse(u["auto_trade"])
        self.assertEqual(u["max_pairs"], 1)

    def test_wrong_password_returns_none(self):
        from src.database import create_user, authenticate_user
        create_user("tuser2", "t2@test.com", "pass1234")
        self.assertIsNone(authenticate_user("tuser2", "wrongpass"))

    def test_login_by_email(self):
        from src.database import create_user, authenticate_user
        create_user("tuser3", "t3@test.com", "pass1234")
        u = authenticate_user("t3@test.com", "pass1234")
        self.assertIsNotNone(u)

    def test_plan_limits_propagate(self):
        from src.database import create_user, authenticate_user
        for plan, expected in [("free",1),("basic",2),("pro",4),("enterprise",99)]:
            create_user(f"u_{plan}", f"{plan}@t.com", "pass1234", plan=plan)
            u = authenticate_user(f"u_{plan}", "pass1234")
            self.assertEqual(u["max_pairs"], expected, f"Plan {plan} max_pairs wrong")

    def test_update_plan(self):
        from src.database import create_user, authenticate_user, update_user_plan
        create_user("planuser", "plan@t.com", "pass1234", plan="free")
        u    = authenticate_user("planuser", "pass1234")
        admin = authenticate_user("admin", "admin123")
        update_user_plan(u["id"], "pro", admin["id"])
        u2 = authenticate_user("planuser", "pass1234")
        self.assertEqual(u2["plan"], "pro")
        self.assertTrue(u2["auto_trade"])
        self.assertEqual(u2["max_pairs"], 4)

    def test_deactivate_reactivate(self):
        from src.database import create_user, authenticate_user, deactivate_user, reactivate_user
        create_user("actuser", "act@t.com", "pass1234")
        u     = authenticate_user("actuser", "pass1234")
        admin = authenticate_user("admin", "admin123")
        deactivate_user(u["id"], admin["id"])
        self.assertIsNone(authenticate_user("actuser", "pass1234"))
        reactivate_user(u["id"], admin["id"])
        self.assertIsNotNone(authenticate_user("actuser", "pass1234"))

    def test_log_and_get_trades(self):
        from src.database import create_user, authenticate_user, log_trade, close_trade, get_user_trades
        create_user("tradeuser", "trade@t.com", "pass1234", plan="pro")
        u   = authenticate_user("tradeuser", "pass1234")
        tid = log_trade(u["id"], "EUR_USD", "BUY", 1.085, 1000, "auto", "b-001")
        close_trade(tid, 1.087, 20.0)
        trades = get_user_trades(u["id"])
        self.assertTrue(any(t["id"] == tid for t in trades))
        closed = next(t for t in trades if t["id"] == tid)
        self.assertEqual(closed["status"], "closed")
        self.assertAlmostEqual(closed["pnl"], 20.0)

    def test_trade_stats(self):
        from src.database import create_user, authenticate_user, log_trade, close_trade, get_trade_stats
        create_user("statsuser", "stats@t.com", "pass1234", plan="pro")
        u  = authenticate_user("statsuser", "pass1234")
        t1 = log_trade(u["id"], "EUR_USD", "BUY",  1.085, 1000)
        t2 = log_trade(u["id"], "GBP_USD", "SELL", 1.270, 1000)
        close_trade(t1,  1.087,  20.0)
        close_trade(t2,  1.272, -20.0)
        ts = get_trade_stats(u["id"])
        self.assertEqual(ts["closed_trades"], 2)
        self.assertAlmostEqual(ts["total_pnl"], 0.0)
        self.assertEqual(ts["win_rate"], 50.0)

    def test_trade_stats_all_users(self):
        from src.database import get_trade_stats
        ts = get_trade_stats(None)
        self.assertIn("total_trades", ts)
        self.assertIn("win_rate", ts)

    def test_log_signal(self):
        from src.database import log_signal, get_signals_log
        log_signal("EUR_USD", "BUY", 0.68, "HIGH", "trending", True, 1.085)
        sigs = get_signals_log(limit=5)
        self.assertGreater(len(sigs), 0)
        self.assertEqual(sigs[0]["pair"], "EUR_USD")

    def test_notifications_on_register(self):
        from src.database import create_user, authenticate_user, get_notifications
        create_user("notifuser", "notif@t.com", "pass1234")
        u     = authenticate_user("notifuser", "pass1234")
        notifs = get_notifications(u["id"])
        self.assertGreater(len(notifs), 0)

    def test_mark_notifications_read(self):
        from src.database import create_user, authenticate_user, get_notifications, mark_notifications_read
        create_user("readuser", "read@t.com", "pass1234")
        u = authenticate_user("readuser", "pass1234")
        mark_notifications_read(u["id"])
        unread = get_notifications(u["id"], unread_only=True)
        self.assertEqual(len(unread), 0)

    def test_platform_stats(self):
        from src.database import get_platform_stats
        stats = get_platform_stats()
        for key in ["total_users","total_trades","total_signals","total_pnl","new_today","plans"]:
            self.assertIn(key, stats)

    def test_update_profile(self):
        from src.database import create_user, authenticate_user, update_user_profile, get_user_by_id
        create_user("profuser", "prof@t.com", "pass1234", "Old Name")
        u = authenticate_user("profuser", "pass1234")
        update_user_profile(u["id"], full_name="New Name", phone="+977-123")
        u2 = get_user_by_id(u["id"])
        self.assertEqual(u2["full_name"], "New Name")
        self.assertEqual(u2["phone"], "+977-123")

    def test_update_password(self):
        from src.database import create_user, authenticate_user, update_user_password
        create_user("pwuser", "pw@t.com", "oldpass1")
        u = authenticate_user("pwuser", "oldpass1")
        update_user_password(u["id"], "newpass2")
        self.assertIsNone(authenticate_user("pwuser", "oldpass1"))
        self.assertIsNotNone(authenticate_user("pwuser", "newpass2"))

    def test_duplicate_username_fails(self):
        from src.database import create_user
        create_user("dup", "dup@t.com", "pass1234")
        with self.assertRaises(Exception):
            create_user("dup", "dup2@t.com", "pass1234")


# ── TestAuth ──────────────────────────────────────────────────────────────────

class TestAuth(unittest.TestCase):

    def setUp(self):
        self.tmpdir  = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "auth_test.db")
        import src.database as db_mod
        self._orig   = db_mod.DB_PATH
        db_mod.DB_PATH = self.db_path
        from src.database import init_db
        init_db()

    def tearDown(self):
        import src.database as db_mod
        db_mod.DB_PATH = self._orig

    def test_session_lifecycle(self):
        from src.database import create_session, get_session, destroy_session
        tok = create_session(1, "admin")
        s   = get_session(tok)
        self.assertIsNotNone(s)
        self.assertEqual(s["user_id"], 1)
        destroy_session(tok)
        self.assertIsNone(get_session(tok))

    def test_password_hashing(self):
        from src.database import hash_password, verify_password
        import secrets
        salt = secrets.token_hex(16)
        h    = hash_password("mypassword", salt)
        self.assertTrue(verify_password("mypassword", salt, h))
        self.assertFalse(verify_password("wrongpass",  salt, h))

    def test_hash_is_not_plaintext(self):
        from src.database import hash_password
        h = hash_password("secret", "salt123")
        self.assertNotIn("secret", h)


# ── TestDataLoader ────────────────────────────────────────────────────────────

class TestDataLoader(unittest.TestCase):

    def test_load_roundtrip(self):
        from src.data_loader import load_forex_data
        df = make_ohlc(100)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f.name, index=False); path = f.name
        try:
            loaded = load_forex_data(path)
            self.assertEqual(len(loaded), 100)
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        from src.data_loader import load_forex_data
        with self.assertRaises(FileNotFoundError):
            load_forex_data("/tmp/nonexistent_xyz.csv")

    def test_validate_ohlc_passes(self):
        from src.data_loader import validate_ohlc
        validate_ohlc(make_ohlc(50))

    def test_validate_ohlc_missing_column(self):
        from src.data_loader import validate_ohlc
        with self.assertRaises(ValueError):
            validate_ohlc(make_ohlc(50).drop(columns=["High"]))

    def test_validate_ohlc_empty(self):
        from src.data_loader import validate_ohlc
        with self.assertRaises(ValueError):
            validate_ohlc(make_ohlc(0))

    def test_deduplicates_on_load(self):
        from src.data_loader import load_forex_data
        df  = make_ohlc(50)
        dup = pd.concat([df, df.head(10)], ignore_index=True)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            dup.to_csv(f.name, index=False); path = f.name
        try:
            loaded = load_forex_data(path)
            self.assertEqual(len(loaded), 50)
        finally:
            os.unlink(path)

    def test_adds_volume_if_missing(self):
        from src.data_loader import load_forex_data
        df = make_ohlc(30).drop(columns=["Volume"])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f.name, index=False); path = f.name
        try:
            loaded = load_forex_data(path)
            self.assertIn("Volume", loaded.columns)
        finally:
            os.unlink(path)


# ── TestFeatures ──────────────────────────────────────────────────────────────

class TestFeatures(unittest.TestCase):

    def setUp(self):
        from src.features import add_features
        self.df_raw = make_ohlc(400)
        self.df     = add_features(self.df_raw)

    def test_v1_features_present(self):
        from src.features import FEATURE_COLUMNS
        for col in FEATURE_COLUMNS:
            self.assertIn(col, self.df.columns, f"Missing: {col}")

    def test_v2_features_present(self):
        from src.features import FEATURE_COLUMNS_V2
        for col in FEATURE_COLUMNS_V2:
            self.assertIn(col, self.df.columns, f"Missing V2: {col}")

    def test_target_is_binary(self):
        self.assertEqual(set(self.df["target"].unique()), {0, 1})

    def test_no_nans_in_features(self):
        from src.features import FEATURE_COLUMNS_V2
        self.assertFalse(self.df[FEATURE_COLUMNS_V2].isnull().any().any())

    def test_rsi_in_range(self):
        self.assertTrue((self.df["rsi_14"] >= 0).all())
        self.assertTrue((self.df["rsi_14"] <= 100).all())

    def test_dow_weekdays_only(self):
        self.assertTrue((self.df["dow"] >= 0).all())
        self.assertTrue((self.df["dow"] <= 4).all())

    def test_target_correctness(self):
        for i in range(5, 10):
            feat_date = self.df["Date"].iloc[i]
            raw_idx   = self.df_raw[self.df_raw["Date"] == feat_date].index[0]
            expected  = int(self.df_raw["Close"].iloc[raw_idx+1] > self.df_raw["Close"].iloc[raw_idx])
            self.assertEqual(int(self.df["target"].iloc[i]), expected)


# ── TestModel ─────────────────────────────────────────────────────────────────

class TestModel(unittest.TestCase):

    def setUp(self):
        from src.features import add_features, FEATURE_COLUMNS_V2
        df       = add_features(make_ohlc(600))
        self.X   = df[FEATURE_COLUMNS_V2].values
        self.y   = df["target"].values

    def test_train_returns_model(self):
        from src.model import train_random_forest
        m = train_random_forest(self.X[:400], self.y[:400])
        self.assertIsNotNone(m)

    def test_predict_shape_and_probas(self):
        from src.model import train_random_forest, evaluate_model
        m = train_random_forest(self.X[:400], self.y[:400])
        preds, probas, _ = evaluate_model(m, self.X[400:], self.y[400:])
        self.assertEqual(len(preds), len(self.X[400:]))
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_accuracy_valid_range(self):
        from src.model import train_random_forest, evaluate_model
        m = train_random_forest(self.X[:400], self.y[:400])
        _, _, metrics = evaluate_model(m, self.X[400:], self.y[400:])
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(   metrics["accuracy"], 1.0)

    def test_save_load_roundtrip(self):
        from src.model import train_random_forest, evaluate_model, save_model_bundle, load_model_bundle
        from src.features import FEATURE_COLUMNS_V2
        m = train_random_forest(self.X[:400], self.y[:400])
        with tempfile.TemporaryDirectory() as d:
            import config.settings as s
            orig_mp, orig_mep = s.MODEL_PATH, s.METADATA_PATH
            s.MODEL_PATH    = os.path.join(d, "model.pkl")
            s.METADATA_PATH = os.path.join(d, "meta.json")
            try:
                save_model_bundle(m, list(FEATURE_COLUMNS_V2), {"accuracy_train": 0.65})
                m2, meta = load_model_bundle()
                np.testing.assert_array_equal(
                    m.predict(self.X[400:]),
                    m2.predict(self.X[400:])
                )
                self.assertEqual(meta["accuracy_train"], 0.65)
            finally:
                s.MODEL_PATH, s.METADATA_PATH = orig_mp, orig_mep

    def test_regularised_model_lower_overfit_gap(self):
        from src.model import train_random_forest, evaluate_model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        m_new = train_random_forest(self.X[:400], self.y[:400])
        _, _, tr_new = evaluate_model(m_new, self.X[:400], self.y[:400])
        _, _, te_new = evaluate_model(m_new, self.X[400:], self.y[400:])
        gap_new = tr_new["accuracy"] - te_new["accuracy"]
        m_old = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42)
        m_old.fit(self.X[:400], self.y[:400])
        gap_old = (accuracy_score(self.y[:400], m_old.predict(self.X[:400])) -
                   accuracy_score(self.y[400:], m_old.predict(self.X[400:])))
        self.assertLess(gap_new, gap_old,
                        f"New gap {gap_new:.4f} should be less than old {gap_old:.4f}")


# ── TestBacktest ──────────────────────────────────────────────────────────────

class TestBacktest(unittest.TestCase):

    def setUp(self):
        from src.features import add_features
        self.df = add_features(make_ohlc(400))

    def test_returns_tuple(self):
        from src.backtest import run_backtest
        preds  = np.ones(len(self.df), dtype=int)
        probas = np.column_stack([np.full(len(self.df), 0.4), np.full(len(self.df), 0.6)])
        bt_df, results = run_backtest(self.df, preds, probas, 0.55, 0.0001)
        self.assertIsInstance(bt_df, pd.DataFrame)
        self.assertIsInstance(results, dict)

    def test_result_keys(self):
        from src.backtest import run_backtest
        preds  = np.ones(len(self.df), dtype=int)
        probas = np.column_stack([np.full(len(self.df), 0.4), np.full(len(self.df), 0.6)])
        _, results = run_backtest(self.df, preds, probas, 0.55, 0.0001)
        for key in ["total_strategy_return","total_market_return","win_rate","max_drawdown","num_trades"]:
            self.assertIn(key, results)

    def test_spread_reduces_return(self):
        from src.backtest import run_backtest
        preds  = np.ones(len(self.df), dtype=int)
        probas = np.column_stack([np.full(len(self.df), 0.4), np.full(len(self.df), 0.6)])
        _, r0 = run_backtest(self.df, preds, probas, 0.55, 0.0)
        _, r1 = run_backtest(self.df, preds, probas, 0.55, 0.001)
        self.assertGreaterEqual(r0["total_strategy_return"], r1["total_strategy_return"])

    def test_no_trades_at_high_threshold(self):
        from src.backtest import run_backtest
        preds  = np.ones(len(self.df), dtype=int)
        probas = np.column_stack([np.full(len(self.df), 0.94), np.full(len(self.df), 0.06)])
        _, results = run_backtest(self.df, preds, probas, 0.95, 0.0001)
        self.assertEqual(results["num_trades"], 0)


# ── TestApiFetcher ────────────────────────────────────────────────────────────

class TestApiFetcher(unittest.TestCase):

    def test_rate_limit_note(self):
        from src.api_fetcher import _parse_av_response
        with self.assertRaises(RuntimeError):
            _parse_av_response({"Note": "rate limit"})

    def test_error_message(self):
        from src.api_fetcher import _parse_av_response
        with self.assertRaises(ValueError):
            _parse_av_response({"Error Message": "bad call"})

    def test_missing_ts_key(self):
        from src.api_fetcher import _parse_av_response
        with self.assertRaises(ValueError):
            _parse_av_response({"Meta Data": {}})

    def test_valid_response_parses(self):
        from src.api_fetcher import _parse_av_response
        fake = {"Time Series FX (Daily)": {
            f"2024-{(i//28)+1:02d}-{(i%28)+1:02d}": {
                "1. open":"1.10","2. high":"1.11","3. low":"1.09","4. close":"1.105",
            }
            for i in range(150)
        }}
        df = _parse_av_response(fake)
        self.assertEqual(len(df), 150)
        self.assertIn("Close", df.columns)

    def test_too_few_rows_raises(self):
        from src.api_fetcher import _parse_av_response
        fake = {"Time Series FX (Daily)": {
            "2024-01-01": {"1. open":"1.10","2. high":"1.11","3. low":"1.09","4. close":"1.105"}
        }}
        with self.assertRaises(ValueError):
            _parse_av_response(fake)

    def test_merge_combines_data(self):
        import unittest.mock as mock
        from src import api_fetcher
        old_df = make_ohlc(200, seed=1)
        new_df = make_ohlc(50, seed=2)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            old_df.to_csv(f.name, index=False); path = f.name
        try:
            with mock.patch.object(api_fetcher, "fetch_eurusd_from_alpha_vantage", return_value=new_df):
                merged = api_fetcher.fetch_and_merge(save_path=path)
            self.assertGreaterEqual(len(merged), max(len(old_df), len(new_df)))
            self.assertEqual(merged["Date"].nunique(), len(merged))
        finally:
            os.unlink(path)


# ── TestTrainPipeline ─────────────────────────────────────────────────────────

class TestTrainPipeline(unittest.TestCase):

    def test_walk_forward_columns(self):
        from src.features import add_features, FEATURE_COLUMNS_V2
        from src.train_pipeline import walk_forward_validation
        df    = add_features(make_ohlc(600))
        wf_df = walk_forward_validation(df, FEATURE_COLUMNS_V2,
                                        train_size=300, test_size=100, step_size=200)
        for col in ["split_id","accuracy","strategy_return","num_trades"]:
            self.assertIn(col, wf_df.columns)

    def test_walk_forward_accuracy_range(self):
        from src.features import add_features, FEATURE_COLUMNS_V2
        from src.train_pipeline import walk_forward_validation
        df    = add_features(make_ohlc(600))
        wf_df = walk_forward_validation(df, FEATURE_COLUMNS_V2,
                                        train_size=300, test_size=100, step_size=200)
        self.assertTrue((wf_df["accuracy"] >= 0).all())
        self.assertTrue((wf_df["accuracy"] <= 1).all())


# ── TestMultiPair ─────────────────────────────────────────────────────────────

class TestMultiPair(unittest.TestCase):

    def test_portfolio_signals_no_models(self):
        """Should return ok=False for all pairs, not crash."""
        import unittest.mock as mock
        from src import multi_pair_manager as mpm
        original = mpm.get_pair_signal

        def mock_no_model(pair, threshold=0.55):
            return {"pair": pair, "ok": False, "reason": "No model"}

        mpm.get_pair_signal = mock_no_model
        try:
            df = mpm.get_portfolio_signals(0.60)
            self.assertFalse(df.empty)
            self.assertTrue((df["ok"] == False).all())
        finally:
            mpm.get_pair_signal = original

    def test_portfolio_signals_with_one_model(self):
        """Should rank the ok pair first."""
        import unittest.mock as mock
        from src import multi_pair_manager as mpm
        original = mpm.get_pair_signal

        def mock_mixed(pair, threshold=0.55):
            if pair == "EUR_USD":
                return {"pair": pair, "ok": True, "signal": "BUY",
                        "prob_up": 0.68, "prob_down": 0.32, "confidence": "HIGH",
                        "above_threshold": True, "regime": "trending", "trend": "bullish",
                        "vol_regime": "normal", "regime_ok": True, "tradeable": True,
                        "wf_accuracy": 0.53, "test_accuracy": 0.52, "latest_date": "2026-01-01"}
            return {"pair": pair, "ok": False, "reason": "No model"}

        mpm.get_pair_signal = mock_mixed
        try:
            df = mpm.get_portfolio_signals(0.60)
            self.assertEqual(df.iloc[0]["pair"], "EUR_USD")
            self.assertTrue(df.iloc[0]["ok"])
        finally:
            mpm.get_pair_signal = original

    def test_path_helpers(self):
        from config.settings import data_path, model_path, meta_path, wf_path
        for pair in ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]:
            self.assertIn(pair, data_path(pair))
            self.assertIn(pair, model_path(pair))
            self.assertIn(pair, meta_path(pair))
            self.assertIn(pair, wf_path(pair))
            self.assertTrue(model_path(pair).endswith(".pkl"))
            self.assertTrue(data_path(pair).endswith(".csv"))

    def test_active_pairs_list(self):
        from config.settings import ACTIVE_PAIRS
        self.assertEqual(len(ACTIVE_PAIRS), 4)
        self.assertIn("EUR_USD", ACTIVE_PAIRS)
        self.assertIn("GBP_USD", ACTIVE_PAIRS)
        self.assertIn("USD_JPY", ACTIVE_PAIRS)
        self.assertIn("AUD_USD", ACTIVE_PAIRS)


# ── Runner ────────────────────────────────────────────────────────────────────

# ── TestTradingEngine ─────────────────────────────────────────────────────────

class TestTradingEngine(unittest.TestCase):

    def test_calculate_sl_tp_buy(self):
        from src.trading_engine import calculate_sl_tp
        sl, tp = calculate_sl_tp("EUR_USD", "BUY", 1.0850, sl_pips=20, tp_pips=40)
        self.assertLess(sl, 1.0850, "SL must be below entry for BUY")
        self.assertGreater(tp, 1.0850, "TP must be above entry for BUY")
        self.assertAlmostEqual(sl, 1.0830, places=4)
        self.assertAlmostEqual(tp, 1.0890, places=4)

    def test_calculate_sl_tp_sell(self):
        from src.trading_engine import calculate_sl_tp
        sl, tp = calculate_sl_tp("EUR_USD", "SELL", 1.0850, sl_pips=20, tp_pips=40)
        self.assertGreater(sl, 1.0850, "SL must be above entry for SELL")
        self.assertLess(tp, 1.0850, "TP must be below entry for SELL")

    def test_calculate_sl_tp_jpy(self):
        """JPY pairs use pip=0.01 not 0.0001."""
        from src.trading_engine import calculate_sl_tp
        sl, tp = calculate_sl_tp("USD_JPY", "BUY", 148.50, sl_pips=20, tp_pips=40)
        self.assertAlmostEqual(sl, 148.30, places=2)
        self.assertAlmostEqual(tp, 148.90, places=2)

    def test_calculate_sl_tp_rr_ratio(self):
        """TP should be 2× SL distance (20 pips SL, 40 pips TP)."""
        from src.trading_engine import calculate_sl_tp
        sl, tp = calculate_sl_tp("GBP_USD", "BUY", 1.2700, sl_pips=20, tp_pips=40)
        sl_dist = 1.2700 - sl
        tp_dist = tp - 1.2700
        self.assertAlmostEqual(tp_dist / sl_dist, 2.0, places=1)

    def test_calculate_position_size_normal(self):
        from src.trading_engine import calculate_position_size
        units = calculate_position_size(
            balance=10000, risk_pct=0.01, stop_loss_pips=20, pip_value=0.0001
        )
        self.assertGreaterEqual(units, 100)
        self.assertLessEqual(units, 100000)
        # 1% of 10000 = $100 risk, 20 pips × 0.0001 = 0.002 per unit → 50000 units
        self.assertAlmostEqual(units, 50000, delta=1000)

    def test_calculate_position_size_min_floor(self):
        from src.trading_engine import calculate_position_size
        units = calculate_position_size(balance=100, risk_pct=0.001, stop_loss_pips=100)
        self.assertGreaterEqual(units, 100, "Must respect minimum units")

    def test_calculate_position_size_max_cap(self):
        from src.trading_engine import calculate_position_size
        units = calculate_position_size(balance=10000000, risk_pct=0.10, stop_loss_pips=1)
        self.assertLessEqual(units, 100000, "Must respect maximum units cap")

    def test_calculate_position_size_zero_sl(self):
        from src.trading_engine import calculate_position_size
        units = calculate_position_size(balance=10000, risk_pct=0.01, stop_loss_pips=0)
        self.assertGreaterEqual(units, 100, "Zero SL should return minimum, not crash")

    def test_get_client_for_user_no_account(self):
        """Should raise ValueError when user has no linked account."""
        from src.trading_engine import get_client_for_user
        from src.database import init_db, create_user, authenticate_user
        import tempfile, src.database as db_mod
        tmpdir  = tempfile.mkdtemp()
        orig    = db_mod.DB_PATH
        db_mod.DB_PATH = f"{tmpdir}/test.db"
        init_db()
        try:
            create_user("notradingacc", "nt@test.com", "pass1234")
            u = authenticate_user("notradingacc", "pass1234")
            with self.assertRaises(ValueError) as ctx:
                get_client_for_user(u["id"])
            self.assertIn("No trading account", str(ctx.exception))
        finally:
            db_mod.DB_PATH = orig

    def test_get_risk_metrics_no_account(self):
        """Should return error dict gracefully when no account linked."""
        from src.trading_engine import get_risk_metrics
        result = get_risk_metrics(user_id=99999)
        self.assertIn("error", result, "Should return error dict, not raise")


# ── TestOandaClient ───────────────────────────────────────────────────────────

class TestOandaClient(unittest.TestCase):
    """
    All tests use mocked HTTP responses — no real Oanda account needed.
    Tests verify the client correctly parses responses and handles errors.
    """

    def _mock_client(self, api_key="testkey", account_id="101-001-TEST-001", env="practice"):
        """Build a client bypassing the credential validation."""
        from src.oanda_client import OandaClient
        import unittest.mock as mock
        with mock.patch.dict("os.environ", {
            "OANDA_API_KEY":    api_key,
            "OANDA_ACCOUNT_ID": account_id,
            "OANDA_ENVIRONMENT": env,
        }):
            return OandaClient(api_key=api_key, account_id=account_id, environment=env)

    def test_missing_api_key_raises(self):
        from src.oanda_client import OandaClient
        with self.assertRaises(ValueError) as ctx:
            OandaClient(api_key="", account_id="101-001-TEST", environment="practice")
        self.assertIn("API key", str(ctx.exception))

    def test_missing_account_id_raises(self):
        from src.oanda_client import OandaClient
        with self.assertRaises(ValueError) as ctx:
            OandaClient(api_key="validkey", account_id="", environment="practice")
        self.assertIn("Account ID", str(ctx.exception))

    def test_placeholder_api_key_raises(self):
        from src.oanda_client import OandaClient
        with self.assertRaises(ValueError):
            OandaClient(api_key="YOUR_OANDA_API_KEY_HERE",
                        account_id="101-001-TEST", environment="practice")

    def test_practice_base_url(self):
        client = self._mock_client(env="practice")
        self.assertIn("fxpractice", client.base_url)

    def test_live_base_url(self):
        client = self._mock_client(env="live")
        self.assertIn("fxtrade", client.base_url)

    def test_get_account_summary_parses_correctly(self):
        import unittest.mock as mock
        client = self._mock_client()
        fake_response = {
            "account": {
                "balance": "100000.00", "NAV": "100150.50",
                "unrealizedPL": "150.50", "pl": "200.00",
                "openTradeCount": "3", "marginUsed": "500.00",
                "marginAvailable": "99500.00", "currency": "USD", "leverage": "50",
            }
        }
        with mock.patch.object(client, "_get", return_value=fake_response):
            result = client.get_account_summary()
        self.assertAlmostEqual(result["balance"],      100000.0)
        self.assertAlmostEqual(result["nav"],          100150.5)
        self.assertAlmostEqual(result["unrealized_pl"], 150.5)
        self.assertAlmostEqual(result["realized_pl"],   200.0)
        self.assertEqual(result["open_trades"],        3)
        self.assertEqual(result["currency"],           "USD")

    def test_get_live_price_calculates_spread(self):
        import unittest.mock as mock
        client = self._mock_client()
        fake_response = {"prices": [{
            "bids": [{"price": "1.08400"}],
            "asks": [{"price": "1.08403"}],
            "tradeable": True,
            "time": "2026-01-01T08:00:00Z",
            "instrument": "EUR_USD",
        }]}
        with mock.patch.object(client, "_get", return_value=fake_response):
            result = client.get_live_price("EUR_USD")
        self.assertAlmostEqual(result["bid"], 1.08400, places=5)
        self.assertAlmostEqual(result["ask"], 1.08403, places=5)
        self.assertAlmostEqual(result["mid"], 1.08402, places=4)
        self.assertAlmostEqual(result["spread"], 0.3, places=1)  # 0.3 pips
        self.assertTrue(result["tradeable"])

    def test_get_candles_skips_incomplete(self):
        import unittest.mock as mock
        client = self._mock_client()
        fake_response = {"candles": [
            {"time": "2024-01-01T00:00:00Z", "complete": True,
             "mid": {"o":"1.10","h":"1.11","l":"1.09","c":"1.105"}, "volume": 100},
            {"time": "2024-01-02T00:00:00Z", "complete": False,  # current bar — skip
             "mid": {"o":"1.11","h":"1.12","l":"1.10","c":"1.115"}, "volume": 50},
        ]}
        with mock.patch.object(client, "_get", return_value=fake_response):
            df = client.get_candles("EUR_USD", "D", 2)
        self.assertEqual(len(df), 1, "Incomplete candle must be excluded")
        self.assertAlmostEqual(df["Close"].iloc[0], 1.105)

    def test_get_candles_empty_raises(self):
        import unittest.mock as mock
        client = self._mock_client()
        with mock.patch.object(client, "_get", return_value={"candles": []}):
            with self.assertRaises(RuntimeError):
                client.get_candles("EUR_USD", "D", 10)

    def test_place_market_order_parses_fill(self):
        import unittest.mock as mock
        client = self._mock_client()
        fake_response = {"orderFillTransaction": {
            "orderID": "123", "tradeOpened": {"tradeID": "456"},
            "instrument": "EUR_USD", "units": "1000",
            "price": "1.08402", "time": "2026-01-01T08:00:00Z", "pl": "0",
        }}
        with mock.patch.object(client, "_post", return_value=fake_response):
            result = client.place_market_order("EUR_USD", 1000)
        self.assertEqual(result["order_id"],   "123")
        self.assertEqual(result["trade_id"],   "456")
        self.assertAlmostEqual(result["fill_price"], 1.08402)
        self.assertEqual(result["units"],      1000)

    def test_get_open_trades_parses_correctly(self):
        import unittest.mock as mock
        client = self._mock_client()
        fake_response = {"trades": [{
            "id": "789", "instrument": "EUR_USD", "currentUnits": "1000",
            "price": "1.08400", "unrealizedPL": "20.50",
            "openTime": "2026-01-01T08:00:00Z",
            "stopLossOrder": {"price": "1.08200"},
            "takeProfitOrder": {"price": "1.08800"},
        }]}
        with mock.patch.object(client, "_get", return_value=fake_response):
            trades = client.get_open_trades()
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["trade_id"],    "789")
        self.assertEqual(trades[0]["instrument"],  "EUR_USD")
        self.assertEqual(trades[0]["units"],       1000)
        self.assertAlmostEqual(trades[0]["unrealized_pl"], 20.5)
        self.assertEqual(trades[0]["stop_loss"],   "1.08200")

    def test_get_transaction_history_filters_fills(self):
        import unittest.mock as mock
        client = self._mock_client()
        fake_response = {"transactions": [
            {"id":"1","type":"ORDER_FILL","instrument":"EUR_USD",
             "units":"1000","price":"1.084","pl":"0","time":"2026-01-01T08:00:00Z"},
            {"id":"2","type":"HEARTBEAT","time":"2026-01-01T08:01:00Z"},  # filtered out
        ]}
        with mock.patch.object(client, "_get", return_value=fake_response):
            txns = client.get_transaction_history(count=10)
        self.assertEqual(len(txns), 1, "HEARTBEAT should be filtered out")
        self.assertEqual(txns[0]["type"], "ORDER_FILL")

    def test_validate_credentials_valid(self):
        import unittest.mock as mock
        client = self._mock_client()
        fake_summary = {
            "balance":100000.0,"nav":100000.0,"unrealized_pl":0.0,
            "realized_pl":0.0,"open_trades":0,"margin_used":0.0,
            "margin_avail":100000.0,"currency":"USD","leverage":"50",
        }
        with mock.patch.object(client, "get_account_summary", return_value=fake_summary):
            result = client.validate_credentials()
        self.assertTrue(result["valid"])
        self.assertEqual(result["balance"], 100000.0)
        self.assertEqual(result["currency"], "USD")

    def test_validate_credentials_invalid(self):
        import unittest.mock as mock
        client = self._mock_client()
        with mock.patch.object(client, "get_account_summary", side_effect=RuntimeError("401")):
            result = client.validate_credentials()
        self.assertFalse(result["valid"])
        self.assertIn("error", result)

    def test_http_error_raises_runtime_error(self):
        import unittest.mock as mock, requests as req
        client = self._mock_client()
        mock_resp = mock.Mock()
        mock_resp.ok = False
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        with mock.patch("requests.get", return_value=mock_resp):
            with self.assertRaises(RuntimeError) as ctx:
                client.get_account_summary()
            self.assertIn("401", str(ctx.exception))

    def test_get_all_prices_multiple_instruments(self):
        import unittest.mock as mock
        client = self._mock_client()
        fake_response = {"prices": [
            {"instrument":"EUR_USD","bids":[{"price":"1.08400"}],
             "asks":[{"price":"1.08403"}],"tradeable":True,"time":"2026-01-01T08:00:00Z"},
            {"instrument":"GBP_USD","bids":[{"price":"1.26900"}],
             "asks":[{"price":"1.26905"}],"tradeable":True,"time":"2026-01-01T08:00:00Z"},
        ]}
        with mock.patch.object(client, "_get", return_value=fake_response):
            prices = client.get_all_prices(["EUR_USD","GBP_USD"])
        self.assertEqual(len(prices), 2)
        self.assertEqual(prices[0]["instrument"], "EUR_USD")
        self.assertEqual(prices[1]["instrument"], "GBP_USD")
        self.assertAlmostEqual(prices[0]["spread_pips"], 0.3, places=1)


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestDatabase, TestAuth, TestDataLoader, TestFeatures,
        TestModel, TestBacktest, TestApiFetcher,
        TestTrainPipeline, TestMultiPair,
        TestTradingEngine, TestOandaClient,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
