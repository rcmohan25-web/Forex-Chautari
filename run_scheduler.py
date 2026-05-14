"""
ForexChautari — Automated Multi-pair Scheduler

Jobs (UTC):
  08:05 — fetch fresh candles for all pairs
  08:10 — portfolio signal check (all pairs, log to DB)
  08:30 — daily account summary to Telegram
  00:01 Mon — retrain all pair models

Run:
    python run_scheduler.py
    nohup python run_scheduler.py > logs/scheduler.log 2>&1 &
"""

import sys, time, subprocess, os
import schedule
from datetime import datetime, date

from src.multi_pair_manager import fetch_all_pairs, run_portfolio_signal_check
from src.oanda_client import OandaClient
from src.alerter import Alerter
from src.database import (
    log_signal, get_signals_log, init_db, get_auto_trade_users,
    get_platform_settings, setting_bool,
)
from src.data_loader import load_forex_data
from src.logger import get_logger
from config.settings import (
    DEFAULT_SIGNAL_THRESHOLD, DEFAULT_MAX_POSITIONS,
    SIGNALS_LOG_PATH, PAPER_TRADES_PATH,
)

logger  = get_logger("scheduler")
alerter = Alerter()


def job_fetch_data():
    logger.info("=== Fetching data for all pairs ===")
    try:
        settings = get_platform_settings()
        if not setting_bool(settings, "auto_fetch_enabled", True):
            logger.info("Auto-fetch disabled in platform settings")
            return
        count = int(settings.get("fetch_count", 100) or 100)
        results = fetch_all_pairs(count=count)
        ok   = [p for p,r in results.items() if r["ok"]]
        fail = [p for p,r in results.items() if not r["ok"]]
        logger.info(f"Fetch — ok: {ok}  failed: {fail}")
        if fail:
            alerter.send_error(f"Data fetch failed for: {fail}")
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        alerter.send_error(f"Fetch error: {e}")


def _latest_price_from_csv(pair: str) -> float:
    try:
        from config.settings import data_path
        df = load_forex_data(data_path(pair))
        return float(df["Close"].iloc[-1])
    except Exception:
        return 0.0


def job_portfolio_signal_check():
    logger.info("=== Portfolio signal check ===")
    try:
        settings = get_platform_settings()
        from src.multi_pair_manager import get_portfolio_signals
        sig_df = get_portfolio_signals(DEFAULT_SIGNAL_THRESHOLD)

        # Log all signals to database
        for _, row in sig_df.iterrows():
            if row.get("ok"):
                try:
                    log_signal(
                        pair=row["pair"],
                        signal=row["signal"],
                        prob_up=float(row["prob_up"]),
                        confidence=row["confidence"],
                        regime=str(row.get("regime","unknown")),
                        tradeable=bool(row.get("tradeable", False)),
                        price=_latest_price_from_csv(row["pair"]),
                    )
                except Exception as e:
                    logger.warning(f"Signal log failed for {row['pair']}: {e}")

        # Run auto-trading only for users who explicitly enabled it.
        results = []
        if setting_bool(settings, "auto_trade_enabled", True):
            auto_users = get_auto_trade_users()
            for au in auto_users:
                user_results = run_portfolio_signal_check(
                    threshold=float(au.get("threshold") or DEFAULT_SIGNAL_THRESHOLD),
                    max_positions=int(au.get("max_positions") or DEFAULT_MAX_POSITIONS),
                    user_id=int(au["id"]),
                    account_db_id=au.get("trading_account_id"),
                    default_units=int(au.get("units") or 1000),
                    sl_pips=float(au.get("sl_pips") or 20),
                    tp_pips=float(au.get("tp_pips") or 40),
                    use_regime_filter=bool(au.get("use_regime_filter", True)),
                )
                for r in user_results:
                    r["username"] = au["username"]
                results.extend(user_results)
        else:
            auto_users = []
            logger.info("Platform auto-trade disabled in settings")

        traded  = [r for r in results if r.get("action") == "order_placed"]
        skipped = [r for r in results if r.get("action") == "skipped"]
        errors  = [r for r in results if r.get("action") == "error"]

        summary = (
            f"📊 <b>Portfolio Signal Check</b>\n\n"
            f"👤 Auto users: {len(auto_users)}\n"
            f"✅ Traded : {len(traded)}\n"
            f"⏭ Skipped: {len(skipped)}\n"
            f"❌ Errors : {len(errors)}\n"
        )
        if traded:
            summary += "\n<b>Trades:</b>\n"
            for r in traded:
                summary += f"  • {r.get('username','user')} · {r['pair']}  {r['signal']}  @ {r.get('price','?')}\n"
        alerter._send(summary)
        logger.info(f"Signal check done — {len(traded)} traded, {len(skipped)} skipped")

    except Exception as e:
        logger.error(f"Portfolio signal check failed: {e}")
        alerter.send_error(f"Portfolio check error: {e}")


def job_daily_summary():
    logger.info("=== Sending daily summary ===")
    try:
        oanda   = OandaClient()
        summary = oanda.get_account_summary()
        today   = date.today().isoformat()

        # Count signals today from DB
        try:
            sigs_today = [s for s in get_signals_log(limit=500)
                          if s.get("created_at","").startswith(today)]
            signals_today = len(sigs_today)
        except Exception:
            signals_today = 0

        # Count paper trades today
        trades_today = 0
        try:
            import json
            if os.path.exists(PAPER_TRADES_PATH):
                with open(PAPER_TRADES_PATH) as f:
                    all_trades = json.load(f)
                trades_today = sum(1 for t in all_trades
                                   if t.get("timestamp","").startswith(today))
        except Exception:
            pass

        alerter.send_daily_summary(
            balance=summary["balance"],
            nav=summary["nav"],
            unrealized_pl=summary["unrealized_pl"],
            open_trades=summary["open_trades"],
            signals_today=signals_today,
            trades_today=trades_today,
        )
    except Exception as e:
        logger.error(f"Daily summary failed: {e}")
        alerter.send_error(f"Daily summary error: {e}")


def job_weekly_retrain():
    settings = get_platform_settings()
    if not setting_bool(settings, "auto_train_enabled", True):
        logger.info("Auto-train disabled in platform settings")
        return
    weekday = int(settings.get("train_weekday_utc", 0) or 0)
    if datetime.utcnow().weekday() != weekday:
        return
    logger.info("=== Weekly retrain (all pairs) ===")
    alerter._send("🔄 <b>Weekly Retrain Started</b> — all pairs")
    try:
        result = subprocess.run(
            [sys.executable, "train_all.py", "--fetch"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            alerter._send(f"✅ <b>Weekly Retrain Complete</b>\n<pre>{result.stdout[-800:]}</pre>")
            logger.info("Weekly retrain complete")
        else:
            alerter.send_error(f"Retrain failed:\n{result.stderr[-500:]}")
    except Exception as e:
        logger.error(f"Weekly retrain error: {e}")
        alerter.send_error(f"Retrain error: {e}")


def schedule_jobs_from_settings():
    settings = get_platform_settings()
    schedule.clear()
    schedule.every().day.at(settings.get("fetch_time_utc", "08:05")).do(job_fetch_data)
    schedule.every().day.at(settings.get("signal_check_time_utc", "08:10")).do(job_portfolio_signal_check)
    schedule.every().day.at(settings.get("daily_summary_time_utc", "08:30")).do(job_daily_summary)
    schedule.every().day.at(settings.get("train_time_utc", "00:01")).do(job_weekly_retrain)

if __name__ == "__main__":
    # Ensure DB is initialised
    init_db()
    schedule_jobs_from_settings()
    settings = get_platform_settings()

    logger.info("ForexChautari scheduler started")
    alerter._send(
        f"🚀 <b>ForexChautari Scheduler Started</b>\n\n"
        f"Pairs: EUR/USD · GBP/USD · USD/JPY · AUD/USD\n"
        f"Jobs:\n"
        f"  {settings.get('fetch_time_utc','08:05')} UTC — fetch data\n"
        f"  {settings.get('signal_check_time_utc','08:10')} UTC — signal check + trades\n"
        f"  {settings.get('daily_summary_time_utc','08:30')} UTC — daily summary\n"
        f"  weekday {settings.get('train_weekday_utc','0')} "
        f"{settings.get('train_time_utc','00:01')} UTC — retrain all"
    )

    # Run once on startup
    logger.info("Running startup checks...")
    job_fetch_data()
    job_portfolio_signal_check()

    while True:
        schedule.run_pending()
        time.sleep(60)
