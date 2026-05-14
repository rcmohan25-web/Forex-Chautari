"""
Telegram alerter — sends trading signals, risk alerts, and errors.

Setup:
  1. Open Telegram → search for @BotFather
  2. Send: /newbot  → follow prompts → copy the token
  3. Send any message to your new bot
  4. Visit: https://api.telegram.org/bot<TOKEN>/getUpdates
  5. Copy the "id" value from the "chat" object → that's your CHAT_ID
  6. Add both to .env:
       TELEGRAM_BOT_TOKEN=...
       TELEGRAM_CHAT_ID=...

All methods fail silently (log warning) if credentials are missing,
so the app works even without Telegram configured.
"""

import requests
from src.logger import get_logger
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = get_logger("alerter")

_API = "https://api.telegram.org/bot"

# Emoji map
_SIG_EMOJI  = {"BUY": "🟢", "SELL": "🔴"}
_CONF_EMOJI = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "💤"}


class Alerter:
    """Sends formatted Telegram messages for trading events."""

    def __init__(self):
        self.token   = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.enabled = bool(
            self.token and self.chat_id
            and self.token != "YOUR_BOT_TOKEN_HERE"
            and self.chat_id != "YOUR_CHAT_ID_HERE"
        )
        if not self.enabled:
            logger.info("Telegram not configured — alerts disabled. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")

    def _send(self, text: str) -> bool:
        """Send a message. Returns True on success, False on failure."""
        if not self.enabled:
            logger.info(f"[TELEGRAM DISABLED] Would send: {text[:80]}...")
            return False
        try:
            r = requests.post(
                f"{_API}{self.token}/sendMessage",
                json={
                    "chat_id":    self.chat_id,
                    "text":       text,
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
            if not r.ok:
                logger.warning(f"Telegram send failed [{r.status_code}]: {r.text}")
                return False
            return True
        except Exception as e:
            logger.warning(f"Telegram error: {e}")
            return False

    # ── Message types ─────────────────────────────────────────────────────────

    def send_signal(
        self,
        signal: str,
        prob_up: float,
        confidence: str,
        price: float,
        instrument: str,
        traded: bool,
        reason: str = "",
        trade_id: str = "",
        sl: float = None,
        tp: float = None,
    ) -> bool:
        """Send a signal notification — traded or skipped."""
        sig_e  = _SIG_EMOJI.get(signal, "⚪")
        conf_e = _CONF_EMOJI.get(confidence, "")
        action = "✅ <b>TRADE PLACED</b>" if traded else "⏭ <b>SIGNAL (skipped)</b>"

        lines = [
            f"{action}",
            f"",
            f"{sig_e} <b>{signal}</b> on {instrument.replace('_','/')}",
            f"{conf_e} Confidence: <b>{confidence}</b>",
            f"📊 Prob UP: <b>{prob_up:.4f}</b>  |  Prob DOWN: <b>{1-prob_up:.4f}</b>",
            f"💱 Price: <b>{price:.5f}</b>",
        ]
        if traded and sl and tp:
            lines += [
                f"🛑 Stop Loss:   <b>{sl:.5f}</b>",
                f"🎯 Take Profit: <b>{tp:.5f}</b>",
            ]
        if traded and trade_id:
            lines.append(f"🔖 Trade ID: <code>{trade_id}</code>")
        if not traded and reason:
            lines.append(f"ℹ️ Reason: {reason}")

        return self._send("\n".join(lines))

    def send_risk_alert(self, reason: str) -> bool:
        """Send a risk management alert."""
        text = (
            f"⚠️ <b>RISK ALERT — Trade Blocked</b>\n\n"
            f"Reason: {reason}\n\n"
            f"The system will continue monitoring but will not place orders "
            f"until the risk condition clears."
        )
        return self._send(text)

    def send_error(self, error: str) -> bool:
        """Send an error alert."""
        text = (
            f"🔴 <b>SYSTEM ERROR</b>\n\n"
            f"<code>{error[:500]}</code>\n\n"
            f"Check logs for details."
        )
        return self._send(text)

    def send_daily_summary(
        self,
        balance: float,
        nav: float,
        unrealized_pl: float,
        open_trades: int,
        signals_today: int,
        trades_today: int,
    ) -> bool:
        """Send end-of-day account summary."""
        pl_emoji = "📈" if unrealized_pl >= 0 else "📉"
        text = (
            f"📋 <b>Daily Summary</b>\n\n"
            f"💰 Balance: <b>${balance:,.2f}</b>\n"
            f"📊 NAV: <b>${nav:,.2f}</b>\n"
            f"{pl_emoji} Unrealized P&L: <b>${unrealized_pl:+,.2f}</b>\n"
            f"📂 Open trades: <b>{open_trades}</b>\n"
            f"📡 Signals today: <b>{signals_today}</b>\n"
            f"✅ Trades placed: <b>{trades_today}</b>"
        )
        return self._send(text)

    def send_data_stale_alert(self, days_old: int, instrument: str) -> bool:
        """Alert when data hasn't been updated."""
        text = (
            f"⚠️ <b>Stale Data Warning</b>\n\n"
            f"{instrument} data is <b>{days_old} days old</b>.\n"
            f"Fetch failed or hasn't run. Check the API connection."
        )
        return self._send(text)

    def send_model_degraded_alert(self, wf_accuracy: float) -> bool:
        """Alert when walk-forward accuracy drops below threshold."""
        text = (
            f"🟡 <b>Model Performance Warning</b>\n\n"
            f"Walk-forward accuracy dropped to <b>{wf_accuracy:.3f}</b>\n"
            f"(below the 51% minimum threshold)\n\n"
            f"Consider retraining the model with fresh data."
        )
        return self._send(text)

    def test(self) -> bool:
        """Send a test message to verify the connection works."""
        return self._send(
            "✅ <b>Forex ML Terminal — Telegram connected!</b>\n\n"
            "You will receive signal alerts, risk notifications, and daily summaries here."
        )
