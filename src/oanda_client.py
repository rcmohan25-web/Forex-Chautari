"""
ForexChautari — Oanda v20 REST API client.
Supports per-user credentials (passed in) and environment-level defaults.
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Optional
from src.logger import get_logger

logger = get_logger("oanda")

_URLS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live":     "https://api-fxtrade.oanda.com",
}

GRANULARITY_MAP = {
    "daily": "D", "H4": "H4", "H1": "H1", "M15": "M15", "M5": "M5",
}


class OandaClient:
    """
    Initialise with explicit credentials OR fall back to env vars.
    This allows per-user credentials stored in the DB.
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        account_id: Optional[str] = None,
        environment: Optional[str] = None,
    ):
        self.api_key = (
            os.getenv("OANDA_API_KEY", "")
            if api_key is None
            else api_key
        )

        self.account_id = (
            os.getenv("OANDA_ACCOUNT_ID", "")
            if account_id is None
            else account_id
        )
        
        self.env        = environment or os.getenv("OANDA_ENVIRONMENT", "practice")

        if not self.api_key or self.api_key in ("YOUR_OANDA_API_KEY_HERE", ""):
            raise ValueError(
                "Oanda API key not set.\n"
                "Add to .env: OANDA_API_KEY=your_key\n"
                "Or get a free practice key at: oanda.com/register"
            )
        if not self.account_id or self.account_id in ("YOUR_OANDA_ACCOUNT_ID_HERE", ""):
            raise ValueError(
                "Oanda Account ID not set.\n"
                "Add to .env: OANDA_ACCOUNT_ID=101-001-XXXXXXX-001"
            )

        self.base_url = _URLS.get(self.env, _URLS["practice"])
        self.headers  = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get(self, path: str, params: dict = None) -> dict:
        r = requests.get(f"{self.base_url}{path}", headers=self.headers,
                         params=params, timeout=15)
        if not r.ok:
            raise RuntimeError(f"Oanda GET {path} [{r.status_code}]: {r.text[:300]}")
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        r = requests.post(f"{self.base_url}{path}", headers=self.headers,
                          data=json.dumps(body), timeout=15)
        if not r.ok:
            raise RuntimeError(f"Oanda POST {path} [{r.status_code}]: {r.text[:300]}")
        return r.json()

    def _put(self, path: str, body: dict) -> dict:
        r = requests.put(f"{self.base_url}{path}", headers=self.headers,
                         data=json.dumps(body), timeout=15)
        if not r.ok:
            raise RuntimeError(f"Oanda PUT {path} [{r.status_code}]: {r.text[:300]}")
        return r.json()

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account_summary(self) -> dict:
        data = self._get(f"/v3/accounts/{self.account_id}/summary")
        a    = data["account"]
        return {
            "balance":       float(a["balance"]),
            "nav":           float(a["NAV"]),
            "unrealized_pl": float(a["unrealizedPL"]),
            "realized_pl":   float(a.get("pl", 0)),
            "open_trades":   int(a["openTradeCount"]),
            "margin_used":   float(a["marginUsed"]),
            "margin_avail":  float(a.get("marginAvailable", 0)),
            "currency":      a["currency"],
            "leverage":      a.get("leverage", "N/A"),
        }

    def get_daily_pnl(self) -> float:
        data = self._get(f"/v3/accounts/{self.account_id}/summary")
        return float(data["account"].get("pl", 0))

    # ── Market data ───────────────────────────────────────────────────────────

    def get_candles(self, instrument="EUR_USD", granularity="D", count=500) -> pd.DataFrame:
        data = self._get(
            f"/v3/instruments/{instrument}/candles",
            params={"granularity": granularity, "count": count, "price": "M"},
        )
        rows = []
        for c in data.get("candles", []):
            if not c.get("complete", True):
                continue
            mid = c["mid"]
            rows.append({
                "Date":   pd.to_datetime(c["time"]).tz_localize(None),
                "Open":   float(mid["o"]),
                "High":   float(mid["h"]),
                "Low":    float(mid["l"]),
                "Close":  float(mid["c"]),
                "Volume": int(c.get("volume", 0)),
            })
        if not rows:
            raise RuntimeError(f"No candles for {instrument} {granularity}")
        return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)

    def get_live_price(self, instrument="EUR_USD") -> dict:
        data  = self._get(
            f"/v3/accounts/{self.account_id}/pricing",
            params={"instruments": instrument},
        )
        price = data["prices"][0]
        bid   = float(price["bids"][0]["price"])
        ask   = float(price["asks"][0]["price"])
        return {
            "instrument": instrument,
            "bid":        bid,
            "ask":        ask,
            "mid":        round((bid + ask) / 2, 5),
            "spread":     round((ask - bid) * 10000, 2),  # in pips
            "spread_raw": round(ask - bid, 5),
            "tradeable":  price.get("tradeable", True),
            "timestamp":  price["time"],
        }

    def get_all_prices(self, instruments: list) -> list:
        """Get live prices for multiple instruments at once."""
        inst_str = ",".join(instruments)
        data     = self._get(
            f"/v3/accounts/{self.account_id}/pricing",
            params={"instruments": inst_str},
        )
        result = []
        for price in data.get("prices", []):
            bid = float(price["bids"][0]["price"])
            ask = float(price["asks"][0]["price"])
            result.append({
                "instrument": price["instrument"],
                "bid":        bid,
                "ask":        ask,
                "mid":        round((bid + ask) / 2, 5),
                "spread_pips":round((ask - bid) * 10000, 2),
                "tradeable":  price.get("tradeable", True),
            })
        return result

    # ── Orders ────────────────────────────────────────────────────────────────

    def place_market_order(
        self,
        instrument:        str,
        units:             int,
        stop_loss_price:   Optional[float] = None,
        take_profit_price: Optional[float] = None,
        client_tag:        str = "forexchautari",
    ) -> dict:
        order: dict = {
            "type":        "MARKET",
            "instrument":  instrument,
            "units":       str(units),
            "timeInForce": "FOK",
            "clientExtensions": {"tag": client_tag},
        }
        if stop_loss_price:
            order["stopLossOnFill"]   = {"price": f"{stop_loss_price:.5f}"}
        if take_profit_price:
            order["takeProfitOnFill"] = {"price": f"{take_profit_price:.5f}"}

        data = self._post(f"/v3/accounts/{self.account_id}/orders", {"order": order})
        fill = data.get("orderFillTransaction", {})
        return {
            "order_id":   fill.get("orderID", ""),
            "trade_id":   fill.get("tradeOpened", {}).get("tradeID", ""),
            "instrument": instrument,
            "units":      int(fill.get("units", units)),
            "fill_price": float(fill.get("price", 0)),
            "time":       fill.get("time", ""),
            "pl":         float(fill.get("pl", 0)),
        }

    def place_limit_order(
        self,
        instrument: str,
        units:      int,
        price:      float,
        stop_loss:  Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> dict:
        """Place a limit order at a specific price."""
        order: dict = {
            "type":        "LIMIT",
            "instrument":  instrument,
            "units":       str(units),
            "price":       f"{price:.5f}",
            "timeInForce": "GTC",
        }
        if stop_loss:
            order["stopLossOnFill"]   = {"price": f"{stop_loss:.5f}"}
        if take_profit:
            order["takeProfitOnFill"] = {"price": f"{take_profit:.5f}"}
        data = self._post(f"/v3/accounts/{self.account_id}/orders", {"order": order})
        return data

    def modify_trade_sl_tp(
        self,
        trade_id:   str,
        stop_loss:  Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> dict:
        """Modify stop loss / take profit on an open trade."""
        body: dict = {}
        if stop_loss:
            body["stopLoss"]   = {"price": f"{stop_loss:.5f}"}
        if take_profit:
            body["takeProfit"] = {"price": f"{take_profit:.5f}"}
        return self._put(
            f"/v3/accounts/{self.account_id}/trades/{trade_id}/orders", body
        )

    def close_trade(self, trade_id: str) -> dict:
        data = self._put(
            f"/v3/accounts/{self.account_id}/trades/{trade_id}/close", {}
        )
        fill = data.get("orderFillTransaction", {})
        return {
            "trade_id":   trade_id,
            "fill_price": float(fill.get("price", 0)),
            "pl":         float(fill.get("pl", 0)),
            "time":       fill.get("time", ""),
        }

    def close_all_positions(self) -> list:
        positions = self._get(
            f"/v3/accounts/{self.account_id}/openPositions"
        ).get("positions", [])
        results = []
        for pos in positions:
            inst  = pos["instrument"]
            long  = int(pos["long"]["units"])
            short = int(pos["short"]["units"])
            if long > 0:
                results.append(self._put(
                    f"/v3/accounts/{self.account_id}/positions/{inst}/close",
                    {"longUnits": "ALL"}
                ))
            if short < 0:
                results.append(self._put(
                    f"/v3/accounts/{self.account_id}/positions/{inst}/close",
                    {"shortUnits": "ALL"}
                ))
        return results

    # ── Positions & history ───────────────────────────────────────────────────

    def get_open_trades(self) -> list:
        data = self._get(f"/v3/accounts/{self.account_id}/openTrades")
        result = []
        for t in data.get("trades", []):
            result.append({
                "trade_id":      t["id"],
                "instrument":    t["instrument"],
                "units":         int(t["currentUnits"]),
                "open_price":    float(t["price"]),
                "unrealized_pl": float(t["unrealizedPL"]),
                "open_time":     t["openTime"][:19].replace("T", " "),
                "stop_loss":     t.get("stopLossOrder", {}).get("price", "—"),
                "take_profit":   t.get("takeProfitOrder", {}).get("price", "—"),
            })
        return result

    def get_transaction_history(self, count: int = 50) -> list:
        data = self._get(
            f"/v3/accounts/{self.account_id}/transactions",
            params={"count": count},
        )
        result = []
        for t in data.get("transactions", []):
            if t.get("type") in ("ORDER_FILL", "MARKET_ORDER", "LIMIT_ORDER"):
                result.append({
                    "id":         t.get("id"),
                    "type":       t.get("type"),
                    "instrument": t.get("instrument", ""),
                    "units":      t.get("units", 0),
                    "price":      t.get("price", 0),
                    "pl":         float(t.get("pl", 0)),
                    "time":       t.get("time", "")[:19].replace("T", " "),
                })
        return result

    def get_pending_orders(self) -> list:
        data = self._get(f"/v3/accounts/{self.account_id}/pendingOrders")
        result = []
        for o in data.get("orders", []):
            result.append({
                "order_id":   o.get("id"),
                "type":       o.get("type"),
                "instrument": o.get("instrument", ""),
                "units":      o.get("units", 0),
                "price":      o.get("price", "market"),
                "created":    o.get("createTime", "")[:19],
            })
        return result

    def cancel_order(self, order_id: str) -> dict:
        return self._put(
            f"/v3/accounts/{self.account_id}/orders/{order_id}/cancel", {}
        )

    # ── Validation ────────────────────────────────────────────────────────────

    def validate_credentials(self) -> dict:
        """Test credentials and return account info. Safe to call on connection."""
        try:
            summary = self.get_account_summary()
            return {
                "valid":      True,
                "balance":    summary["balance"],
                "currency":   summary["currency"],
                "environment": self.env,
                "account_id": self.account_id,
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}
