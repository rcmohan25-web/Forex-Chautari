"""
Regime detector — classifies the market as trending or ranging.

Uses ADX (Average Directional Index) as the primary regime indicator:
  - ADX > 25 = trending market  → trust momentum signals (MACD, MA crossover)
  - ADX < 20 = ranging market   → trust mean-reversion signals (RSI, BB position)
  - 20–25    = transitioning    → reduce position size / skip

Additionally detects:
  - Volatility regime (high/low vs 90-day average)
  - Trend direction (bullish / bearish) using MA alignment

The paper trader uses this to filter out signals that go against
the current market regime, reducing false positives.
"""

import numpy as np
import pandas as pd
from src.logger import get_logger

logger = get_logger("regime")


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute the Average Directional Index (ADX).
    Requires columns: High, Low, Close.
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional movement
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Smoothed TR and DM
    atr      = pd.Series(tr).ewm(span=period, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr

    # DX and ADX
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx.fillna(0)


def compute_plus_di(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    atr     = pd.Series(tr).ewm(span=period, adjust=False).mean()
    return 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr


def compute_minus_di(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr      = pd.Series(tr).ewm(span=period, adjust=False).mean()
    return 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr


class RegimeDetector:
    """
    Classifies the current market regime and decides if a signal is tradeable.
    """

    TRENDING_THRESHOLD    = 25.0   # ADX above this = trending
    TRANSITIONING_LOWER   = 20.0   # ADX below this = ranging
    HIGH_VOL_MULTIPLIER   = 1.5    # vol > 1.5x average = high volatility

    def __init__(self, adx_period: int = 14, vol_window: int = 90):
        self.adx_period = adx_period
        self.vol_window = vol_window

    def detect(self, df: pd.DataFrame) -> dict:
        """
        Run full regime detection on a OHLC DataFrame.

        Returns a dict with:
          - adx: current ADX value
          - adx_regime: "trending" | "transitioning" | "ranging"
          - trend_direction: "bullish" | "bearish" | "neutral"
          - vol_regime: "high" | "normal" | "low"
          - tradeable_signals: list of signal types that are valid
          - summary: human-readable string
        """
        df = df.copy()

        # ── ADX regime ────────────────────────────────────────────────────────
        adx      = compute_adx(df, self.adx_period)
        plus_di  = compute_plus_di(df, self.adx_period)
        minus_di = compute_minus_di(df, self.adx_period)

        current_adx      = float(adx.iloc[-1])
        current_plus_di  = float(plus_di.iloc[-1])
        current_minus_di = float(minus_di.iloc[-1])

        if current_adx >= self.TRENDING_THRESHOLD:
            adx_regime = "trending"
        elif current_adx >= self.TRANSITIONING_LOWER:
            adx_regime = "transitioning"
        else:
            adx_regime = "ranging"

        # ── Trend direction ───────────────────────────────────────────────────
        ma20 = df["Close"].rolling(20).mean().iloc[-1]
        ma50 = df["Close"].rolling(50).mean().iloc[-1]
        last_close = float(df["Close"].iloc[-1])

        if last_close > ma20 > ma50 and current_plus_di > current_minus_di:
            trend_direction = "bullish"
        elif last_close < ma20 < ma50 and current_minus_di > current_plus_di:
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"

        # ── Volatility regime ─────────────────────────────────────────────────
        returns    = df["Close"].pct_change()
        current_vol = float(returns.rolling(10).std().iloc[-1])
        avg_vol     = float(returns.rolling(self.vol_window).std().iloc[-1])

        if avg_vol > 0:
            vol_ratio = current_vol / avg_vol
        else:
            vol_ratio = 1.0

        if vol_ratio >= self.HIGH_VOL_MULTIPLIER:
            vol_regime = "high"
        elif vol_ratio <= 0.5:
            vol_regime = "low"
        else:
            vol_regime = "normal"

        # ── What signals are valid in this regime ─────────────────────────────
        tradeable_signals = []
        if adx_regime == "trending":
            # In a trend, trust momentum (MACD, MA signals)
            if trend_direction == "bullish":
                tradeable_signals = ["BUY"]
            elif trend_direction == "bearish":
                tradeable_signals = ["SELL"]
            else:
                tradeable_signals = ["BUY", "SELL"]
        elif adx_regime == "ranging":
            # In a range, trust mean-reversion (RSI, BB)
            tradeable_signals = ["BUY", "SELL"]    # both valid; model decides
        else:
            # Transitioning — skip unless very high confidence
            tradeable_signals = []

        # Extra safety: reduce exposure in extreme volatility
        if vol_regime == "high":
            tradeable_signals = []   # wait for volatility to settle

        summary = (
            f"ADX={current_adx:.1f} ({adx_regime}) | "
            f"Direction={trend_direction} | "
            f"Vol={vol_regime} ({vol_ratio:.2f}x avg) | "
            f"Tradeable={tradeable_signals}"
        )
        logger.info(f"Regime: {summary}")

        return {
            "adx":               round(current_adx, 2),
            "plus_di":           round(current_plus_di, 2),
            "minus_di":          round(current_minus_di, 2),
            "adx_regime":        adx_regime,
            "trend_direction":   trend_direction,
            "vol_regime":        vol_regime,
            "vol_ratio":         round(vol_ratio, 2),
            "tradeable_signals": tradeable_signals,
            "summary":           summary,
        }

    def is_tradeable(self, regime: dict, signal: str) -> bool:
        """Check if a given signal is valid for the current regime."""
        return signal in regime.get("tradeable_signals", [])

    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ADX, +DI, -DI, and regime columns to a DataFrame.
        Useful for including regime info in the training features.
        """
        df = df.copy()
        df["adx"]      = compute_adx(df, self.adx_period)
        df["plus_di"]  = compute_plus_di(df, self.adx_period)
        df["minus_di"] = compute_minus_di(df, self.adx_period)
        df["adx_trending"]    = (df["adx"] >= self.TRENDING_THRESHOLD).astype(int)
        df["adx_ranging"]     = (df["adx"] < self.TRANSITIONING_LOWER).astype(int)
        return df
