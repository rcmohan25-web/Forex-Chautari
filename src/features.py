import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_bollinger(series: pd.Series, window=20, num_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    width = (upper - lower) / ma
    return ma, upper, lower, width


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Price returns ---
    df["return_1"] = df["Close"].pct_change(1)
    df["return_3"] = df["Close"].pct_change(3)
    df["return_5"] = df["Close"].pct_change(5)

    # --- Moving average ratios ---
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_ratio_10"] = df["Close"] / df["ma_10"]
    df["ma_ratio_20"] = df["Close"] / df["ma_20"]

    # --- Volatility ---
    df["volatility_10"] = df["return_1"].rolling(10).std()
    df["volatility_20"] = df["return_1"].rolling(20).std()
    df["vol_ratio"] = df["volatility_10"] / (df["volatility_20"] + 1e-9)

    # --- Candle structure ---
    df["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["open_close_range"] = (df["Close"] - df["Open"]) / df["Open"]

    # --- RSI ---
    df["rsi_14"] = compute_rsi(df["Close"], 14)
    df["rsi_slope"] = df["rsi_14"].diff(3)

    # --- MACD ---
    macd, macd_signal, macd_hist = compute_macd(df["Close"])
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    # --- ATR ---
    df["atr_14"] = compute_atr(df, 14)

    # --- Bollinger Bands ---
    bb_mid, bb_upper, bb_lower, bb_width = compute_bollinger(df["Close"], 20, 2)
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["bb_width"] = bb_width
    df["bb_position"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # --- Lagged returns (added in v2) ---
    df["lag1_return"] = df["return_1"].shift(1)
    df["lag2_return"] = df["return_1"].shift(2)

    # --- Momentum (added in v2) ---
    df["momentum_5"]  = df["Close"] / df["Close"].shift(5) - 1
    df["momentum_10"] = df["Close"] / df["Close"].shift(10) - 1

    # --- Day of week (added in v2 — captures weekly seasonality) ---
    df["dow"] = pd.to_datetime(df["Date"]).dt.dayofweek

    # --- Target: will next close be higher? ---
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)


    from src.regime_detector import RegimeDetector
    rd = RegimeDetector()
    df = rd.add_regime_features(df)

    df = df.dropna().reset_index(drop=True)
    return df


# v1 features (kept for backward compatibility)
FEATURE_COLUMNS = [
    "return_1", "return_3", "return_5",
    "ma_ratio_10", "ma_ratio_20",
    "volatility_10", "volatility_20",
    "high_low_range", "open_close_range",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "atr_14", "bb_width", "bb_position",
]

# v2 features — expanded set used by the improved model
FEATURE_COLUMNS_V2 = FEATURE_COLUMNS + [
    "vol_ratio", "rsi_slope",
    "lag1_return", "lag2_return",
    "momentum_5", "momentum_10",
    "dow",
    # ── Regime features (added in v3) ──
    "adx", "plus_di", "minus_di",
    "adx_trending", "adx_ranging",
]
