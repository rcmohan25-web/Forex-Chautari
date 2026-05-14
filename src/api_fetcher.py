"""
Live data fetcher for EUR/USD from Alpha Vantage.

Improvements over original:
- Merges new API data with existing CSV (no data loss)
- Validates the response has enough rows before saving
- Exposes fetch_and_merge() as the recommended call from the dashboard
- Raises typed, descriptive errors so the dashboard can show clear messages
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_AV_URL = "https://www.alphavantage.co/query"
_TS_KEY = "Time Series FX (Daily)"
_MIN_ROWS = 100  # refuse to save if API returns fewer rows than this


def _parse_av_response(data: dict) -> pd.DataFrame:
    """Parse raw Alpha Vantage JSON into a clean OHLC DataFrame."""
    if "Note" in data:
        raise RuntimeError(
            "Alpha Vantage rate limit reached. Free tier allows 25 requests/day. "
            "Wait a few minutes or upgrade your API plan."
        )
    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage info message (likely rate limit): {data['Information']}")
    if _TS_KEY not in data:
        raise ValueError(
            f"Unexpected API response — missing '{_TS_KEY}'. "
            f"Keys received: {list(data.keys())}"
        )

    rows = []
    for dt, values in data[_TS_KEY].items():
        rows.append({
            "Date":   dt,
            "Open":   float(values["1. open"]),
            "High":   float(values["2. high"]),
            "Low":    float(values["3. low"]),
            "Close":  float(values["4. close"]),
            "Volume": 0,
        })

    df = pd.DataFrame(rows)
    if len(df) < _MIN_ROWS:
        raise ValueError(
            f"API returned only {len(df)} rows (expected ≥ {_MIN_ROWS}). "
            "The response may be truncated or rate-limited."
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    return df


def fetch_eurusd_from_alpha_vantage(
    outputsize: str = "full",
    save_path: str = "data/EURUSD.csv",
) -> pd.DataFrame:
    """
    Fetch EUR/USD daily OHLC from Alpha Vantage and save to CSV.
    outputsize: 'compact' (last 100 bars) | 'full' (20 years)
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_REAL_KEY_HERE":
        raise ValueError(
            "Alpha Vantage API key not configured. "
            "Edit .env and set ALPHAVANTAGE_API_KEY=<your_key>. "
            "Free keys available at https://www.alphavantage.co/support/#api-key"
        )

    response = requests.get(
        _AV_URL,
        params={
            "function":    "FX_DAILY",
            "from_symbol": "EUR",
            "to_symbol":   "USD",
            "outputsize":  outputsize,
            "apikey":      api_key,
        },
        timeout=30,
    )
    response.raise_for_status()
    df = _parse_av_response(response.json())

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df.to_csv(save_path, index=False)
    return df


def fetch_and_merge(
    save_path: str = "data/EURUSD.csv",
    outputsize: str = "compact",
) -> pd.DataFrame:
    """
    Fetch the latest bars and MERGE them with the existing CSV.

    - Uses 'compact' (last 100 bars) by default — faster and within free tier limits.
    - Existing rows are kept; new rows are appended; duplicates are removed.
    - Falls back gracefully if the existing file is missing.

    Returns the full merged DataFrame.
    """
    new_df = fetch_eurusd_from_alpha_vantage(outputsize=outputsize, save_path=None)

    if os.path.exists(save_path):
        try:
            existing = pd.read_csv(save_path, parse_dates=["Date"])
            merged = (
                pd.concat([existing, new_df], ignore_index=True)
                .drop_duplicates(subset=["Date"])
                .sort_values("Date")
                .reset_index(drop=True)
            )
        except Exception:
            merged = new_df  # corrupt CSV — just use fresh data
    else:
        merged = new_df

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    merged.to_csv(save_path, index=False)
    return merged
