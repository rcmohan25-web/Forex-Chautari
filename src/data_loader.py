import os
import pandas as pd


def validate_ohlc(df: pd.DataFrame):
    required = ["Date", "Open", "High", "Low", "Close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if df.empty:
        raise ValueError("Input dataframe is empty")

    if (df["High"] < df["Low"]).any():
        raise ValueError("Found rows where High < Low")

    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Found non-positive OHLC prices")


def load_forex_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)

    if "Volume" not in df.columns:
        df["Volume"] = 0

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    validate_ohlc(df)
    return df
