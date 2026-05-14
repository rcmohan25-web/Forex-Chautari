import numpy as np
import pandas as pd


def run_backtest(
    test_df: pd.DataFrame,
    predictions,
    probas=None,
    threshold=0.55,
    spread_cost=0.0001,
    allow_short: bool = False,
):
    df = test_df.copy().reset_index(drop=True)
    df["prediction"] = predictions

    if probas is not None:
        df["prob_up"] = probas[:, 1]
        if allow_short:
            df["signal"] = np.select(
                [df["prob_up"] >= threshold, df["prob_up"] <= 1 - threshold],
                [1, -1],
                default=0,
            )
        else:
            df["signal"] = np.where(df["prob_up"] >= threshold, 1, 0)
    else:
        df["signal"] = np.where(df["prediction"] == 1, 1, -1 if allow_short else 0)

    df["next_return"] = df["Close"].shift(-1) / df["Close"] - 1

    trade_change = df["signal"].diff().abs().fillna(df["signal"])
    df["trade_cost"] = trade_change * spread_cost

    df["gross_strategy_return"] = df["signal"] * df["next_return"]
    df["strategy_return"] = df["gross_strategy_return"] - df["trade_cost"]
    df["market_return"] = df["next_return"]

    df = df.dropna().reset_index(drop=True)

    df["cum_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["cum_market"] = (1 + df["market_return"]).cumprod()

    rolling_max = df["cum_strategy"].cummax()
    drawdown = (df["cum_strategy"] - rolling_max) / rolling_max
    active_returns = df.loc[df["signal"] != 0, "strategy_return"]
    gains = active_returns[active_returns > 0].sum()
    losses = active_returns[active_returns < 0].sum()
    downside = active_returns[active_returns < 0]
    sharpe = (
        float(active_returns.mean() / active_returns.std() * np.sqrt(252))
        if len(active_returns) > 1 and active_returns.std() > 0 else 0.0
    )
    sortino = (
        float(active_returns.mean() / downside.std() * np.sqrt(252))
        if len(downside) > 1 and downside.std() > 0 else 0.0
    )
    trade_entries = int((df["signal"].diff().fillna(df["signal"]) != 0).sum())

    results = {
        "total_strategy_return": float(df["cum_strategy"].iloc[-1] - 1),
        "total_market_return": float(df["cum_market"].iloc[-1] - 1),
        "win_rate": float((df["strategy_return"] > 0).mean()),
        "max_drawdown": float(drawdown.min()),
        "num_trades": int((df["signal"] != 0).sum()) if allow_short else int(df["signal"].sum()),
        "trade_entries": trade_entries,
        "avg_trade_cost": float(df["trade_cost"].mean()),
        "profit_factor": float(gains / abs(losses)) if losses < 0 else (float("inf") if gains > 0 else 0.0),
        "expectancy": float(active_returns.mean()) if len(active_returns) else 0.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "exposure": float((df["signal"] != 0).mean()),
        "avg_win": float(active_returns[active_returns > 0].mean()) if (active_returns > 0).any() else 0.0,
        "avg_loss": float(active_returns[active_returns < 0].mean()) if (active_returns < 0).any() else 0.0,
    }

    return df, results
