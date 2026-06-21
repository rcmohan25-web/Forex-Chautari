import numpy as np
import pandas as pd
from config.settings import REALISTIC_SPREAD_MULTIPLIER


def run_backtest(
    test_df: pd.DataFrame,
    predictions,
    probas=None,
    threshold=0.55,
    spread_cost=0.0001,
    allow_short: bool = False,
    slippage_cost: float = 0.0,
    swap_cost_per_day: float = 0.0,
    spread_multiplier: float = REALISTIC_SPREAD_MULTIPLIER,
):
    """
    Backtest a signal series against held-out price data.

    Returns two parallel views in `results`:
      - gross-of-realistic-cost (existing keys, e.g. `strategy_return`,
        `profit_factor`) — only the raw `spread_cost` is deducted. Kept
        for backward compatibility; DO NOT use these to decide whether a
        model has a real edge, they overstate it.
      - net-of-realistic-cost (new `net_*` keys) — additionally deducts:
          * a realistic spread = spread_cost × spread_multiplier (default 1.5x)
          * per-trade slippage (slippage_cost, same price units as spread_cost)
          * overnight swap/holding cost charged for every day a position
            stays open (swap_cost_per_day, same price units as spread_cost)

    The net_* fields are what should be used to judge a tradable edge.
    """
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

    # ── Realistic, net-of-cost view ──────────────────────────────────────────
    realistic_spread_cost = spread_cost * spread_multiplier
    df["realistic_trade_cost"] = trade_change * (realistic_spread_cost + slippage_cost)
    df["swap_cost"] = np.where(df["signal"] != 0, swap_cost_per_day, 0.0)
    df["net_strategy_return"] = (
        df["gross_strategy_return"] - df["realistic_trade_cost"] - df["swap_cost"]
    )

    df = df.dropna().reset_index(drop=True)

    df["cum_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["cum_market"] = (1 + df["market_return"]).cumprod()
    df["cum_net_strategy"] = (1 + df["net_strategy_return"]).cumprod()

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

    # ── Net-of-cost stats (the realistic numbers) ────────────────────────────
    net_rolling_max = df["cum_net_strategy"].cummax()
    net_drawdown = (df["cum_net_strategy"] - net_rolling_max) / net_rolling_max
    net_active_returns = df.loc[df["signal"] != 0, "net_strategy_return"]
    net_gains = net_active_returns[net_active_returns > 0].sum()
    net_losses = net_active_returns[net_active_returns < 0].sum()
    net_downside = net_active_returns[net_active_returns < 0]
    net_sharpe = (
        float(net_active_returns.mean() / net_active_returns.std() * np.sqrt(252))
        if len(net_active_returns) > 1 and net_active_returns.std() > 0 else 0.0
    )

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

        # ── Net of realistic cost: spread×multiplier + slippage + overnight swap ──
        "net_total_strategy_return": float(df["cum_net_strategy"].iloc[-1] - 1),
        "net_win_rate": float((df["net_strategy_return"] > 0).mean()),
        "net_max_drawdown": float(net_drawdown.min()),
        "net_profit_factor": float(net_gains / abs(net_losses)) if net_losses < 0 else (float("inf") if net_gains > 0 else 0.0),
        "net_expectancy": float(net_active_returns.mean()) if len(net_active_returns) else 0.0,
        "net_sharpe": net_sharpe,
        "avg_realistic_trade_cost": float(df["realistic_trade_cost"].mean() + df["swap_cost"].mean()),
    }

    return df, results
