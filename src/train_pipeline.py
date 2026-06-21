import pandas as pd
from src.features import FEATURE_COLUMNS_V2
from src.model import train_random_forest, evaluate_model
from src.backtest import run_backtest


def walk_forward_validation(
    df: pd.DataFrame,
    feature_columns=None,
    train_size: int = 300,
    test_size: int = 100,
    step_size: int = 100,
    threshold: float = 0.55,
    spread_cost: float = 0.0001,
    slippage_cost: float = 0.0,
    swap_cost_per_day: float = 0.0,
):
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS_V2

    results = []
    start = 0
    split_id = 0

    while start + train_size + test_size <= len(df):
        train_df = df.iloc[start:start + train_size]
        test_df  = df.iloc[start + train_size:start + train_size + test_size]

        X_train = train_df[feature_columns]
        y_train = train_df["target"]
        X_test  = test_df[feature_columns]
        y_test  = test_df["target"]

        model = train_random_forest(X_train, y_train)
        preds, probas, metrics = evaluate_model(model, X_test, y_test)
        _, bt = run_backtest(
            test_df=test_df,
            predictions=preds,
            probas=probas,
            threshold=threshold,
            spread_cost=spread_cost,
            allow_short=True,
            slippage_cost=slippage_cost,
            swap_cost_per_day=swap_cost_per_day,
        )

        results.append({
            "split_id": split_id,
            "train_start": str(train_df["Date"].iloc[0]),
            "train_end":   str(train_df["Date"].iloc[-1]),
            "test_start":  str(test_df["Date"].iloc[0]),
            "test_end":    str(test_df["Date"].iloc[-1]),
            "accuracy": metrics["accuracy"],
            "strategy_return": bt["total_strategy_return"],
            "market_return":   bt["total_market_return"],
            "win_rate":        bt["win_rate"],
            "max_drawdown":    bt["max_drawdown"],
            "num_trades":      bt["num_trades"],
            "trade_entries":   bt["trade_entries"],
            "profit_factor":   bt["profit_factor"],
            "expectancy":      bt["expectancy"],
            "sharpe":          bt["sharpe"],
            "sortino":         bt["sortino"],
            "exposure":        bt["exposure"],
            # ── Net of realistic cost (spread×1.5 + slippage + overnight swap) ──
            "net_strategy_return": bt["net_total_strategy_return"],
            "net_win_rate":        bt["net_win_rate"],
            "net_max_drawdown":    bt["net_max_drawdown"],
            "net_profit_factor":   bt["net_profit_factor"],
            "net_expectancy":      bt["net_expectancy"],
            "net_sharpe":          bt["net_sharpe"],
            "avg_realistic_trade_cost": bt["avg_realistic_trade_cost"],
        })

        start += step_size
        split_id += 1

    return pd.DataFrame(results)
