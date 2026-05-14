import pandas as pd


def run_realistic_backtest(
    df: pd.DataFrame,
    signal_col: str = "signal",
    initial_balance: float = 10000.0,
    risk_per_trade: float = 0.01,
    stop_loss_pct: float = 0.002,
    take_profit_pct: float = 0.004,
    spread: float = 0.0001,
    slippage: float = 0.00005
):
    data = df.copy().reset_index(drop=True)

    balance = initial_balance
    equity_curve = []
    trades = []

    in_position = False
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    position_size = 0.0
    entry_time = None

    for i in range(len(data) - 1):
        row = data.iloc[i]
        next_row = data.iloc[i + 1]

        equity_curve.append({
            "Date": row["Date"],
            "Balance": balance
        })

        if not in_position and row[signal_col] == 1:
            risk_amount = balance * risk_per_trade
            entry_price = next_row["Open"] + spread + slippage
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)

            stop_distance = entry_price - stop_loss
            if stop_distance <= 0:
                continue

            position_size = risk_amount / stop_distance
            entry_time = next_row["Date"]
            in_position = True
            continue

        if in_position:
            low_price = next_row["Low"]
            high_price = next_row["High"]

            exit_price = None
            exit_reason = None

            if low_price <= stop_loss:
                exit_price = stop_loss - slippage
                exit_reason = "stop_loss"
            elif high_price >= take_profit:
                exit_price = take_profit - slippage
                exit_reason = "take_profit"

            if exit_price is not None:
                pnl = (exit_price - entry_price) * position_size
                balance += pnl

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": next_row["Date"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "position_size": position_size,
                    "pnl": pnl,
                    "balance_after_trade": balance,
                    "exit_reason": exit_reason
                })

                in_position = False
                entry_price = 0.0
                stop_loss = 0.0
                take_profit = 0.0
                position_size = 0.0
                entry_time = None

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)

    if not equity_df.empty:
        equity_df["Peak"] = equity_df["Balance"].cummax()
        equity_df["Drawdown"] = (equity_df["Balance"] - equity_df["Peak"]) / equity_df["Peak"]

    summary = {
        "initial_balance": initial_balance,
        "final_balance": float(balance),
        "total_return": float((balance - initial_balance) / initial_balance),
        "num_trades": int(len(trades_df)),
        "win_rate": float((trades_df["pnl"] > 0).mean()) if not trades_df.empty else 0.0,
        "max_drawdown": float(equity_df["Drawdown"].min()) if not equity_df.empty else 0.0
    }

    return equity_df, trades_df, summary
