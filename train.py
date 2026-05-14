import os
from config.settings import (
    DATA_PATH,
    WF_RESULTS_PATH,
    DEFAULT_SIGNAL_THRESHOLD,
    DEFAULT_SPREAD_COST,
    DEFAULT_WF_TRAIN_SIZE,
    DEFAULT_WF_TEST_SIZE,
    DEFAULT_WF_STEP_SIZE,
)
from src.data_loader import load_forex_data
from src.features import add_features, FEATURE_COLUMNS_V2
from src.model import train_random_forest, evaluate_model, save_model_bundle
from src.train_pipeline import walk_forward_validation
from src.logger import get_logger

logger = get_logger("train")


def main():
    logger.info("Loading data...")
    df = load_forex_data(DATA_PATH)
    df = add_features(df)

    logger.info("Running walk-forward validation...")
    wf_df = walk_forward_validation(
        df=df,
        feature_columns=FEATURE_COLUMNS_V2,
        train_size=DEFAULT_WF_TRAIN_SIZE,
        test_size=DEFAULT_WF_TEST_SIZE,
        step_size=DEFAULT_WF_STEP_SIZE,
        threshold=DEFAULT_SIGNAL_THRESHOLD,
        spread_cost=DEFAULT_SPREAD_COST,
    )

    os.makedirs(os.path.dirname(WF_RESULTS_PATH), exist_ok=True)
    wf_df.to_csv(WF_RESULTS_PATH, index=False)

    # --- Train final model on 80% of data; hold out last 20% for a real test ---
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]

    logger.info(f"Training final model on {len(train_df)} rows, testing on {len(test_df)} rows...")
    X_train = train_df[FEATURE_COLUMNS_V2]
    y_train = train_df["target"]
    X_test  = test_df[FEATURE_COLUMNS_V2]
    y_test  = test_df["target"]

    model = train_random_forest(X_train, y_train)
    _, _, train_metrics = evaluate_model(model, X_train, y_train)
    _, _, test_metrics  = evaluate_model(model, X_test, y_test)

    metadata = {
        "model_version": "v2",
        "feature_set": "FEATURE_COLUMNS_V2",
        "rows_total": len(df),
        "rows_train": len(train_df),
        "rows_test": len(test_df),
        "accuracy_train": train_metrics["accuracy"],
        "accuracy_test": test_metrics["accuracy"],
        "walk_forward_mean_accuracy": float(wf_df["accuracy"].mean()) if not wf_df.empty else None,
        "walk_forward_mean_strategy_return": float(wf_df["strategy_return"].mean()) if not wf_df.empty else None,
        "walk_forward_mean_profit_factor": float(wf_df["profit_factor"].replace([float("inf")], 999).mean())
                                           if not wf_df.empty and "profit_factor" in wf_df else None,
        "walk_forward_mean_expectancy": float(wf_df["expectancy"].mean())
                                        if not wf_df.empty and "expectancy" in wf_df else None,
        "walk_forward_mean_sharpe": float(wf_df["sharpe"].mean())
                                    if not wf_df.empty and "sharpe" in wf_df else None,
        "walk_forward_mean_exposure": float(wf_df["exposure"].mean())
                                      if not wf_df.empty and "exposure" in wf_df else None,
        "walk_forward_profitable_splits": int((wf_df["strategy_return"] > 0).sum()) if not wf_df.empty else None,
        "walk_forward_total_splits": len(wf_df),
        "feature_columns": FEATURE_COLUMNS_V2,
        "feature_count": len(FEATURE_COLUMNS_V2),
        "rf_max_depth": 5,
        "rf_min_samples_leaf": 20,
        "rf_n_estimators": 200,
        "signal_threshold": DEFAULT_SIGNAL_THRESHOLD,
    }

    save_model_bundle(model, FEATURE_COLUMNS_V2, metadata)

    logger.info("Training complete.")

    print("\n=== Training Results ===")
    print(f"  Train accuracy (in-sample) : {train_metrics['accuracy']:.4f}")
    print(f"  Test  accuracy (held-out)  : {test_metrics['accuracy']:.4f}")
    print(f"  Overfitting gap            : {train_metrics['accuracy'] - test_metrics['accuracy']:.4f}")
    print(f"\n=== Walk-Forward Results ({len(wf_df)} splits) ===")
    print(f"  Mean accuracy              : {wf_df['accuracy'].mean():.4f}")
    print(f"  Mean strategy return       : {wf_df['strategy_return'].mean():.4f}")
    print(f"  Profitable splits          : {(wf_df['strategy_return'] > 0).sum()} / {len(wf_df)}")
    print(f"\nModel saved to: {os.path.abspath('models/model.pkl')}")


if __name__ == "__main__":
    main()
