"""
Train models for all active pairs.

Usage:
    python train_all.py              # train all 4 pairs
    python train_all.py EUR_USD      # train one specific pair
    python train_all.py --fetch      # fetch fresh data first, then train all

Output per pair:
    models/{PAIR}_model.pkl
    models/{PAIR}_metadata.json
    models/{PAIR}_wf_results.csv
"""

import sys
from src.multi_pair_manager import (
    fetch_all_pairs, fetch_pair_data,
    train_all_pairs, train_pair,
    ACTIVE_PAIRS,
)
from src.logger import get_logger

logger = get_logger("train_all")


def print_result(pair: str, result: dict):
    if not result["ok"]:
        print(f"  ❌ {pair}: {result['error']}")
        return
    m = result["metadata"]
    print(f"  ✅ {pair}:")
    print(f"       Train acc : {m['accuracy_train']:.4f}")
    print(f"       Test acc  : {m['accuracy_test']:.4f}")
    print(f"       WF acc    : {m['walk_forward_mean_accuracy']:.4f}")
    if m.get("walk_forward_mean_profit_factor") is not None:
        print(f"       WF PF     : {m['walk_forward_mean_profit_factor']:.2f}")
    if m.get("walk_forward_mean_sharpe") is not None:
        print(f"       WF Sharpe : {m['walk_forward_mean_sharpe']:.2f}")
    print(f"       Gap       : {m['accuracy_train']-m['accuracy_test']:.4f}")
    print(f"       WF profit : {m['walk_forward_profitable_splits']}/{m['walk_forward_total_splits']} splits")


def main():
    args       = sys.argv[1:]
    fetch_flag = "--fetch" in args
    pairs_arg  = [a for a in args if not a.startswith("--")]

    target_pairs = pairs_arg if pairs_arg else ACTIVE_PAIRS

    # Optionally fetch fresh data first
    if fetch_flag:
        print(f"\nFetching data for {target_pairs}...")
        if pairs_arg:
            for p in pairs_arg:
                try:
                    df = fetch_pair_data(p, count=500)
                    print(f"  ✅ {p}: {len(df)} rows")
                except Exception as e:
                    print(f"  ❌ {p}: {e}")
        else:
            fetch_results = fetch_all_pairs(count=500)
            for p, r in fetch_results.items():
                if r["ok"]:
                    print(f"  ✅ {p}: {r['rows']} rows")
                else:
                    print(f"  ❌ {p}: {r['error']}")

    # Train
    print(f"\nTraining models for: {target_pairs}")
    print("-" * 50)

    if len(target_pairs) == 1:
        pair = target_pairs[0]
        try:
            meta   = train_pair(pair)
            result = {"ok": True, "metadata": meta}
        except Exception as e:
            result = {"ok": False, "error": str(e)}
        print_result(pair, result)
    else:
        results = train_all_pairs()
        for pair in target_pairs:
            print_result(pair, results.get(pair, {"ok": False, "error": "not run"}))

    print("\nDone. Models saved to models/")


if __name__ == "__main__":
    main()
