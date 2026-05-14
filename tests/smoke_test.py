import os
from src.data_loader import load_forex_data
from src.features import add_features


def run_smoke_test():
    path = "data/EURUSD.csv"
    if not os.path.exists(path):
        print("Smoke test skipped: data file not found")
        return

    df = load_forex_data(path)
    df2 = add_features(df)

    assert not df.empty
    assert not df2.empty
    assert "target" in df2.columns
    print("Smoke test passed")


if __name__ == "__main__":
    run_smoke_test()
