import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config.settings import MODEL_PATH, METADATA_PATH


def train_random_forest(X_train, y_train):
    """
    Regularised Random Forest.

    Key changes vs the original:
      - max_depth reduced from 10 → 5  (limits tree complexity)
      - min_samples_leaf raised from 5 → 20 (forces more evidence per leaf)
      - min_samples_split raised from 10 → 40 (harder to split a node)
      - n_estimators reduced from 300 → 200 (faster, similar variance)
      - max_features='sqrt' is already the RF default — kept explicit

    Why: the original model reached 87% in-sample accuracy but ~50%
    out-of-sample (random). The trees were deep enough to memorise
    individual candles. These tighter settings force the model to learn
    broader patterns that generalise better.

    Reality check: EUR/USD daily direction is close to a 50/50 coin flip.
    Expect walk-forward accuracy in the 51–55% range — any consistent
    edge above 53% with positive expected return is genuinely useful.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=40,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "report_text": classification_report(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    return preds, probas, metrics


def save_model_bundle(model, feature_columns, metadata: dict):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    full_metadata = {"feature_columns": feature_columns, **metadata}
    with open(METADATA_PATH, "w") as f:
        json.dump(full_metadata, f, indent=2)


def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Run training first.")
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Model metadata not found. Run training first.")
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return model, metadata


def predict_latest(model, df: pd.DataFrame, feature_columns):
    latest = df.iloc[-1:][feature_columns]
    pred = int(model.predict(latest)[0])
    prob_up = float(model.predict_proba(latest)[0][1])
    return pred, prob_up
