import os
import json
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)
from config.settings import MODEL_PATH, METADATA_PATH

# ── Calibration guard ────────────────────────────────────────────────────────
# CalibratedClassifierCV uses k-fold cross-validation to fit the sigmoid
# mapping.  With cv=5 a 300-row window yields ~240 training / 60 validation
# samples per fold — workable.  Below 200 rows the folds become too small and
# calibration is unreliable, so we fall back to the uncalibrated RF.
MIN_CALIBRATION_SAMPLES = 200


def train_random_forest(X_train, y_train):
    """
    Regularised Random Forest (uncalibrated).

    Kept for backward compatibility and as the base-estimator template used by
    train_random_forest_calibrated().  Prefer the calibrated version for all
    production training paths.

    Key regularisation vs original:
      - max_depth reduced from 10 → 5  (limits tree complexity)
      - min_samples_leaf raised from 5 → 20 (forces more evidence per leaf)
      - min_samples_split raised from 10 → 40 (harder to split a node)
      - n_estimators reduced from 300 → 200 (faster, similar variance)

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


def train_random_forest_calibrated(X_train, y_train):
    """
    Train a calibrated Random Forest using Platt scaling (sigmoid method).

    Why calibration matters:
    Random Forests are known to produce over-confident predicted probabilities —
    a model that outputs 0.68 may not actually be right 68% of the time.
    CalibratedClassifierCV corrects this by fitting a sigmoid function (logistic
    regression) on held-out fold predictions, mapping raw RF outputs to
    genuine probability estimates.

    Important implementation detail:
    CalibratedClassifierCV with cv=5 re-fits the base estimator internally on
    EACH fold.  The base_estimator passed in serves only as a hyperparameter
    template; it is never used directly for inference.  Do NOT pre-train the
    base estimator before passing it in — doing so has no effect on predictions.

    Why sigmoid over isotonic regression:
    Isotonic regression is non-parametric and requires substantially more data
    to avoid overfitting its own calibration.  At the 300–600 row walk-forward
    window sizes used by ForexChautari, sigmoid (Platt scaling) is the correct
    choice.

    Expected behaviour:
    Calibration COMPRESSES probabilities toward 0.5 for a model with ~51–55%
    accuracy.  Seeing fewer signals cross the 0.55 trade threshold after
    calibration is correct — it means the system is no longer acting on
    overconfident RF outputs.
    """
    base = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=40,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    if len(X_train) < MIN_CALIBRATION_SAMPLES:
        # Not enough rows to fit a reliable sigmoid mapping on each fold.
        # Fall back to the uncalibrated RF rather than producing garbage
        # calibration from too-small folds.
        base.fit(X_train, y_train)
        return base

    # sigmoid = Platt scaling; cv=5 re-fits base on each of 5 folds internally.
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on held-out data.

    Returns (predictions, probability_array, metrics_dict).

    metrics_dict now includes 'brier_score':
      - Brier score = mean squared error between predicted probability and
        actual outcome label.  Lower is better.
      - Perfect model  = 0.0
      - Coin flip (50% always) ≈ 0.25
      - Range for a 51–55% forex model after calibration: ~0.20–0.23.

    After Platt scaling, Brier score should improve relative to the raw RF
    because the predicted probabilities are better aligned with true frequencies.
    """
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)
    prob_positive = probas[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "report_text": classification_report(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "brier_score": float(brier_score_loss(y_test, prob_positive)),
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
