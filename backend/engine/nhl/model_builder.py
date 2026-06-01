"""
Walk-forward XGBoost trainer for NHL home win prediction.

Strategy:
  - Time-based split: 70% train / 15% calibration / 15% validation
  - Platt scaling (LogisticRegression) calibrates XGBoost raw probabilities
  - C hyperparameter grid-searched to minimise Brier score on validation set
  - Final model retrained on all data (train + calibration sets)
  - Saves pkl bundle: {xgb, platt, features, train_end, val_accuracy, val_brier, val_auc, val_ece}
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from db import get_conn
from utils import FEATURES

MODEL_DIR  = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "xgb_nhl_v1.pkl"

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=6,
    gamma=0.2,
    reg_lambda=2.0,
    reg_alpha=0.5,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

_PLATT_C_GRID = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Probability-weighted mean absolute deviation between predicted confidence and observed accuracy."""
    bins    = np.linspace(0, 1, n_bins + 1)
    ece     = 0.0
    n       = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _tune_platt(
    xgb_model,
    X_cal: np.ndarray, y_cal: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> tuple:
    """Grid-search Platt scaling C to minimise Brier score on the validation set."""
    raw_cal = xgb_model.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    raw_val = xgb_model.predict_proba(X_val)[:, 1].reshape(-1, 1)

    best_platt = None
    best_brier = float("inf")

    for c in _PLATT_C_GRID:
        platt = LogisticRegression(C=c, max_iter=1000, random_state=42)
        platt.fit(raw_cal, y_cal)
        cal_val = np.clip(platt.predict_proba(raw_val)[:, 1], 0.0, 1.0)
        brier   = float(brier_score_loss(y_val, cal_val))
        if brier < best_brier:
            best_brier = brier
            best_platt = platt

    return best_platt, best_brier


def _load_features() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql(
        f"SELECT game_id, {', '.join(FEATURES)}, home_win FROM features "
        "ORDER BY game_id",
        conn,
    )
    conn.close()
    return df


def build_and_train_model() -> dict:
    MODEL_DIR.mkdir(exist_ok=True)

    df = _load_features()
    df = df.dropna(subset=FEATURES + ["home_win"]).copy()

    if len(df) < 200:
        print(f"Insufficient data ({len(df)} rows) — need at least 200 to train.")
        return {}

    print(f"Training on {len(df)} completed games...")

    X = df[FEATURES].values.astype(float)
    y = df["home_win"].values.astype(int)

    n         = len(df)
    train_end = int(n * 0.70)
    cal_end   = int(n * 0.85)

    X_train, y_train = X[:train_end],        y[:train_end]
    X_cal,   y_cal   = X[train_end:cal_end], y[train_end:cal_end]
    X_val,   y_val   = X[cal_end:],          y[cal_end:]

    # ── Train XGBoost ────────────────────────────────────────────────────────
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # ── Brier-optimised Platt calibration ────────────────────────────────────
    best_platt, _ = _tune_platt(xgb, X_cal, y_cal, X_val, y_val)

    # ── Evaluate on validation set ────────────────────────────────────────────
    raw_val = xgb.predict_proba(X_val)[:, 1].reshape(-1, 1)
    cal_val = np.clip(best_platt.predict_proba(raw_val)[:, 1], 0.0, 1.0)

    preds_bin = (cal_val >= 0.5).astype(int)
    accuracy  = float((preds_bin == y_val).mean())
    brier     = float(brier_score_loss(y_val, cal_val))
    auc       = float(roc_auc_score(y_val, cal_val))
    ece       = _expected_calibration_error(y_val, cal_val)

    print(f"\nValidation results:")
    print(f"  Brier score : {brier:.4f}  ← primary metric")
    print(f"  ECE         : {ece:.4f}  (0.00 = perfect calibration)")
    print(f"  Accuracy    : {accuracy:.4f}")
    print(f"  AUC         : {auc:.4f}")
    print(f"  Home win rate in validation: {y_val.mean():.3f}")

    # ── Retrain final model on train + calibration sets ───────────────────────
    xgb_final = XGBClassifier(**XGB_PARAMS)
    xgb_final.fit(X[:cal_end], y[:cal_end], verbose=False)

    platt_final, _ = _tune_platt(xgb_final, X_cal, y_cal, X_val, y_val)

    train_end_date = str(df.iloc[cal_end - 1]["game_id"])[:8] if len(df) > cal_end else datetime.now().strftime("%Y%m%d")

    bundle = {
        "xgb":          xgb_final,
        "platt":         platt_final,
        "features":      FEATURES,
        "train_end":     train_end_date,
        "val_accuracy":  round(accuracy, 4),
        "val_brier":     round(brier, 4),
        "val_auc":       round(auc, 4),
        "val_ece":       round(ece, 4),
        "n_train":       train_end,
        "n_val":         len(y_val),
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Model saved → {MODEL_PATH}")

    # ── Log to DB ─────────────────────────────────────────────────────────────
    conn = get_conn()
    conn.execute(
        "INSERT INTO model_runs (run_date, train_end, val_accuracy, val_brier, val_auc, n_train, n_val) "
        "VALUES (?,?,?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d"), train_end_date,
         accuracy, brier, auc, train_end, len(y_val)),
    )
    conn.commit()
    conn.close()

    # ── Feature importance ────────────────────────────────────────────────────
    importances = sorted(
        zip(FEATURES, xgb_final.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\nTop 10 feature importances:")
    for feat, imp in importances[:10]:
        print(f"  {feat:<30} {imp:.4f}")

    return bundle


if __name__ == "__main__":
    from db import setup_db
    setup_db()
    build_and_train_model()
