"""
XGBoost + Platt calibration trainer for NBA home win prediction.
Trains two separate models: 'regular' (regular season) and 'playoff'.
Chronological 70/15/15 train/calibration/validation split.
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from db import get_conn, setup_db
from utils import FEATURES

MODEL_DIR = Path(__file__).parent / "models"

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=5,
    gamma=0.1,
    reg_lambda=2.0,
    reg_alpha=0.3,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

_PLATT_C_GRID = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def _tune_platt(xgb, X_cal, y_cal, X_val, y_val):
    raw_cal = xgb.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    raw_val = xgb.predict_proba(X_val)[:, 1].reshape(-1, 1)
    best_platt, best_brier = None, float("inf")
    for c in _PLATT_C_GRID:
        platt = LogisticRegression(C=c, max_iter=1000, random_state=42)
        platt.fit(raw_cal, y_cal)
        proba = np.clip(platt.predict_proba(raw_val)[:, 1], 0, 1)
        brier = brier_score_loss(y_val, proba)
        if brier < best_brier:
            best_brier, best_platt = brier, platt
    return best_platt, best_brier


def build_and_train(game_type: str = "regular") -> dict:
    """
    game_type: 'regular' or 'playoff'
    Returns bundle dict saved to models/xgb_nba_{game_type}.pkl
    """
    MODEL_DIR.mkdir(exist_ok=True)
    setup_db()

    conn = get_conn()
    df   = pd.read_sql(
        f"SELECT {', '.join(FEATURES)}, home_win FROM features "
        "WHERE game_type = ? AND home_win IS NOT NULL "
        "ORDER BY game_date ASC",
        conn, params=(game_type,),
    )
    conn.close()

    df = df.dropna(subset=FEATURES + ["home_win"]).copy()
    min_rows = 200 if game_type == "regular" else 100
    if len(df) < min_rows:
        print(f"Insufficient {game_type} data ({len(df)} rows) — need {min_rows}.")
        return {}

    print(f"Training {game_type} model on {len(df)} games...")

    X = df[FEATURES].values.astype(float)
    y = df["home_win"].values.astype(int)

    n         = len(df)
    train_end = int(n * 0.70)
    cal_end   = int(n * 0.85)

    X_train, y_train = X[:train_end],        y[:train_end]
    X_cal,   y_cal   = X[train_end:cal_end], y[train_end:cal_end]
    X_val,   y_val   = X[cal_end:],          y[cal_end:]

    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    best_platt, _ = _tune_platt(xgb, X_cal, y_cal, X_val, y_val)

    raw_val = xgb.predict_proba(X_val)[:, 1].reshape(-1, 1)
    cal_val = np.clip(best_platt.predict_proba(raw_val)[:, 1], 0, 1)

    accuracy = float((( cal_val >= 0.5).astype(int) == y_val).mean())
    brier    = float(brier_score_loss(y_val, cal_val))
    auc      = float(roc_auc_score(y_val, cal_val))
    ece_val  = _ece(y_val, cal_val)

    print(f"\n  [{game_type.upper()}] Validation results:")
    print(f"    Brier : {brier:.4f}  (0.25 = random)")
    print(f"    ECE   : {ece_val:.4f}")
    print(f"    Acc   : {accuracy:.4f}")
    print(f"    AUC   : {auc:.4f}")
    print(f"    n_train={train_end}  n_val={len(y_val)}")

    # Retrain final model on train + calibration sets
    xgb_final = XGBClassifier(**XGB_PARAMS)
    xgb_final.fit(X[:cal_end], y[:cal_end], verbose=False)
    platt_final, _ = _tune_platt(xgb_final, X_cal, y_cal, X_val, y_val)

    bundle = {
        "xgb":          xgb_final,
        "platt":        platt_final,
        "features":     FEATURES,
        "game_type":    game_type,
        "val_accuracy": round(accuracy, 4),
        "val_brier":    round(brier, 4),
        "val_auc":      round(auc, 4),
        "val_ece":      round(ece_val, 4),
        "n_train":      train_end,
        "n_val":        len(y_val),
    }

    model_path = MODEL_DIR / f"xgb_nba_{game_type}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  Model saved → {model_path}")

    conn = get_conn()
    conn.execute(
        "INSERT INTO model_runs (run_date, game_type, val_accuracy, val_brier, val_auc, n_train, n_val) "
        "VALUES (?,?,?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d"), game_type,
         accuracy, brier, auc, train_end, len(y_val)),
    )
    conn.commit()
    conn.close()

    importances = sorted(
        zip(FEATURES, xgb_final.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print(f"\n  Top 10 features ({game_type}):")
    for feat, imp in importances[:10]:
        print(f"    {feat:<30} {imp:.4f}")

    return bundle


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[2] / ".env")
    build_and_train("regular")
    build_and_train("playoff")
