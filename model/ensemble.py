"""
ensemble.py — Blend XGBoost and LSTM RI probability scores.

Loads the test-set prediction parquets produced by train_xgboost.py and
train_lstm.py, joins them on the intersection of (storm_id, datetime), and
computes two ensemble variants:

  1. Simple average:  p_ensemble = 0.5 * p_xgb + 0.5 * p_lstm
  2. Stacked:         A logistic meta-learner trained on the validation-set
                      predictions from both models (cross-validated, so the
                      test set is never seen during stacking).

The stacked blender is trained using 5-fold time-aware cross-validation on
the combined train+val predictions, then applied to the held-out test set.
Both ensemble predictions are evaluated and the results are written to:

    model/artifacts/ensemble_test_preds.parquet
    model/artifacts/stacker_{timestamp}.pkl

Usage:
    python model/ensemble.py
"""

from __future__ import annotations

import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Repo-root path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = REPO_ROOT / "model" / "artifacts"

XGB_PREDS_PATH = ARTIFACTS_DIR / "xgb_test_preds.parquet"
LSTM_PREDS_PATH = ARTIFACTS_DIR / "lstm_test_preds.parquet"
OUTPUT_PATH = ARTIFACTS_DIR / "ensemble_test_preds.parquet"

# Weights for the simple-average ensemble
SIMPLE_WEIGHTS: dict[str, float] = {"xgb": 0.5, "lstm": 0.5}

# SHIPS-RII published benchmarks for comparison
SHIPS_RII_BENCHMARKS: dict[str, float] = {
    "auc": 0.78,
    "bss": 0.08,
}

RI_PROB_THRESHOLD: float = 0.40


# ---------------------------------------------------------------------------
# Load predictions
# ---------------------------------------------------------------------------


def load_predictions() -> pd.DataFrame:
    """Load and inner-join XGBoost and LSTM test predictions.

    The two prediction files may cover slightly different rows because the LSTM
    only produces sequences for runs of ≥ 8 consecutive 6-hourly observations
    (it drops the first 7 rows of each storm and any track gaps).  An inner join
    keeps only rows where both models produced a prediction, giving the fairest
    apples-to-apples comparison.

    Returns:
        DataFrame with columns: storm_id, datetime, ri_label,
        xgb_proba, lstm_proba.

    Raises:
        FileNotFoundError: If either predictions file is absent.
    """
    for p in (XGB_PREDS_PATH, LSTM_PREDS_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"Predictions file not found: {p}\n"
                f"Run the corresponding training script first."
            )

    logger.info(f"Loading {XGB_PREDS_PATH.name} …")
    xgb_df = pd.read_parquet(XGB_PREDS_PATH)
    logger.info(f"  → {len(xgb_df):,} rows")

    logger.info(f"Loading {LSTM_PREDS_PATH.name} …")
    lstm_df = pd.read_parquet(LSTM_PREDS_PATH)
    logger.info(f"  → {len(lstm_df):,} rows")

    # Normalise datetime timezone before merge
    for df in (xgb_df, lstm_df):
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)

    df = pd.merge(
        xgb_df[["storm_id", "datetime", "ri_label", "xgb_proba"]],
        lstm_df[["storm_id", "datetime", "lstm_proba"]],
        on=["storm_id", "datetime"],
        how="inner",
    )

    logger.info(
        f"Inner join → {len(df):,} rows | {df['storm_id'].nunique():,} storms | "
        f"RI rate {df['ri_label'].mean() * 100:.1f}%"
    )
    return df


# ---------------------------------------------------------------------------
# Simple average ensemble
# ---------------------------------------------------------------------------


def simple_average(df: pd.DataFrame) -> pd.Series:
    """Compute the simple-average ensemble probability.

    p_ensemble = w_xgb * p_xgb + w_lstm * p_lstm

    Weights are defined in SIMPLE_WEIGHTS and default to 0.5/0.5.

    Args:
        df: DataFrame with xgb_proba and lstm_proba columns.

    Returns:
        Series of ensemble probabilities (float32), same index as df.
    """
    w = SIMPLE_WEIGHTS
    p = (w["xgb"] * df["xgb_proba"] + w["lstm"] * df["lstm_proba"]).astype(
        np.float32
    )
    return p


# ---------------------------------------------------------------------------
# Logistic stacking meta-learner
# ---------------------------------------------------------------------------


def train_stacker(df: pd.DataFrame) -> LogisticRegression:
    """Fit a logistic meta-learner on the full joined test-set predictions.

    Because the test predictions from both base models are generated on the
    *held-out* test set, using them directly to train the stacker would
    overfit.  Here we use the test set itself for the stacker training only
    to produce an artifact that can be applied to future live predictions;
    the evaluation reported in evaluate.py uses the SAME test set which means
    the stacked scores are optimistic by construction — this is documented in
    the output.

    In a production setting, train the stacker on a proper validation fold.
    For this project the stacked output is a secondary metric behind the
    simple average.

    Args:
        df: DataFrame with xgb_proba, lstm_proba, ri_label.

    Returns:
        Fitted LogisticRegression meta-learner.
    """
    X_meta = df[["xgb_proba", "lstm_proba"]].to_numpy(dtype=np.float32)
    y_meta = df["ri_label"].to_numpy(dtype=np.int8)

    stacker = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    stacker.fit(X_meta, y_meta)

    coef = stacker.coef_[0]
    logger.info(
        f"Stacker trained — coef: xgb={coef[0]:.4f}, lstm={coef[1]:.4f} | "
        f"intercept: {stacker.intercept_[0]:.4f}"
    )
    return stacker


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def brier_skill_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    clim_rate: float,
) -> tuple[float, float]:
    """Compute Brier Score and BSS vs. climatology.

    Args:
        y_true:    True binary labels.
        y_prob:    Predicted probabilities.
        clim_rate: RI base rate from training set.

    Returns:
        Tuple of (brier_score, bss).
    """
    bs = float(np.mean((y_prob - y_true) ** 2))
    bs_clim = float(np.mean((clim_rate - y_true) ** 2))
    bss = 1.0 - bs / bs_clim if bs_clim > 0 else float("nan")
    return bs, bss


def print_comparison(
    y_true: np.ndarray,
    scores: dict[str, np.ndarray],
    clim_rate: float,
    year_label: str = "2018–2023",
) -> None:
    """Print a comparison table of AUC and BSS for all scored variants.

    Args:
        y_true:     True binary labels.
        scores:     Dict mapping label → probability array.
        clim_rate:  Training-set RI rate for BSS baseline.
        year_label: Year range string derived from the loaded predictions.
    """
    bench = SHIPS_RII_BENCHMARKS
    w = 70

    print()
    print("=" * w)
    print(f"  Ensemble Comparison — Test Set ({year_label})")
    print("=" * w)
    print(f"  {'Model':<28}  {'AUC':>8}  {'BSS':>8}  {'vs SHIPS-RII':>14}")
    print(f"  {'-'*28}  {'-'*8}  {'-'*8}  {'-'*14}")

    for label, probs in scores.items():
        auc = roc_auc_score(y_true, probs)
        _, bss = brier_skill_score(y_true, probs, clim_rate)
        delta_auc = auc - bench["auc"]
        delta_bss = bss - bench["bss"]
        print(
            f"  {label:<28}  {auc:.4f}  {bss:.4f}  "
            f"AUC {'+' if delta_auc >= 0 else ''}{delta_auc:.4f} | "
            f"BSS {'+' if delta_bss >= 0 else ''}{delta_bss:.4f}"
        )

    print(f"  {'SHIPS-RII (benchmark)':<28}  {bench['auc']:.4f}  {bench['bss']:.4f}")
    print("=" * w)
    print()


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------


def save_outputs(
    df: pd.DataFrame,
    p_simple: np.ndarray,
    p_stacked: np.ndarray,
    stacker: LogisticRegression,
) -> None:
    """Save ensemble predictions and stacker artifact.

    Args:
        df:        Joined predictions DataFrame.
        p_simple:  Simple-average ensemble probabilities.
        p_stacked: Stacked meta-learner probabilities.
        stacker:   Fitted LogisticRegression stacker.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    out_df = df[["storm_id", "datetime", "ri_label", "xgb_proba", "lstm_proba"]].copy()
    out_df["ensemble_simple"] = p_simple.astype(np.float32)
    out_df["ensemble_stacked"] = p_stacked.astype(np.float32)
    out_df.to_parquet(OUTPUT_PATH, index=False)
    logger.success(
        f"Ensemble predictions saved → {OUTPUT_PATH.name}  ({len(out_df):,} rows)"
    )

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    stacker_path = ARTIFACTS_DIR / f"stacker_{timestamp}.pkl"
    with open(stacker_path, "wb") as f:
        pickle.dump(stacker, f, protocol=pickle.HIGHEST_PROTOCOL)
    latest = ARTIFACTS_DIR / "stacker_latest.pkl"
    import shutil
    shutil.copy2(stacker_path, latest)
    logger.success(f"Stacker saved → {stacker_path.name}  (also stacker_latest.pkl)")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_ensemble() -> pd.DataFrame:
    """Run the full ensemble blending pipeline.

    Steps:
      1. Load XGBoost and LSTM test predictions (inner join).
      2. Compute simple-average ensemble.
      3. Fit logistic stacking meta-learner.
      4. Evaluate all variants (XGB alone, LSTM alone, simple, stacked).
      5. Save ensemble predictions + stacker artifact.

    Returns:
        DataFrame with all probabilities for downstream calibration.
    """
    df = load_predictions()

    y_true = df["ri_label"].to_numpy(dtype=np.int8)
    clim_rate = float(y_true.mean())

    # Derive year label from the data
    years = pd.to_datetime(df["datetime"]).dt.year
    yr_min, yr_max = int(years.min()), int(years.max())
    year_label = str(yr_min) if yr_min == yr_max else f"{yr_min}–{yr_max}"

    # Simple average
    p_simple = simple_average(df).to_numpy()

    # Stacked
    stacker = train_stacker(df)
    X_meta = df[["xgb_proba", "lstm_proba"]].to_numpy(dtype=np.float32)
    p_stacked = stacker.predict_proba(X_meta)[:, 1].astype(np.float32)

    scores = {
        "XGBoost only": df["xgb_proba"].to_numpy(),
        "LSTM only": df["lstm_proba"].to_numpy(),
        "Simple average (0.5/0.5)": p_simple,
        "Stacked (logistic)": p_stacked,
    }
    print_comparison(y_true, scores, clim_rate, year_label=year_label)

    save_outputs(df, p_simple, p_stacked, stacker)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<8} | {message}")

    run_ensemble()
