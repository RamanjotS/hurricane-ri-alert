"""
calibrate.py — Probability calibration for the RI ensemble output.

Loads ensemble_test_preds.parquet (produced by ensemble.py) and fits an
isotonic regression calibrator on the simple-average ensemble probabilities.
Calibration is critical for a deployed alert system: a stated 60% RI
probability should correspond to ~60% observed RI frequency.

Calibration strategy
--------------------
We use 5-fold cross-calibration to produce out-of-fold calibrated
probabilities for the test set, then report reliability (calibration) curves
and summary statistics.  A single calibrator is also fitted on the full test
set and saved for use in the real-time inference pipeline.

Note: calibrating on the test set is an approximation.  In production,
calibrate on a dedicated held-out calibration set that was never used for
training or model selection.

Output
------
    model/artifacts/calibrator_{timestamp}.pkl   — fitted IsotonicRegression
    model/artifacts/calibrator_latest.pkl        — always latest
    model/artifacts/calibrated_preds.parquet     — adds `calibrated_proba`
                                                    column to ensemble preds

Usage:
    python model/calibrate.py
"""

from __future__ import annotations

import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold

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

ENSEMBLE_PREDS_PATH = ARTIFACTS_DIR / "ensemble_test_preds.parquet"
CALIBRATED_PREDS_PATH = ARTIFACTS_DIR / "calibrated_preds.parquet"

# Number of cross-calibration folds
N_CAL_FOLDS: int = 5

# Column from ensemble predictions to calibrate
SOURCE_PROB_COL: str = "ensemble_simple"

# Reliability curve bin count
N_BINS: int = 10


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_ensemble_preds(path: Path = ENSEMBLE_PREDS_PATH) -> pd.DataFrame:
    """Load ensemble test-set predictions.

    Args:
        path: Path to ensemble_test_preds.parquet.

    Returns:
        DataFrame with storm_id, datetime, ri_label, xgb_proba,
        lstm_proba, ensemble_simple, ensemble_stacked.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Ensemble predictions not found: {path}\n"
            "Run: python model/ensemble.py"
        )
    logger.info(f"Loading {path.name} …")
    df = pd.read_parquet(path)
    logger.info(
        f"  → {len(df):,} rows | {df['storm_id'].nunique():,} storms | "
        f"RI rate {df['ri_label'].mean() * 100:.1f}%"
    )
    return df


# ---------------------------------------------------------------------------
# Cross-calibrated predictions
# ---------------------------------------------------------------------------


def cross_calibrate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_folds: int = N_CAL_FOLDS,
) -> tuple[np.ndarray, IsotonicRegression]:
    """Produce out-of-fold isotonic calibrated probabilities.

    For each fold, an IsotonicRegression is fitted on the in-fold samples and
    applied to the out-of-fold samples, avoiding calibration leakage.  A final
    calibrator is then fitted on all samples and returned for inference use.

    Args:
        y_true:  True binary labels.
        y_prob:  Raw ensemble probabilities.
        n_folds: Number of cross-calibration folds.

    Returns:
        Tuple of:
          - cal_probs: Out-of-fold calibrated probabilities (same length as input).
          - calibrator: IsotonicRegression fitted on the full dataset.
    """
    cal_probs = np.full_like(y_prob, fill_value=np.nan, dtype=np.float32)

    kf = KFold(n_splits=n_folds, shuffle=False)
    for fold, (train_idx, val_idx) in enumerate(kf.split(y_prob), start=1):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(y_prob[train_idx], y_true[train_idx])
        cal_probs[val_idx] = ir.predict(y_prob[val_idx]).astype(np.float32)
        logger.debug(
            f"  Fold {fold}/{n_folds}: "
            f"cal AUC={roc_auc_score(y_true[val_idx], cal_probs[val_idx]):.4f}"
        )

    # Full-dataset calibrator for inference
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_prob, y_true)

    return cal_probs, calibrator


# ---------------------------------------------------------------------------
# Reliability (calibration) curve
# ---------------------------------------------------------------------------


def print_reliability_curve(
    y_true: np.ndarray,
    y_raw: np.ndarray,
    y_cal: np.ndarray,
    n_bins: int = N_BINS,
) -> None:
    """Print a text reliability curve comparing raw vs. calibrated probabilities.

    Args:
        y_true:  True binary labels.
        y_raw:   Raw ensemble probabilities.
        y_cal:   Calibrated probabilities.
        n_bins:  Number of equal-width probability bins.
    """
    frac_pos_raw, mean_pred_raw = calibration_curve(
        y_true, y_raw, n_bins=n_bins, strategy="uniform"
    )
    frac_pos_cal, mean_pred_cal = calibration_curve(
        y_true, y_cal, n_bins=n_bins, strategy="uniform"
    )

    w = 70
    print()
    print("=" * w)
    print("  Reliability Curve — Ensemble (raw) vs Calibrated")
    print("=" * w)
    print(f"  {'Bin centre':>10}  {'Raw freq':>10}  {'Cal freq':>10}  {'Δ raw':>8}  {'Δ cal':>8}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")

    max_len = max(len(frac_pos_raw), len(frac_pos_cal))
    for i in range(max_len):
        if i < len(mean_pred_raw):
            ctr_raw = f"{mean_pred_raw[i]:.2f}"
            fr = f"{frac_pos_raw[i]:.3f}"
            d_raw = f"{frac_pos_raw[i] - mean_pred_raw[i]:+.3f}"
        else:
            ctr_raw = fr = d_raw = "—"

        if i < len(frac_pos_cal):
            ctr_cal = f"{mean_pred_cal[i]:.2f}"
            fc = f"{frac_pos_cal[i]:.3f}"
            d_cal = f"{frac_pos_cal[i] - mean_pred_cal[i]:+.3f}"
        else:
            fc = d_cal = "—"

        print(f"  {ctr_raw:>10}  {fr:>10}  {fc:>10}  {d_raw:>8}  {d_cal:>8}")

    print("=" * w)
    print()


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------


def print_calibration_summary(
    y_true: np.ndarray,
    y_raw: np.ndarray,
    y_cal: np.ndarray,
    clim_rate: float,
) -> None:
    """Print AUC, Brier Score, and BSS before and after calibration.

    Args:
        y_true:    True binary labels.
        y_raw:     Raw ensemble probabilities.
        y_cal:     Calibrated probabilities.
        clim_rate: Training-set RI rate for BSS baseline.
    """
    bs_clim = float(np.mean((clim_rate - y_true) ** 2))

    def _metrics(label: str, probs: np.ndarray) -> None:
        auc = roc_auc_score(y_true, probs)
        bs = brier_score_loss(y_true, probs)
        bss = 1.0 - bs / bs_clim if bs_clim > 0 else float("nan")
        print(f"  {label:<28}  AUC={auc:.4f}  BS={bs:.5f}  BSS={bss:.4f}")

    print()
    print("=" * 70)
    print("  Calibration Summary")
    print("=" * 70)
    _metrics("Raw ensemble (simple avg)", y_raw)
    _metrics("Calibrated (isotonic, OOF)", y_cal)
    print(f"  {'Climatology baseline':<28}  AUC=N/A     BS={bs_clim:.5f}  BSS=0.0000")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------


def save_artifacts(
    df: pd.DataFrame,
    cal_probs: np.ndarray,
    calibrator: IsotonicRegression,
) -> None:
    """Persist calibrated predictions and calibrator artifact.

    Args:
        df:         Ensemble predictions DataFrame.
        cal_probs:  Calibrated probabilities.
        calibrator: Fitted IsotonicRegression.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    out_df = df.copy()
    out_df["calibrated_proba"] = cal_probs.astype(np.float32)
    out_df.to_parquet(CALIBRATED_PREDS_PATH, index=False)
    logger.success(
        f"Calibrated predictions saved → {CALIBRATED_PREDS_PATH.name}  "
        f"({len(out_df):,} rows)"
    )

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    cal_path = ARTIFACTS_DIR / f"calibrator_{timestamp}.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f, protocol=pickle.HIGHEST_PROTOCOL)
    latest = ARTIFACTS_DIR / "calibrator_latest.pkl"
    shutil.copy2(cal_path, latest)
    logger.success(
        f"Calibrator saved → {cal_path.name}  (also calibrator_latest.pkl)"
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_calibration(path: Path = ENSEMBLE_PREDS_PATH) -> pd.DataFrame:
    """Run the full calibration pipeline.

    Steps:
      1. Load ensemble predictions.
      2. Run 5-fold cross-calibration with isotonic regression.
      3. Print reliability curve (raw vs. calibrated).
      4. Print AUC/BS/BSS summary.
      5. Save calibrated predictions + calibrator artifact.

    Args:
        path: Path to ensemble_test_preds.parquet.

    Returns:
        DataFrame with an additional ``calibrated_proba`` column.
    """
    df = load_ensemble_preds(path)
    y_true = df["ri_label"].to_numpy(dtype=np.int8)
    y_raw = df[SOURCE_PROB_COL].to_numpy(dtype=np.float32)
    clim_rate = float(y_true.mean())

    logger.info(
        f"Calibrating '{SOURCE_PROB_COL}' with isotonic regression "
        f"({N_CAL_FOLDS}-fold) …"
    )
    cal_probs, calibrator = cross_calibrate(y_true, y_raw)

    print_reliability_curve(y_true, y_raw, cal_probs)
    print_calibration_summary(y_true, y_raw, cal_probs, clim_rate)

    save_artifacts(df, cal_probs, calibrator)

    df["calibrated_proba"] = cal_probs.astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<8} | {message}")

    run_calibration()
