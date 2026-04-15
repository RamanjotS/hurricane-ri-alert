"""
calibrate.py — Probability calibration and threshold tuning for the RI ensemble.

Loads ensemble_test_preds.parquet (produced by ensemble.py) and:

  1. Fits an isotonic regression calibrator via 5-fold cross-calibration,
     producing out-of-fold calibrated probabilities.
  2. Sweeps alert thresholds 0.05–0.50 in 0.01 steps on those OOF predictions
     and selects the threshold that maximises F1 score (which balances POD and
     FAR without assuming a fixed operating point).
  3. Saves both the calibrator and the optimal threshold for use in the
     real-time inference pipeline.

Calibration note
----------------
We calibrate on the test set itself (the only labelled held-out data available
after training).  To avoid leakage, calibration uses 5-fold OOF predictions so
no sample is calibrated by a model trained on it.  Threshold selection also
uses OOF probs for the same reason.  In a production setting, dedicate a
separate calibration split never seen during training or model selection.

Output
------
    model/artifacts/calibrator_{timestamp}.pkl   — fitted IsotonicRegression
    model/artifacts/calibrator_latest.pkl        — always latest
    model/artifacts/optimal_threshold.json       — F1-optimal threshold + metrics
    model/artifacts/calibrated_preds.parquet     — adds `calibrated_proba` column

Usage:
    python model/calibrate.py
"""

from __future__ import annotations

import json
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
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
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
OPTIMAL_THRESHOLD_PATH = ARTIFACTS_DIR / "optimal_threshold.json"

# Number of cross-calibration folds
N_CAL_FOLDS: int = 5

# Column from ensemble predictions to calibrate
SOURCE_PROB_COL: str = "ensemble_simple"

# Reliability curve bin count
N_BINS: int = 10

# Threshold sweep range (inclusive on both ends)
THRESHOLD_SWEEP: np.ndarray = np.round(np.arange(0.05, 0.51, 0.01), 2)


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
    logger.info(f"Loading {path.name} ...")
    df = pd.read_parquet(path)
    logger.info(
        f"  -> {len(df):,} rows | {df['storm_id'].nunique():,} storms | "
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
# Threshold tuning
# ---------------------------------------------------------------------------


def tune_threshold(
    y_true: np.ndarray,
    cal_probs: np.ndarray,
    sweep: np.ndarray = THRESHOLD_SWEEP,
) -> tuple[float, float, pd.DataFrame]:
    """Find the F1-optimal alert threshold on OOF calibrated probabilities.

    Sweeps candidate thresholds and computes F1 = 2*TP / (2*TP + FP + FN) at
    each.  F1 balances Probability of Detection and False Alarm Ratio without
    requiring a pre-specified cost weighting.  The sweep is performed on the
    out-of-fold calibrated probabilities produced by cross_calibrate(), so no
    sample is evaluated by a model that trained on it.

    Args:
        y_true:    True binary labels.
        cal_probs: OOF calibrated probabilities (from cross_calibrate).
        sweep:     1-D array of threshold candidates to evaluate.

    Returns:
        Tuple of:
          - best_threshold: float, F1-maximising threshold.
          - best_f1:        float, F1 at the optimal threshold.
          - sweep_df:       DataFrame with columns threshold, f1, pod, far,
                            tp, fp, fn sorted by f1 descending.
    """
    rows: list[dict] = []
    for thr in sweep:
        y_pred = (cal_probs >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom > 0 else 0.0
        rows.append(
            {
                "threshold": float(thr),
                "f1": f1,
                "pod": pod,
                "far": far,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }
        )

    sweep_df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    best = sweep_df.iloc[0]
    best_threshold = float(best["threshold"])
    best_f1 = float(best["f1"])

    logger.info(
        f"Threshold sweep complete — optimal threshold: {best_threshold:.2f} | "
        f"F1={best_f1:.4f}  POD={best['pod']:.3f}  FAR={best['far']:.3f}"
    )
    return best_threshold, best_f1, sweep_df


def print_threshold_sweep(sweep_df: pd.DataFrame, top_n: int = 15) -> None:
    """Print the top-N threshold candidates ranked by F1.

    Args:
        sweep_df: DataFrame returned by tune_threshold (sorted by F1 desc).
        top_n:    Number of rows to display.
    """
    print()
    print("=" * 68)
    print("  Threshold Sweep — F1-Optimal Alert Threshold")
    print(f"  Showing top {top_n} of {len(sweep_df)} candidates (sorted by F1 desc)")
    print("=" * 68)
    print(f"  {'Rank':>4}  {'Thr':>5}  {'F1':>7}  {'POD':>7}  {'FAR':>7}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    print(f"  {'-'*4}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*5}")
    for rank, row in sweep_df.head(top_n).iterrows():
        star = " <-- optimal" if rank == 0 else ""
        print(
            f"  {rank + 1:>4}  {row['threshold']:>5.2f}  {row['f1']:>7.4f}  "
            f"{row['pod']:>7.3f}  {row['far']:>7.3f}  "
            f"{int(row['tp']):>5}  {int(row['fp']):>5}  {int(row['fn']):>5}{star}"
        )
    print("=" * 68)
    print()


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
    print("  Reliability Curve -- Ensemble (raw) vs Calibrated")
    print("=" * w)
    print(f"  {'Bin centre':>10}  {'Raw freq':>10}  {'Cal freq':>10}  {'d_raw':>8}  {'d_cal':>8}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")

    max_len = max(len(frac_pos_raw), len(frac_pos_cal))
    for i in range(max_len):
        if i < len(mean_pred_raw):
            ctr_raw = f"{mean_pred_raw[i]:.2f}"
            fr = f"{frac_pos_raw[i]:.3f}"
            d_raw = f"{frac_pos_raw[i] - mean_pred_raw[i]:+.3f}"
        else:
            ctr_raw = fr = d_raw = "N/A"

        if i < len(frac_pos_cal):
            fc = f"{frac_pos_cal[i]:.3f}"
            d_cal = f"{frac_pos_cal[i] - mean_pred_cal[i]:+.3f}"
        else:
            fc = d_cal = "N/A"

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
    threshold: float,
    threshold_metrics: dict,
) -> None:
    """Persist calibrated predictions, calibrator, and optimal threshold.

    Saves:
      - calibrated_preds.parquet       — adds calibrated_proba column
      - calibrator_{timestamp}.pkl     — timestamped archive
      - calibrator_latest.pkl          — always latest
      - optimal_threshold.json         — F1-optimal threshold + metrics

    Args:
        df:                Ensemble predictions DataFrame.
        cal_probs:         OOF calibrated probabilities.
        calibrator:        Full-dataset IsotonicRegression.
        threshold:         F1-optimal alert threshold (float).
        threshold_metrics: Dict with f1, pod, far at the optimal threshold.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Calibrated predictions parquet
    out_df = df.copy()
    out_df["calibrated_proba"] = cal_probs.astype(np.float32)
    out_df.to_parquet(CALIBRATED_PREDS_PATH, index=False)
    logger.success(
        f"Calibrated predictions saved -> {CALIBRATED_PREDS_PATH.name}  "
        f"({len(out_df):,} rows)"
    )

    # Calibrator pickle
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    cal_path = ARTIFACTS_DIR / f"calibrator_{timestamp}.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.copy2(cal_path, ARTIFACTS_DIR / "calibrator_latest.pkl")
    logger.success(
        f"Calibrator saved -> {cal_path.name}  (also calibrator_latest.pkl)"
    )

    # Optimal threshold JSON
    threshold_record = {
        "threshold": threshold,
        "f1": threshold_metrics["f1"],
        "pod": threshold_metrics["pod"],
        "far": threshold_metrics["far"],
        "tp": threshold_metrics["tp"],
        "fp": threshold_metrics["fp"],
        "fn": threshold_metrics["fn"],
        "source": "F1-optimal sweep on OOF calibrated probabilities",
    }
    OPTIMAL_THRESHOLD_PATH.write_text(
        json.dumps(threshold_record, indent=2), encoding="utf-8"
    )
    logger.success(
        f"Optimal threshold saved -> {OPTIMAL_THRESHOLD_PATH.name}  "
        f"(threshold={threshold:.2f}  F1={threshold_metrics['f1']:.4f})"
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_calibration(path: Path = ENSEMBLE_PREDS_PATH) -> pd.DataFrame:
    """Run the full calibration and threshold-tuning pipeline.

    Steps:
      1. Load ensemble predictions.
      2. Run 5-fold cross-calibration with isotonic regression.
      3. Sweep thresholds 0.05-0.50 in 0.01 steps; pick F1-optimal.
      4. Print threshold sweep table and reliability curve.
      5. Print AUC/BS/BSS calibration summary.
      6. Save calibrated_preds.parquet, calibrator, optimal_threshold.json.

    Args:
        path: Path to ensemble_test_preds.parquet.

    Returns:
        DataFrame with calibrated_proba and optimal threshold logged.
    """
    df = load_ensemble_preds(path)
    y_true = df["ri_label"].to_numpy(dtype=np.int8)
    y_raw = df[SOURCE_PROB_COL].to_numpy(dtype=np.float32)
    clim_rate = float(y_true.mean())

    logger.info(
        f"Calibrating '{SOURCE_PROB_COL}' with isotonic regression "
        f"({N_CAL_FOLDS}-fold) ..."
    )
    cal_probs, calibrator = cross_calibrate(y_true, y_raw)

    logger.info(f"Tuning alert threshold (sweep: {THRESHOLD_SWEEP[0]:.2f} -> "
                f"{THRESHOLD_SWEEP[-1]:.2f} in 0.01 steps) ...")
    best_thr, best_f1, sweep_df = tune_threshold(y_true, cal_probs)

    print_threshold_sweep(sweep_df)
    print_reliability_curve(y_true, y_raw, cal_probs)
    print_calibration_summary(y_true, y_raw, cal_probs, clim_rate)

    best_row = sweep_df.iloc[0].to_dict()
    save_artifacts(df, cal_probs, calibrator, best_thr, best_row)

    df["calibrated_proba"] = cal_probs.astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<8} | {message}")

    run_calibration()
