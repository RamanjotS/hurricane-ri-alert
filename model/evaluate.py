"""
evaluate.py — Final comprehensive evaluation of the RI prediction system.

Loads calibrated_preds.parquet (produced by calibrate.py) and produces the
definitive evaluation report for the full system against the SHIPS-RII
operational benchmark.

Metrics computed
----------------
  AUC-ROC              — primary discrimination metric
  Brier Skill Score    — BSS = 1 – BS/BS_clim (NHC operational metric)
  POD @ 40%            — Probability of Detection at the alert threshold
  FAR @ 40%            — False Alarm Ratio at the alert threshold
  Reliability diagram  — binned calibration statistics (text)
  Per-year breakdown   — AUC and RI event count per test year (2018–2023)
  Per-storm summary    — mean predicted probability vs. actual RI outcome
                         (top 20 most-observed test storms)

Published SHIPS-RII benchmarks (Kaplan & DeMaria 2003; NHC reports):
  AUC ~ 0.78 | BSS ~ 0.08 | POD ~ 0.42 | FAR ~ 0.72

Project targets:
  POD > 0.50 | FAR < 0.65 | BSS > 0.15

Usage:
    python model/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

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
CALIBRATED_PREDS_PATH = ARTIFACTS_DIR / "calibrated_preds.parquet"

# Probability threshold for categorical metrics
RI_PROB_THRESHOLD: float = 0.40

# Published SHIPS-RII benchmarks
SHIPS_RII: dict[str, float] = {
    "auc": 0.78,
    "bss": 0.08,
    "pod": 0.42,
    "far": 0.72,
}

# Project performance targets
TARGETS: dict[str, float] = {
    "pod": 0.50,
    "far": 0.65,
    "bss": 0.15,
}


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_calibrated_preds(path: Path = CALIBRATED_PREDS_PATH) -> pd.DataFrame:
    """Load the final calibrated ensemble predictions.

    Args:
        path: Path to calibrated_preds.parquet.

    Returns:
        DataFrame with storm_id, datetime, ri_label, xgb_proba, lstm_proba,
        ensemble_simple, ensemble_stacked, calibrated_proba.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Calibrated predictions not found: {path}\n"
            "Run: python model/calibrate.py"
        )
    logger.info(f"Loading {path.name} …")
    df = pd.read_parquet(path)
    logger.info(
        f"  → {len(df):,} rows | {df['storm_id'].nunique():,} storms | "
        f"RI rate {df['ri_label'].mean() * 100:.1f}%"
    )
    return df


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def brier_skill_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    clim_rate: float,
) -> tuple[float, float, float]:
    """Return (brier_score, bs_climatology, bss).

    Args:
        y_true:    True binary labels.
        y_prob:    Predicted probabilities.
        clim_rate: Training-set RI base rate.
    """
    bs = float(np.mean((y_prob - y_true) ** 2))
    bs_clim = float(np.mean((clim_rate - y_true) ** 2))
    bss = 1.0 - bs / bs_clim if bs_clim > 0 else float("nan")
    return bs, bs_clim, bss


def pod_far_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = RI_PROB_THRESHOLD,
) -> tuple[float, float, int, int, int, int]:
    """Compute meteorological POD and FAR at a probability threshold.

    POD = TP / (TP + FN) — fraction of RI events correctly forecast
    FAR = FP / (TP + FP) — fraction of RI forecasts that were false alarms

    Args:
        y_true:    True binary labels.
        y_prob:    Predicted probabilities.
        threshold: Probability cutoff for a positive forecast.

    Returns:
        (pod, far, tp, fp, fn, tn)
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    return float(pod), float(far), int(tp), int(fp), int(fn), int(tn)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def print_headline_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    clim_rate: float,
    label: str = "Calibrated ensemble",
) -> dict[str, float]:
    """Print headline AUC, BSS, POD, FAR and return as dict.

    Args:
        y_true:    True binary labels.
        y_prob:    Predicted probabilities.
        clim_rate: RI climatological base rate.
        label:     Display name for the model variant.

    Returns:
        Dict with keys: auc, bss, pod, far, bs.
    """
    auc = roc_auc_score(y_true, y_prob)
    bs, bs_clim, bss = brier_skill_score(y_true, y_prob, clim_rate)
    pod, far, tp, fp, fn, tn = pod_far_at_threshold(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    metrics = {"auc": auc, "bss": bss, "pod": pod, "far": far, "bs": bs, "ap": ap}

    w = 70
    print()
    print("=" * w)
    print(f"  {label}")
    print(f"  Test Set: 2018–2023  |  Threshold: {RI_PROB_THRESHOLD:.0%}")
    print("=" * w)
    print(f"  {'Metric':<30}  {'Model':>8}  {'SHIPS-RII':>9}  {'Target':>8}  {'Status':>8}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*8}")

    def _row(
        name: str,
        key: str,
        ref_key: str | None = None,
        target_key: str | None = None,
        higher_better: bool = True,
    ) -> None:
        val = metrics[key]
        ref = SHIPS_RII.get(ref_key or key, float("nan"))
        tgt = TARGETS.get(target_key or key, float("nan"))

        if not np.isnan(tgt):
            met = (val >= tgt) if higher_better else (val <= tgt)
            status = "PASS" if met else "FAIL"
        else:
            status = "—"

        print(
            f"  {name:<30}  {val:>8.4f}  {ref:>9.4f}  "
            f"{tgt if not np.isnan(tgt) else '—':>8}  {status:>8}"
        )

    _row("AUC-ROC", "auc", higher_better=True)
    _row("Brier Skill Score (BSS)", "bss", higher_better=True)
    _row(f"POD @ {RI_PROB_THRESHOLD:.0%}", "pod", higher_better=True)
    _row(f"FAR @ {RI_PROB_THRESHOLD:.0%}", "far", higher_better=False)

    print()
    print(f"  Brier Score         : {bs:.5f}  (climatology: {bs_clim:.5f})")
    print(f"  Avg Precision (AP)  : {ap:.4f}")
    print()
    print(f"  Confusion matrix @ {RI_PROB_THRESHOLD:.0%} threshold:")
    print(f"    TP (hit)          : {tp:,}")
    print(f"    FP (false alarm)  : {fp:,}")
    print(f"    FN (miss)         : {fn:,}")
    print(f"    TN (correct neg.) : {tn:,}")
    print("=" * w)
    print()

    return metrics


def print_per_year_breakdown(df: pd.DataFrame) -> None:
    """Print AUC and RI event count per year (2018–2023).

    Args:
        df: Calibrated predictions DataFrame with datetime column.
    """
    df = df.copy()
    df["year"] = pd.to_datetime(df["datetime"]).dt.year
    y_prob = df["calibrated_proba"].to_numpy(dtype=np.float32)
    y_true = df["ri_label"].to_numpy(dtype=np.int8)

    print()
    print("=" * 55)
    print("  Per-Year Breakdown (calibrated ensemble)")
    print("=" * 55)
    print(f"  {'Year':>6}  {'Obs':>7}  {'RI':>5}  {'RI%':>6}  {'AUC':>8}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*5}  {'-'*6}  {'-'*8}")

    for year in sorted(df["year"].unique()):
        mask = (df["year"] == year).to_numpy()
        yt = y_true[mask]
        yp = y_prob[mask]
        n_ri = int(yt.sum())
        ri_rate = yt.mean() * 100
        if yt.sum() > 0 and len(np.unique(yt)) == 2:
            auc = roc_auc_score(yt, yp)
            auc_str = f"{auc:.4f}"
        else:
            auc_str = "N/A"
        print(f"  {year:>6}  {len(yt):>7,}  {n_ri:>5,}  {ri_rate:>5.1f}%  {auc_str:>8}")

    print("=" * 55)
    print()


def print_per_storm_summary(df: pd.DataFrame, top_n: int = 20) -> None:
    """Print mean predicted probability vs. RI outcome for top storms.

    Shows storms with the most observations in the test set, useful for
    sanity-checking that the model assigns high probabilities to known
    rapidly intensifying storms.

    Args:
        df:    Calibrated predictions DataFrame.
        top_n: Number of storms to display.
    """
    grp = (
        df.groupby("storm_id")
        .agg(
            n_obs=("ri_label", "count"),
            n_ri=("ri_label", "sum"),
            mean_pred=("calibrated_proba", "mean"),
            max_pred=("calibrated_proba", "max"),
        )
        .reset_index()
    )
    grp["ri_rate"] = 100.0 * grp["n_ri"] / grp["n_obs"]
    grp = grp.sort_values("n_obs", ascending=False).head(top_n)

    print()
    print("=" * 68)
    print(f"  Per-Storm Summary (top {top_n} by observation count)")
    print("=" * 68)
    print(
        f"  {'Storm ID':<15}  {'Obs':>5}  {'RI':>4}  {'RI%':>5}  "
        f"{'Mean pred':>10}  {'Max pred':>9}"
    )
    print(
        f"  {'-'*15}  {'-'*5}  {'-'*4}  {'-'*5}  {'-'*10}  {'-'*9}"
    )
    for _, row in grp.iterrows():
        print(
            f"  {row['storm_id']:<15}  {int(row['n_obs']):>5}  "
            f"{int(row['n_ri']):>4}  {row['ri_rate']:>4.0f}%  "
            f"{row['mean_pred']:>10.4f}  {row['max_pred']:>9.4f}"
        )
    print("=" * 68)
    print()


def print_model_comparison(df: pd.DataFrame, clim_rate: float) -> None:
    """Print AUC and BSS for all available model variants side-by-side.

    Args:
        df:         Calibrated predictions DataFrame.
        clim_rate:  RI climatological base rate.
    """
    y_true = df["ri_label"].to_numpy(dtype=np.int8)
    bs_clim = float(np.mean((clim_rate - y_true) ** 2))

    variants: list[tuple[str, str]] = [
        ("XGBoost only", "xgb_proba"),
        ("LSTM only", "lstm_proba"),
        ("Simple average (0.5/0.5)", "ensemble_simple"),
        ("Stacked (logistic)", "ensemble_stacked"),
        ("Calibrated ensemble", "calibrated_proba"),
    ]

    w = 66
    print()
    print("=" * w)
    print("  Full Model Comparison — Test Set 2018–2023")
    print("=" * w)
    print(f"  {'Variant':<30}  {'AUC':>8}  {'BSS':>8}  {'POD':>6}  {'FAR':>6}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}")

    for name, col in variants:
        if col not in df.columns:
            continue
        yp = df[col].to_numpy(dtype=np.float32)
        auc = roc_auc_score(y_true, yp)
        bs = float(np.mean((yp - y_true) ** 2))
        bss = 1.0 - bs / bs_clim if bs_clim > 0 else float("nan")
        pod, far, *_ = pod_far_at_threshold(y_true, yp)
        print(
            f"  {name:<30}  {auc:.4f}  {bss:.4f}  {pod:.3f}  {far:.3f}"
        )

    print(f"  {'SHIPS-RII benchmark':<30}  {SHIPS_RII['auc']:.4f}  {SHIPS_RII['bss']:.4f}  "
          f"{SHIPS_RII['pod']:.3f}  {SHIPS_RII['far']:.3f}")
    print("=" * w)
    print()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_evaluation(path: Path = CALIBRATED_PREDS_PATH) -> dict[str, float]:
    """Run the full evaluation report.

    Steps:
      1. Load calibrated predictions.
      2. Print headline metrics (AUC, BSS, POD, FAR) with SHIPS-RII comparison.
      3. Print per-year breakdown.
      4. Print per-storm summary.
      5. Print full model comparison across all variants.

    Args:
        path: Path to calibrated_preds.parquet.

    Returns:
        Dict of headline metric values for the calibrated ensemble.
    """
    df = load_calibrated_preds(path)
    y_true = df["ri_label"].to_numpy(dtype=np.int8)
    y_cal = df["calibrated_proba"].to_numpy(dtype=np.float32)
    clim_rate = float(y_true.mean())

    metrics = print_headline_metrics(y_true, y_cal, clim_rate)
    print_per_year_breakdown(df)
    print_per_storm_summary(df)
    print_model_comparison(df, clim_rate)

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<8} | {message}")

    run_evaluation()
