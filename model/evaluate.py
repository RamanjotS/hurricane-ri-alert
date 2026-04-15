"""
evaluate.py — Final comprehensive evaluation of the RI prediction system.

Loads calibrated_preds.parquet (produced by calibrate.py) and the F1-optimal
alert threshold from optimal_threshold.json, then produces the definitive
evaluation report against the SHIPS-RII operational benchmark.

Metrics computed
----------------
  AUC-ROC              -- primary discrimination metric
  Brier Skill Score    -- BSS = 1 - BS/BS_clim (NHC operational metric)
  POD @ opt threshold  -- Probability of Detection at the F1-optimal threshold
  FAR @ opt threshold  -- False Alarm Ratio at the F1-optimal threshold
  Reliability diagram  -- 10-bin text chart: predicted vs observed RI frequency
  Per-year breakdown   -- AUC and RI event count per test year (2018-2023)
  Per-storm summary    -- mean predicted probability vs actual RI outcome
                         (top 20 most-observed test storms)

Published SHIPS-RII benchmarks (Kaplan & DeMaria 2003; NHC reports):
  AUC ~ 0.78 | BSS ~ 0.08 | POD ~ 0.42 | FAR ~ 0.72

Project targets:
  POD > 0.50 | FAR < 0.65 | BSS > 0.15

Usage:
    python model/evaluate.py
"""

from __future__ import annotations

import json
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
OPTIMAL_THRESHOLD_PATH = ARTIFACTS_DIR / "optimal_threshold.json"

# Fallback threshold used only when optimal_threshold.json is absent
_FALLBACK_THRESHOLD: float = 0.40

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

# Reliability diagram: number of equal-width bins
RELIABILITY_N_BINS: int = 10

# Reliability diagram: each ASCII block character represents this many percent
RELIABILITY_BLOCK_PCT: float = 2.0


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
    logger.info(f"Loading {path.name} ...")
    df = pd.read_parquet(path)
    logger.info(
        f"  -> {len(df):,} rows | {df['storm_id'].nunique():,} storms | "
        f"RI rate {df['ri_label'].mean() * 100:.1f}%"
    )
    return df


def load_optimal_threshold(path: Path = OPTIMAL_THRESHOLD_PATH) -> float:
    """Load the F1-optimal alert threshold from calibrate.py output.

    Falls back to _FALLBACK_THRESHOLD with a warning if the file is absent.

    Args:
        path: Path to optimal_threshold.json.

    Returns:
        Alert threshold as a float.
    """
    if not path.exists():
        logger.warning(
            f"{path.name} not found -- using fallback threshold "
            f"{_FALLBACK_THRESHOLD:.2f}. Run calibrate.py to generate it."
        )
        return _FALLBACK_THRESHOLD

    record = json.loads(path.read_text(encoding="utf-8"))
    thr = float(record["threshold"])
    logger.info(
        f"Loaded optimal threshold: {thr:.2f}  "
        f"(F1={record['f1']:.4f}  POD={record['pod']:.3f}  FAR={record['far']:.3f}  "
        f"source: {record.get('source', 'unknown')})"
    )
    return thr


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
    threshold: float,
) -> tuple[float, float, int, int, int, int]:
    """Compute meteorological POD and FAR at a probability threshold.

    POD = TP / (TP + FN) -- fraction of RI events correctly forecast
    FAR = FP / (TP + FP) -- fraction of RI forecasts that were false alarms

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
# Reliability diagram
# ---------------------------------------------------------------------------


def print_reliability_diagram(
    y_true: np.ndarray,
    y_cal: np.ndarray,
    n_bins: int = RELIABILITY_N_BINS,
    block_pct: float = RELIABILITY_BLOCK_PCT,
) -> None:
    """Print a text reliability diagram: predicted probability vs observed RI freq.

    For each of n_bins equal-width probability bins [0, 1/n), [1/n, 2/n), ...,
    shows the number of observations, RI events, mean predicted probability, and
    observed RI frequency.  Two ASCII bar charts are printed side-by-side so the
    eye can immediately see whether the model is over- or under-predicting in
    each region of the probability scale.

    Perfect calibration: the P-bar (predicted) and O-bar (observed) should be
    the same length in every row.  Over-prediction (P > O) means the model is
    too aggressive in that bin; under-prediction (O > P) means it is too
    conservative.

    Args:
        y_true:    True binary labels.
        y_cal:     Calibrated probabilities.
        n_bins:    Number of equal-width bins spanning [0, 1].
        block_pct: Percentage represented by each '#' character.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows: list[dict] = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include right edge in the last bin
        if i < n_bins - 1:
            mask = (y_cal >= lo) & (y_cal < hi)
        else:
            mask = (y_cal >= lo) & (y_cal <= hi)

        n = int(mask.sum())
        if n == 0:
            continue

        n_ri = int(y_true[mask].sum())
        pred_mean_pct = float(y_cal[mask].mean()) * 100.0
        obs_freq_pct = (n_ri / n) * 100.0
        gap = obs_freq_pct - pred_mean_pct

        rows.append(
            {
                "bin": f"[{lo:.2f},{hi:.2f})",
                "n": n,
                "n_ri": n_ri,
                "pred_pct": pred_mean_pct,
                "obs_pct": obs_freq_pct,
                "gap": gap,
            }
        )

    w = 74
    print()
    print("=" * w)
    print("  Reliability Diagram -- Calibrated Ensemble (10 equal-width bins)")
    print(f"  Each '#' = {block_pct:.0f}%  |  P = mean predicted  O = observed RI freq")
    print("  Perfect calibration: P-bar and O-bar same length each row")
    print("  gap > 0: model UNDER-predicts in bin  |  gap < 0: OVER-predicts")
    print("=" * w)
    print(
        f"  {'Bin':<12}  {'N':>5}  {'RI':>4}  {'Pred%':>6}  {'Obs%':>6}  "
        f"{'Gap':>6}   Bar (P above, O below)"
    )
    print(f"  {'-'*12}  {'-'*5}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}   {'-'*28}")

    for row in rows:
        p_blocks = int(row["pred_pct"] / block_pct + 0.5)
        o_blocks = int(row["obs_pct"] / block_pct + 0.5)
        p_bar = "#" * min(p_blocks, 28)
        o_bar = "#" * min(o_blocks, 28)
        gap_str = f"{row['gap']:+.1f}%"

        # Calibration quality tag
        abs_gap = abs(row["gap"])
        if abs_gap <= 2.0:
            tag = "  [OK]"
        elif row["gap"] > 2.0:
            tag = "  [under]"
        else:
            tag = "  [OVER]"

        # First line: bin info + predicted bar
        print(
            f"  {row['bin']:<12}  {row['n']:>5,}  {row['n_ri']:>4,}  "
            f"{row['pred_pct']:>5.1f}%  {row['obs_pct']:>5.1f}%  {gap_str:>6}"
            f"   P:{p_bar}"
        )
        # Second line: observed bar + calibration tag
        print(f"  {'':12}  {'':5}  {'':4}  {'':6}  {'':6}  {'':6}   O:{o_bar}{tag}")

    print("=" * w)
    print()


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def print_headline_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    clim_rate: float,
    threshold: float,
    label: str = "Calibrated ensemble",
) -> dict[str, float]:
    """Print headline AUC, BSS, POD, FAR and return as dict.

    Args:
        y_true:    True binary labels.
        y_prob:    Predicted probabilities.
        clim_rate: RI climatological base rate.
        threshold: Alert probability threshold (F1-optimal from calibrate.py).
        label:     Display name for the model variant.

    Returns:
        Dict with keys: auc, bss, pod, far, bs.
    """
    auc = roc_auc_score(y_true, y_prob)
    bs, bs_clim, bss = brier_skill_score(y_true, y_prob, clim_rate)
    pod, far, tp, fp, fn, tn = pod_far_at_threshold(y_true, y_prob, threshold)
    ap = average_precision_score(y_true, y_prob)

    metrics = {"auc": auc, "bss": bss, "pod": pod, "far": far, "bs": bs, "ap": ap}

    w = 70
    print()
    print("=" * w)
    print(f"  {label}")
    print(f"  Test Set: 2018-2023  |  Alert threshold: {threshold:.2f} (F1-optimal)")
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
            status = "--"

        tgt_str = f"{tgt:.2f}" if not np.isnan(tgt) else "--"
        print(
            f"  {name:<30}  {val:>8.4f}  {ref:>9.4f}  "
            f"{tgt_str:>8}  {status:>8}"
        )

    _row("AUC-ROC", "auc", higher_better=True)
    _row("Brier Skill Score (BSS)", "bss", higher_better=True)
    _row(f"POD @ thr={threshold:.2f}", "pod", higher_better=True)
    _row(f"FAR @ thr={threshold:.2f}", "far", higher_better=False)

    print()
    print(f"  Brier Score         : {bs:.5f}  (climatology: {bs_clim:.5f})")
    print(f"  Avg Precision (AP)  : {ap:.4f}")
    print()
    print(f"  Confusion matrix @ threshold={threshold:.2f}:")
    print(f"    TP (hit)          : {tp:,}")
    print(f"    FP (false alarm)  : {fp:,}")
    print(f"    FN (miss)         : {fn:,}")
    print(f"    TN (correct neg.) : {tn:,}")
    print("=" * w)
    print()

    return metrics


def print_per_year_breakdown(
    df: pd.DataFrame,
    threshold: float,
) -> None:
    """Print AUC, POD, FAR per year (2018-2023).

    Args:
        df:        Calibrated predictions DataFrame with datetime column.
        threshold: Alert threshold for POD/FAR computation.
    """
    df = df.copy()
    df["year"] = pd.to_datetime(df["datetime"]).dt.year
    y_prob = df["calibrated_proba"].to_numpy(dtype=np.float32)
    y_true = df["ri_label"].to_numpy(dtype=np.int8)

    print()
    print("=" * 65)
    print("  Per-Year Breakdown (calibrated ensemble)")
    print("=" * 65)
    print(f"  {'Year':>6}  {'Obs':>7}  {'RI':>5}  {'RI%':>6}  {'AUC':>8}  {'POD':>6}  {'FAR':>6}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*6}")

    for year in sorted(df["year"].unique()):
        mask = (df["year"] == year).to_numpy()
        yt = y_true[mask]
        yp = y_prob[mask]
        n_ri = int(yt.sum())
        ri_rate = yt.mean() * 100

        if yt.sum() > 0 and len(np.unique(yt)) == 2:
            auc = roc_auc_score(yt, yp)
            auc_str = f"{auc:.4f}"
            pod, far, *_ = pod_far_at_threshold(yt, yp, threshold)
            pod_str = f"{pod:.3f}"
            far_str = f"{far:.3f}"
        else:
            auc_str = pod_str = far_str = "N/A"

        print(
            f"  {year:>6}  {len(yt):>7,}  {n_ri:>5,}  {ri_rate:>5.1f}%  "
            f"{auc_str:>8}  {pod_str:>6}  {far_str:>6}"
        )

    print("=" * 65)
    print()


def print_per_storm_summary(df: pd.DataFrame, top_n: int = 20) -> None:
    """Print mean predicted probability vs. RI outcome for top storms.

    Shows storms with the most observations in the test set.

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
    print(f"  {'-'*15}  {'-'*5}  {'-'*4}  {'-'*5}  {'-'*10}  {'-'*9}")
    for _, row in grp.iterrows():
        print(
            f"  {row['storm_id']:<15}  {int(row['n_obs']):>5}  "
            f"{int(row['n_ri']):>4}  {row['ri_rate']:>4.0f}%  "
            f"{row['mean_pred']:>10.4f}  {row['max_pred']:>9.4f}"
        )
    print("=" * 68)
    print()


def print_model_comparison(
    df: pd.DataFrame,
    clim_rate: float,
    threshold: float,
) -> None:
    """Print AUC, BSS, POD, FAR for all available model variants side-by-side.

    Args:
        df:         Calibrated predictions DataFrame.
        clim_rate:  RI climatological base rate.
        threshold:  F1-optimal alert threshold.
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
    print(f"  Full Model Comparison -- Test Set 2018-2023  (thr={threshold:.2f})")
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
        pod, far, *_ = pod_far_at_threshold(y_true, yp, threshold)
        print(f"  {name:<30}  {auc:.4f}  {bss:.4f}  {pod:.3f}  {far:.3f}")

    print(
        f"  {'SHIPS-RII benchmark':<30}  {SHIPS_RII['auc']:.4f}  {SHIPS_RII['bss']:.4f}  "
        f"{SHIPS_RII['pod']:.3f}  {SHIPS_RII['far']:.3f}"
    )
    print("=" * w)
    print()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_evaluation(path: Path = CALIBRATED_PREDS_PATH) -> dict[str, float]:
    """Run the full evaluation report.

    Steps:
      1. Load calibrated predictions and F1-optimal threshold.
      2. Print reliability diagram (predicted vs observed RI freq, 10 bins).
      3. Print headline metrics (AUC, BSS, POD, FAR) with SHIPS-RII comparison.
      4. Print per-year breakdown with POD/FAR columns.
      5. Print per-storm summary.
      6. Print full model comparison across all variants.

    Args:
        path: Path to calibrated_preds.parquet.

    Returns:
        Dict of headline metric values for the calibrated ensemble.
    """
    df = load_calibrated_preds(path)
    threshold = load_optimal_threshold()

    y_true = df["ri_label"].to_numpy(dtype=np.int8)
    y_cal = df["calibrated_proba"].to_numpy(dtype=np.float32)
    clim_rate = float(y_true.mean())

    print_reliability_diagram(y_true, y_cal)
    metrics = print_headline_metrics(y_true, y_cal, clim_rate, threshold)
    print_per_year_breakdown(df, threshold)
    print_per_storm_summary(df)
    print_model_comparison(df, clim_rate, threshold)

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<8} | {message}")

    run_evaluation()
