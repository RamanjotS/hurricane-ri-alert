"""
calibrate.py — Probability calibration and threshold tuning for the RI ensemble.

Uses a proper held-out calibration split (2015-2017) to fit all calibration
artifacts.  These years fall inside the training-pool val split: XGBoost and
LSTM gradient updates never touched them (they were used only as an early-
stopping monitor), so calibration is genuinely out-of-sample.

Pipeline
--------
  1. Build 2015-2017 predictions from the saved XGBoost and LSTM models.
  2. Fit calibrator_simple : IsotonicRegression on (cal simple-avg -> y).
  3. Fit stacker           : LogisticRegression on (cal xgb + lstm -> y).
     Compute cal stacked proba from stacker.
  4. Fit calibrator_stacked: IsotonicRegression on (cal stacked -> y).
  5. Run 5-fold OOF calibration within the cal set to get unbiased
     calibrated probabilities for threshold tuning.
  6. Sweep thresholds 0.05-0.50 in 0.01 steps on those OOF probs; pick F1-max.
  7. Apply all calibrators + stacker to the 2018-2023 test set.
  8. Print reliability curve and calibration summary (test set).
  9. Save all artifacts.

Output
------
    model/artifacts/calibrator_simple_{ts}.pkl    -- IsotonicRegression (simple avg)
    model/artifacts/calibrator_simple_latest.pkl
    model/artifacts/stacker_holdout_{ts}.pkl      -- LogisticRegression meta-learner
    model/artifacts/stacker_holdout_latest.pkl
    model/artifacts/calibrator_stacked_{ts}.pkl   -- IsotonicRegression (stacked)
    model/artifacts/calibrator_stacked_latest.pkl
    model/artifacts/optimal_threshold.json        -- F1-optimal threshold + metrics
    model/artifacts/calibrated_preds.parquet      -- adds calibrated_proba +
                                                      stacked_cal_proba columns

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
import torch
import torch.nn as nn
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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

ARTIFACTS_DIR        = REPO_ROOT / "model" / "artifacts"
TRAINING_DATA_PATH   = REPO_ROOT / "data" / "processed" / "training_data.parquet"
XGB_MODEL_PATH       = ARTIFACTS_DIR / "xgb_model_latest.pkl"
LSTM_MODEL_PATH      = ARTIFACTS_DIR / "lstm_model_latest.pt"

ENSEMBLE_PREDS_PATH  = ARTIFACTS_DIR / "ensemble_test_preds.parquet"
CALIBRATED_PREDS_PATH = ARTIFACTS_DIR / "calibrated_preds.parquet"
OPTIMAL_THRESHOLD_PATH = ARTIFACTS_DIR / "optimal_threshold.json"

# Calibration holdout window (within training-pool val split; not used for
# gradient updates in either XGBoost or LSTM)
CAL_START_YEAR: int = 2015
CAL_END_YEAR:   int = 2017

# First year of the test set (matches train_xgboost.py / train_lstm.py)
TEST_YEAR: int = 2018

# Threshold sweep range
THRESHOLD_SWEEP: np.ndarray = np.round(np.arange(0.05, 0.51, 0.01), 2)

# OOF folds used inside the cal set for threshold tuning
N_CAL_FOLDS: int = 5

# Column whose raw probability is the primary thing we calibrate
SOURCE_PROB_COL: str = "ensemble_simple"

# Reliability curve bin count (for the test-set print)
N_BINS: int = 10


# ---------------------------------------------------------------------------
# Inline LSTM model (inference-only copy — avoids importing train_lstm.py)
# ---------------------------------------------------------------------------

class _RILSTMModel(nn.Module):
    """Minimal inference-only LSTM identical in architecture to train_lstm.RILSTMModel."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


# ---------------------------------------------------------------------------
# Sequence builder for LSTM calibration-set predictions
# ---------------------------------------------------------------------------


def _build_lstm_seqs_for_cal(
    df_pool: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
    cal_start: int = CAL_START_YEAR,
    cal_end: int = CAL_END_YEAR,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build LSTM 8-step sequences whose target row falls in the cal window.

    The full training-pool DataFrame is used for context so that sequences
    beginning before CAL_START_YEAR still have their prior 7 steps available.
    Only sequences whose final (target) timestep has a year in
    [cal_start, cal_end] are returned.

    Args:
        df_pool:      Training-pool DataFrame (all years, all storms).
        feature_cols: Ordered list of feature column names.
        seq_len:      Number of consecutive 6-hourly steps per sequence.
        cal_start:    First year to include in cal target filter.
        cal_end:      Last year to include in cal target filter.

    Returns:
        Tuple of (X, y, meta) where X is float32 (N, seq_len, n_features),
        y is int8 (N,), and meta is a DataFrame with storm_id, datetime,
        ri_label for each sequence's target row.
    """
    df_s = df_pool.sort_values(["storm_id", "datetime"]).reset_index(drop=True)
    feat_arr = df_s[feature_cols].to_numpy(dtype=np.float32)
    labels   = df_s["ri_label"].to_numpy(dtype=np.float32)
    sids     = df_s["storm_id"].to_numpy()
    dts      = df_s["datetime"].to_numpy()

    X_list:    list[np.ndarray] = []
    y_list:    list[int]        = []
    meta_rows: list[dict]       = []

    for storm_id, grp in df_s.groupby("storm_id", sort=False):
        idx = grp.index.to_numpy()
        if len(idx) < seq_len:
            continue

        t_ns = dts[idx].astype("datetime64[s]").astype(np.int64)
        gaps_h = np.diff(t_ns) / 3600

        for end in range(seq_len - 1, len(idx)):
            start = end - (seq_len - 1)
            if not np.all(gaps_h[start:end] == 6):
                continue
            tgt = idx[end]
            lbl = labels[tgt]
            if np.isnan(lbl):
                continue
            tgt_year = pd.Timestamp(dts[tgt]).year
            if not (cal_start <= tgt_year <= cal_end):
                continue
            X_list.append(feat_arr[idx[start : end + 1]])
            y_list.append(int(lbl))
            meta_rows.append({
                "storm_id": sids[tgt],
                "datetime": pd.Timestamp(dts[tgt]),
                "ri_label": int(lbl),
            })

    if not X_list:
        empty_meta = pd.DataFrame(columns=["storm_id", "datetime", "ri_label"])
        return (
            np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32),
            np.zeros(0, dtype=np.int8),
            empty_meta,
        )

    return (
        np.stack(X_list, axis=0),
        np.array(y_list, dtype=np.int8),
        pd.DataFrame(meta_rows).reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Build 2015-2017 calibration-set predictions
# ---------------------------------------------------------------------------


def build_cal_predictions() -> pd.DataFrame:
    """Generate 2015-2017 predictions from both base models for calibration.

    Loads the full training dataset, filters to training-pool storms (first
    observed before TEST_YEAR) and observations in [CAL_START_YEAR,
    CAL_END_YEAR], then runs both the saved XGBoost and LSTM models in
    inference mode.

    XGBoost predictions are straightforward (no sequence context needed).
    LSTM predictions use the full training-pool DataFrame for 8-step context
    but filter output to sequences whose target timestep falls in the cal
    window.

    Returns:
        DataFrame with storm_id, datetime, ri_label, xgb_proba, lstm_proba,
        ensemble_simple — inner join of both model outputs on (storm_id, datetime).

    Raises:
        FileNotFoundError: If any required artifact or data file is absent.
    """
    for path in (TRAINING_DATA_PATH, XGB_MODEL_PATH, LSTM_MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run data/scripts/build_training_data.py, "
                "model/train_xgboost.py, and model/train_lstm.py first."
            )

    from data.scripts.build_training_data import ALL_FEATURE_COLUMNS

    logger.info(
        f"Building calibration predictions ({CAL_START_YEAR}-{CAL_END_YEAR}) ..."
    )

    # Load training data; identify training-pool storms
    df_full = pd.read_parquet(TRAINING_DATA_PATH)
    df_full["datetime"] = pd.to_datetime(df_full["datetime"])
    storm_first_year = df_full.groupby("storm_id")["datetime"].min().dt.year
    pool_storms = storm_first_year[storm_first_year < TEST_YEAR].index
    df_pool = df_full[df_full["storm_id"].isin(pool_storms)].copy()
    logger.info(
        f"  Training pool: {len(df_pool):,} rows | {df_pool['storm_id'].nunique():,} storms"
    )

    # ---- XGBoost predictions on 2015-2017 rows ----
    logger.info("  Running XGBoost on 2015-2017 rows ...")
    with open(XGB_MODEL_PATH, "rb") as fh:
        xgb_model = pickle.load(fh)

    cal_year_mask = df_pool["datetime"].dt.year.between(CAL_START_YEAR, CAL_END_YEAR)
    df_cal_rows = df_pool[cal_year_mask].reset_index(drop=True)
    logger.info(
        f"  Cal rows: {len(df_cal_rows):,} | "
        f"RI rate {df_cal_rows['ri_label'].mean() * 100:.1f}%"
    )

    X_xgb = df_cal_rows[ALL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    xgb_proba = xgb_model.predict_proba(X_xgb)[:, 1].astype(np.float32)

    df_xgb_cal = pd.DataFrame({
        "storm_id": df_cal_rows["storm_id"].values,
        "datetime": df_cal_rows["datetime"].values,
        "ri_label": df_cal_rows["ri_label"].values,
        "xgb_proba": xgb_proba,
    })
    logger.info(f"  XGBoost cal rows: {len(df_xgb_cal):,}")

    # ---- LSTM predictions on sequences targeting 2015-2017 ----
    logger.info("  Running LSTM on 2015-2017 target sequences ...")
    ckpt = torch.load(LSTM_MODEL_PATH, map_location="cpu", weights_only=False)
    feature_cols = ckpt["feature_cols"]
    seq_len      = ckpt["seq_len"]
    hidden_size  = ckpt.get("hidden_size", 128)
    num_layers   = ckpt.get("num_layers", 2)
    dropout      = ckpt.get("dropout", 0.3)

    lstm_model = _RILSTMModel(
        n_features=ckpt["n_features"],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    lstm_model.load_state_dict(ckpt["state_dict"])
    lstm_model.eval()

    X_lstm, y_lstm, meta_lstm = _build_lstm_seqs_for_cal(
        df_pool, feature_cols, seq_len
    )
    logger.info(
        f"  LSTM cal sequences: {len(X_lstm):,} | "
        f"RI rate {y_lstm.mean() * 100:.1f}%"
    )

    with torch.no_grad():
        logits = lstm_model(torch.from_numpy(X_lstm)).cpu().numpy().squeeze(-1)
    lstm_proba = torch.sigmoid(torch.from_numpy(logits)).numpy().astype(np.float32)

    df_lstm_cal = pd.DataFrame({
        "storm_id": meta_lstm["storm_id"].values,
        "datetime": pd.to_datetime(meta_lstm["datetime"]).values,
        "lstm_proba": lstm_proba,
    })
    logger.info(f"  LSTM cal rows: {len(df_lstm_cal):,}")

    # ---- Inner join ----
    # Strip timezone before merge if present
    for d in (df_xgb_cal, df_lstm_cal):
        d["datetime"] = pd.to_datetime(d["datetime"])
        if d["datetime"].dt.tz is not None:
            d["datetime"] = d["datetime"].dt.tz_localize(None)

    cal_df = pd.merge(
        df_xgb_cal, df_lstm_cal, on=["storm_id", "datetime"], how="inner"
    )
    cal_df["ensemble_simple"] = (
        0.5 * cal_df["xgb_proba"] + 0.5 * cal_df["lstm_proba"]
    ).astype(np.float32)

    logger.info(
        f"  Cal set (inner join): {len(cal_df):,} rows | "
        f"{cal_df['storm_id'].nunique():,} storms | "
        f"RI rate {cal_df['ri_label'].mean() * 100:.1f}%"
    )
    return cal_df


# ---------------------------------------------------------------------------
# Holdout-calibration: fit on 2015-2017, apply to 2018-2023
# ---------------------------------------------------------------------------


def calibrate_on_holdout(
    test_df: pd.DataFrame,
    cal_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           IsotonicRegression, LogisticRegression, IsotonicRegression]:
    """Fit all calibration artifacts on the 2015-2017 holdout set.

    Three artifacts are trained:
      - calibrator_simple : IsotonicRegression on (cal simple-avg -> y)
      - stacker           : LogisticRegression on ([cal xgb, cal lstm] -> y)
      - calibrator_stacked: IsotonicRegression on (cal stacked proba -> y)

    An additional 5-fold OOF calibration is performed within the cal set to
    produce unbiased cal-set calibrated probabilities for threshold tuning —
    these never touch the 2018-2023 test labels.

    Args:
        test_df: DataFrame from ensemble_test_preds.parquet (2018-2023).
        cal_df:  DataFrame from build_cal_predictions() (2015-2017).

    Returns:
        Tuple of:
          (test_cal_proba,       -- calibrated simple-avg proba on test set
           test_stacked_cal,     -- calibrated stacked proba on test set
           cal_oof_simple,       -- 5-fold OOF calibrated simple-avg on cal set
           cal_y,                -- true labels for cal set (for threshold sweep)
           calibrator_simple,
           stacker,
           calibrator_stacked)
    """
    cal_y       = cal_df["ri_label"].to_numpy(dtype=np.int8)
    cal_simple  = cal_df["ensemble_simple"].to_numpy(dtype=np.float32)
    cal_meta    = cal_df[["xgb_proba", "lstm_proba"]].to_numpy(dtype=np.float32)

    test_y      = test_df["ri_label"].to_numpy(dtype=np.int8)
    test_simple = test_df["ensemble_simple"].to_numpy(dtype=np.float32)
    test_meta   = test_df[["xgb_proba", "lstm_proba"]].to_numpy(dtype=np.float32)

    # -- calibrator_simple: trained on full cal set ----------------------------
    logger.info("  Fitting calibrator_simple on cal set ...")
    calibrator_simple = IsotonicRegression(out_of_bounds="clip")
    calibrator_simple.fit(cal_simple, cal_y)

    # -- stacker: trained on full cal set --------------------------------------
    logger.info("  Fitting stacker on cal set ...")
    stacker = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    stacker.fit(cal_meta, cal_y)
    coef = stacker.coef_[0]
    logger.info(
        f"    Stacker coef: xgb={coef[0]:.4f}  lstm={coef[1]:.4f}  "
        f"intercept={stacker.intercept_[0]:.4f}"
    )

    cal_stacked = stacker.predict_proba(cal_meta)[:, 1].astype(np.float32)

    # -- calibrator_stacked: trained on stacker's cal output -------------------
    logger.info("  Fitting calibrator_stacked on cal set ...")
    calibrator_stacked = IsotonicRegression(out_of_bounds="clip")
    calibrator_stacked.fit(cal_stacked, cal_y)

    # -- 5-fold OOF within cal set (for threshold tuning) ----------------------
    logger.info(f"  5-fold OOF within cal set for threshold tuning ...")
    cal_oof_simple = np.full_like(cal_simple, fill_value=np.nan, dtype=np.float32)
    kf = KFold(n_splits=N_CAL_FOLDS, shuffle=False)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(cal_simple), start=1):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(cal_simple[tr_idx], cal_y[tr_idx])
        cal_oof_simple[val_idx] = ir.predict(cal_simple[val_idx]).astype(np.float32)
        logger.debug(
            f"    Fold {fold}/{N_CAL_FOLDS}: "
            f"OOF AUC={roc_auc_score(cal_y[val_idx], cal_oof_simple[val_idx]):.4f}"
        )

    # -- Apply to test set -----------------------------------------------------
    test_cal_proba   = calibrator_simple.predict(test_simple).astype(np.float32)
    test_stacked_raw = stacker.predict_proba(test_meta)[:, 1].astype(np.float32)
    test_stacked_cal = calibrator_stacked.predict(test_stacked_raw).astype(np.float32)

    logger.info(
        f"  Test set calibrated: simple-avg BSS check — "
        f"clim={float(test_y.mean()):.4f}"
    )

    return (
        test_cal_proba,
        test_stacked_cal,
        cal_oof_simple,
        cal_y,
        calibrator_simple,
        stacker,
        calibrator_stacked,
    )


# ---------------------------------------------------------------------------
# Threshold tuning (unchanged logic, now run on cal-set OOF probs)
# ---------------------------------------------------------------------------


def tune_threshold(
    y_true: np.ndarray,
    cal_probs: np.ndarray,
    sweep: np.ndarray = THRESHOLD_SWEEP,
) -> tuple[float, float, pd.DataFrame]:
    """Find F1-optimal alert threshold on calibrated probabilities.

    Args:
        y_true:    True binary labels.
        cal_probs: Calibrated probabilities (OOF within cal set).
        sweep:     Threshold candidates.

    Returns:
        (best_threshold, best_f1, sweep_df sorted by f1 desc).
    """
    rows: list[dict] = []
    for thr in sweep:
        y_pred = (cal_probs >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom > 0 else 0.0
        rows.append({
            "threshold": float(thr), "f1": f1, "pod": pod, "far": far,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })

    sweep_df = (
        pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    )
    best = sweep_df.iloc[0]
    best_threshold, best_f1 = float(best["threshold"]), float(best["f1"])

    logger.info(
        f"Threshold sweep complete — optimal: {best_threshold:.2f}  "
        f"F1={best_f1:.4f}  POD={best['pod']:.3f}  FAR={best['far']:.3f}  "
        f"(tuned on {CAL_START_YEAR}-{CAL_END_YEAR} OOF cal probs)"
    )
    return best_threshold, best_f1, sweep_df


def print_threshold_sweep(sweep_df: pd.DataFrame, top_n: int = 15) -> None:
    """Print top-N threshold candidates ranked by F1.

    Args:
        sweep_df: Returned by tune_threshold (sorted by F1 desc).
        top_n:    Number of rows to display.
    """
    print()
    print("=" * 72)
    print(
        f"  Threshold Sweep — F1-Optimal (tuned on {CAL_START_YEAR}-{CAL_END_YEAR} "
        f"OOF cal probs, NOT test set)"
    )
    print(f"  Showing top {top_n} of {len(sweep_df)} candidates")
    print("=" * 72)
    print(
        f"  {'Rank':>4}  {'Thr':>5}  {'F1':>7}  {'POD':>7}  "
        f"{'FAR':>7}  {'TP':>5}  {'FP':>5}  {'FN':>5}"
    )
    print(
        f"  {'-'*4}  {'-'*5}  {'-'*7}  {'-'*7}  "
        f"{'-'*7}  {'-'*5}  {'-'*5}  {'-'*5}"
    )
    for rank, row in sweep_df.head(top_n).iterrows():
        star = " <-- optimal" if rank == 0 else ""
        print(
            f"  {rank + 1:>4}  {row['threshold']:>5.2f}  {row['f1']:>7.4f}  "
            f"{row['pod']:>7.3f}  {row['far']:>7.3f}  "
            f"{int(row['tp']):>5}  {int(row['fp']):>5}  {int(row['fn']):>5}{star}"
        )
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Reliability curve (on test set — shows how well holdout cal generalises)
# ---------------------------------------------------------------------------


def print_reliability_curve(
    y_true: np.ndarray,
    y_raw: np.ndarray,
    y_cal: np.ndarray,
    n_bins: int = N_BINS,
) -> None:
    """Print reliability curve for raw vs holdout-calibrated probabilities.

    Args:
        y_true: True binary labels (test set).
        y_raw:  Raw ensemble probabilities (test set).
        y_cal:  Holdout-calibrated probabilities (test set).
        n_bins: Number of equal-width bins.
    """
    frac_pos_raw, mean_pred_raw = calibration_curve(
        y_true, y_raw, n_bins=n_bins, strategy="uniform"
    )
    frac_pos_cal, mean_pred_cal = calibration_curve(
        y_true, y_cal, n_bins=n_bins, strategy="uniform"
    )

    w = 72
    print()
    print("=" * w)
    print("  Reliability Curve (test set) -- Raw vs Holdout-Calibrated")
    print("=" * w)
    print(
        f"  {'Bin centre':>10}  {'Raw freq':>10}  "
        f"{'Cal freq':>10}  {'d_raw':>8}  {'d_cal':>8}"
    )
    print(
        f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}"
    )

    max_len = max(len(frac_pos_raw), len(frac_pos_cal))
    for i in range(max_len):
        if i < len(mean_pred_raw):
            ctr = f"{mean_pred_raw[i]:.2f}"
            fr  = f"{frac_pos_raw[i]:.3f}"
            d_r = f"{frac_pos_raw[i] - mean_pred_raw[i]:+.3f}"
        else:
            ctr = fr = d_r = "N/A"

        if i < len(frac_pos_cal):
            fc  = f"{frac_pos_cal[i]:.3f}"
            d_c = f"{frac_pos_cal[i] - mean_pred_cal[i]:+.3f}"
        else:
            fc = d_c = "N/A"

        print(f"  {ctr:>10}  {fr:>10}  {fc:>10}  {d_r:>8}  {d_c:>8}")

    print("=" * w)
    print()


# ---------------------------------------------------------------------------
# Calibration summary (test set)
# ---------------------------------------------------------------------------


def print_calibration_summary(
    y_true: np.ndarray,
    y_raw: np.ndarray,
    y_cal: np.ndarray,
    y_stacked_cal: np.ndarray,
    clim_rate: float,
) -> None:
    """Print AUC, Brier Score, BSS for raw, calibrated, and stacked-cal variants.

    Args:
        y_true:        True binary labels (test set).
        y_raw:         Raw simple-avg probabilities (test set).
        y_cal:         Holdout-calibrated simple-avg probabilities (test set).
        y_stacked_cal: Holdout-calibrated stacked probabilities (test set).
        clim_rate:     RI climatological rate (from test set).
    """
    bs_clim = float(np.mean((clim_rate - y_true) ** 2))

    def _row(label: str, probs: np.ndarray) -> None:
        auc = roc_auc_score(y_true, probs)
        bs  = brier_score_loss(y_true, probs)
        bss = 1.0 - bs / bs_clim if bs_clim > 0 else float("nan")
        print(f"  {label:<32}  AUC={auc:.4f}  BS={bs:.5f}  BSS={bss:.4f}")

    print()
    print("=" * 72)
    print("  Calibration Summary (2018-2023 test set)")
    print(f"  Calibrator + stacker both trained on {CAL_START_YEAR}-{CAL_END_YEAR} holdout")
    print("=" * 72)
    _row("Raw ensemble (simple avg)", y_raw)
    _row("Holdout-calibrated simple avg", y_cal)
    _row("Holdout-calibrated stacked", y_stacked_cal)
    print(
        f"  {'Climatology baseline':<32}  AUC=N/A     "
        f"BS={bs_clim:.5f}  BSS=0.0000"
    )
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------


def save_artifacts(
    test_df: pd.DataFrame,
    test_cal_proba: np.ndarray,
    test_stacked_cal: np.ndarray,
    calibrator_simple: IsotonicRegression,
    stacker: LogisticRegression,
    calibrator_stacked: IsotonicRegression,
    threshold: float,
    threshold_metrics: dict,
) -> None:
    """Persist all calibration artifacts and the updated predictions parquet.

    Saves:
      calibrator_simple_{ts}.pkl / calibrator_simple_latest.pkl
      stacker_holdout_{ts}.pkl   / stacker_holdout_latest.pkl
      calibrator_stacked_{ts}.pkl/ calibrator_stacked_latest.pkl
      optimal_threshold.json
      calibrated_preds.parquet (calibrated_proba + stacked_cal_proba columns)

    Args:
        test_df:            Ensemble test predictions DataFrame.
        test_cal_proba:     Holdout-calibrated simple-avg proba (test set).
        test_stacked_cal:   Holdout-calibrated stacked proba (test set).
        calibrator_simple:  Fitted IsotonicRegression (simple avg).
        stacker:            Fitted LogisticRegression meta-learner.
        calibrator_stacked: Fitted IsotonicRegression (stacked).
        threshold:          F1-optimal alert threshold.
        threshold_metrics:  Dict with f1, pod, far, tp, fp, fn at threshold.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Predictions parquet
    out_df = test_df.copy()
    out_df["calibrated_proba"]  = test_cal_proba.astype(np.float32)
    out_df["stacked_cal_proba"] = test_stacked_cal.astype(np.float32)
    out_df.to_parquet(CALIBRATED_PREDS_PATH, index=False)
    logger.success(
        f"Calibrated predictions saved -> {CALIBRATED_PREDS_PATH.name}  "
        f"({len(out_df):,} rows, columns: calibrated_proba + stacked_cal_proba)"
    )

    # Three pkl artifacts
    def _save_pkl(obj, stem: str) -> None:
        p = ARTIFACTS_DIR / f"{stem}_{ts}.pkl"
        with open(p, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        shutil.copy2(p, ARTIFACTS_DIR / f"{stem}_latest.pkl")
        logger.success(f"Saved -> {p.name}  (also {stem}_latest.pkl)")

    _save_pkl(calibrator_simple,  "calibrator_simple")
    _save_pkl(stacker,             "stacker_holdout")
    _save_pkl(calibrator_stacked,  "calibrator_stacked")

    # Optimal threshold JSON
    record = {
        "threshold": threshold,
        "f1":  threshold_metrics["f1"],
        "pod": threshold_metrics["pod"],
        "far": threshold_metrics["far"],
        "tp":  threshold_metrics["tp"],
        "fp":  threshold_metrics["fp"],
        "fn":  threshold_metrics["fn"],
        "source": (
            f"F1-optimal sweep on {CAL_START_YEAR}-{CAL_END_YEAR} "
            "OOF calibrated probabilities (holdout split)"
        ),
    }
    OPTIMAL_THRESHOLD_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.success(
        f"Optimal threshold saved -> {OPTIMAL_THRESHOLD_PATH.name}  "
        f"(threshold={threshold:.2f}  F1={threshold_metrics['f1']:.4f})"
    )


# ---------------------------------------------------------------------------
# Load ensemble test predictions
# ---------------------------------------------------------------------------


def load_ensemble_preds(path: Path = ENSEMBLE_PREDS_PATH) -> pd.DataFrame:
    """Load 2018-2023 test-set ensemble predictions from ensemble.py output.

    Args:
        path: Path to ensemble_test_preds.parquet.

    Returns:
        DataFrame with storm_id, datetime, ri_label, xgb_proba, lstm_proba,
        ensemble_simple, ensemble_stacked.

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
# Pipeline
# ---------------------------------------------------------------------------


def run_calibration() -> pd.DataFrame:
    """Run the full holdout-calibration and threshold-tuning pipeline.

    Steps:
      1. Build 2015-2017 calibration-set predictions (both base models).
      2. Load 2018-2023 test predictions from ensemble_test_preds.parquet.
      3. Fit calibrator_simple, stacker, calibrator_stacked on 2015-2017.
      4. Run 5-fold OOF within 2015-2017 for unbiased threshold tuning.
      5. Sweep thresholds; pick F1-optimal on 2015-2017 OOF probs.
      6. Apply calibrators to 2018-2023 test set.
      7. Print threshold sweep, reliability curve, calibration summary.
      8. Save all artifacts.

    Returns:
        DataFrame (2018-2023 test set) with calibrated_proba and
        stacked_cal_proba columns added.
    """
    # Step 1 & 2
    cal_df  = build_cal_predictions()
    test_df = load_ensemble_preds()

    y_test_true = test_df["ri_label"].to_numpy(dtype=np.int8)
    y_test_raw  = test_df[SOURCE_PROB_COL].to_numpy(dtype=np.float32)
    clim_rate   = float(y_test_true.mean())

    # Step 3 & 4
    logger.info("Fitting calibration artifacts on holdout set ...")
    (
        test_cal_proba,
        test_stacked_cal,
        cal_oof_simple,
        cal_y,
        calibrator_simple,
        stacker,
        calibrator_stacked,
    ) = calibrate_on_holdout(test_df, cal_df)

    # Step 5
    logger.info(
        f"Tuning alert threshold on {CAL_START_YEAR}-{CAL_END_YEAR} OOF probs "
        f"(sweep: {THRESHOLD_SWEEP[0]:.2f} -> {THRESHOLD_SWEEP[-1]:.2f}) ..."
    )
    best_thr, best_f1, sweep_df = tune_threshold(cal_y, cal_oof_simple)

    # Step 7 — print reports
    print_threshold_sweep(sweep_df)
    print_reliability_curve(y_test_true, y_test_raw, test_cal_proba)
    print_calibration_summary(
        y_test_true, y_test_raw, test_cal_proba, test_stacked_cal, clim_rate
    )

    # Step 8
    best_row = sweep_df.iloc[0].to_dict()
    save_artifacts(
        test_df,
        test_cal_proba,
        test_stacked_cal,
        calibrator_simple,
        stacker,
        calibrator_stacked,
        best_thr,
        best_row,
    )

    test_df = test_df.copy()
    test_df["calibrated_proba"]  = test_cal_proba.astype(np.float32)
    test_df["stacked_cal_proba"] = test_stacked_cal.astype(np.float32)
    return test_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<8} | {message}")

    run_calibration()
