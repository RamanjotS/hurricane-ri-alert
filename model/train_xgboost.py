"""
train_xgboost.py — Train an XGBoost binary classifier for Rapid Intensification
prediction.

Loads data/processed/training_data.parquet, performs a temporal train/test split
(storms whose first observation is before 2018 train; 2018–2023 test), trains an
XGBClassifier with early stopping on a held-out validation tail, evaluates against
the SHIPS-RII operational benchmark, and writes the model and test-set predictions
to model/artifacts/.

Temporal split rationale
------------------------
Random splitting would leak future storm structure into the training set, inflating
all metrics.  We split on the storm's *first observation year* so that every row
belonging to a given storm stays in one partition:

  Train pool : storms first observed in 1982–2017
  Test       : storms first observed in 2018–2023
  Val        : last 15 % of training-pool rows by time (for early stopping only)

Evaluation metrics
------------------
  AUC-ROC             — primary discrimination metric
  Brier Skill Score   — BSS = 1 – BS / BS_clim  (climatology = training-set RI rate)
  POD @ 40 %          — Probability of Detection  = TP / (TP + FN)
  FAR @ 40 %          — False Alarm Ratio         = FP / (TP + FP)  [meteorological]

SHIPS-RII benchmarks to beat: AUC ~0.78, BSS ~0.08, POD ~0.42, FAR ~0.72.

Usage:
    python model/train_xgboost.py
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
from sklearn.metrics import confusion_matrix, roc_auc_score

# ---------------------------------------------------------------------------
# Repo-root path setup — lets us import from data.scripts as a package
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.scripts.build_training_data import ALL_FEATURE_COLUMNS  # noqa: E402

import xgboost as xgb  # noqa: E402

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TRAINING_DATA_PATH = PROCESSED_DIR / "training_data.parquet"

ARTIFACTS_DIR = REPO_ROOT / "model" / "artifacts"
TEST_PREDS_PATH = ARTIFACTS_DIR / "xgb_test_preds.parquet"

# Temporal split: test set covers storms first observed in [TEST_YEAR, 2023]
TEST_YEAR: int = 2018

# Fraction of training-pool rows (by time) held out for early-stopping validation
VAL_FRAC: float = 0.15

# XGBoost hyperparameters — sourced from CLAUDE.md spec.
# scale_pos_weight is excluded here and computed from the actual training set.
# use_label_encoder was removed in XGBoost 2.0 — do not include it.
XGB_BASE_PARAMS: dict = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "early_stopping_rounds": 30,
    "importance_type": "gain",   # gain-based importance via feature_importances_
    "random_state": 42,
    "n_jobs": -1,
}

# Probability threshold for categorical (POD/FAR) evaluation
RI_PROB_THRESHOLD: float = 0.40

# Published SHIPS-RII operational benchmarks
# (Kaplan & DeMaria 2003; NHC annual verification reports)
SHIPS_RII_BENCHMARKS: dict[str, float] = {
    "auc": 0.78,
    "bss": 0.08,
    "pod": 0.42,
    "far": 0.72,
}

# Top-N features to display in the importance table
TOP_N_FEATURES: int = 15


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_training_data(path: Path = TRAINING_DATA_PATH) -> pd.DataFrame:
    """Load the merged, imputed training DataFrame.

    Args:
        path: Path to training_data.parquet produced by build_training_data.py.

    Returns:
        DataFrame containing ALL_FEATURE_COLUMNS, storm_id, datetime, and ri_label.

    Raises:
        FileNotFoundError: If the Parquet file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found: {path}\n"
            "Run first: python data/scripts/build_training_data.py"
        )
    logger.info(f"Loading {path.name} …")
    df = pd.read_parquet(path)
    logger.info(
        f"  → {len(df):,} rows | {df['storm_id'].nunique():,} storms | "
        f"RI rate {df['ri_label'].mean() * 100:.1f}%"
    )
    return df


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------


def temporal_split(
    df: pd.DataFrame,
    test_year: int = TEST_YEAR,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train, validation, and test by storm first-obs year.

    Splitting strategy:
      1. Compute each storm's first observation year.
      2. Storms first seen before ``test_year`` → training pool.
         Storms first seen in [test_year, 2023]  → test set.
      3. Within the training pool, rows in the last ``val_frac`` fraction of time
         become the validation set (used only for early stopping).

    All rows for a given storm go to the same partition, preventing the look-ahead
    bias that arises when the same storm has rows in both train and test.

    Args:
        df:        Full training DataFrame (already filtered to 1982–2023).
        test_year: First year (inclusive) of the held-out test window.
        val_frac:  Fraction of training-pool rows by time reserved for validation.

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames.
    """
    # Map each storm → year of its first observation
    storm_first_year: pd.Series = (
        df.groupby("storm_id")["datetime"].min().dt.year
    )

    train_pool_storms = storm_first_year[storm_first_year < test_year].index
    test_storms = storm_first_year[storm_first_year >= test_year].index

    train_pool = df[df["storm_id"].isin(train_pool_storms)].copy()
    test_df = df[df["storm_id"].isin(test_storms)].copy()

    # Validation: last val_frac of training-pool rows by time
    sorted_pool = train_pool.sort_values("datetime")
    val_cutoff_idx = int(len(sorted_pool) * (1.0 - val_frac))
    val_cutoff_dt: pd.Timestamp = sorted_pool.iloc[val_cutoff_idx]["datetime"]

    train_df = train_pool[train_pool["datetime"] < val_cutoff_dt].copy()
    val_df = train_pool[train_pool["datetime"] >= val_cutoff_dt].copy()

    # Log split summary
    train_years = (
        train_df["datetime"].dt.year.min(),
        train_df["datetime"].dt.year.max(),
    )
    val_years = (
        val_df["datetime"].dt.year.min(),
        val_df["datetime"].dt.year.max(),
    )
    test_years = (
        test_df["datetime"].dt.year.min(),
        test_df["datetime"].dt.year.max(),
    )

    logger.info("Temporal split:")
    logger.info(
        f"  Train : {len(train_df):>7,} rows | "
        f"{train_df['storm_id'].nunique():,} storms | "
        f"years {train_years[0]}–{train_years[1]} | "
        f"RI rate {train_df['ri_label'].mean() * 100:.1f}%"
    )
    logger.info(
        f"  Val   : {len(val_df):>7,} rows | "
        f"{val_df['storm_id'].nunique():,} storms | "
        f"years {val_years[0]}–{val_years[1]} | "
        f"RI rate {val_df['ri_label'].mean() * 100:.1f}%"
    )
    logger.info(
        f"  Test  : {len(test_df):>7,} rows | "
        f"{test_df['storm_id'].nunique():,} storms | "
        f"years {test_years[0]}–{test_years[1]} | "
        f"RI rate {test_df['ri_label'].mean() * 100:.1f}%"
    )

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Feature matrices
# ---------------------------------------------------------------------------


def build_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """Extract NumPy feature matrices and label vectors from the split DataFrames.

    Args:
        train_df: Training partition DataFrame.
        val_df:   Validation partition DataFrame.
        test_df:  Test partition DataFrame.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    X_train = train_df[ALL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_df["ri_label"].to_numpy(dtype=np.int8)

    X_val = val_df[ALL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_val = val_df["ri_label"].to_numpy(dtype=np.int8)

    X_test = test_df[ALL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = test_df["ri_label"].to_numpy(dtype=np.int8)

    logger.info(
        f"Feature matrix shapes — "
        f"train {X_train.shape}, val {X_val.shape}, test {X_test.shape}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# Model construction & training
# ---------------------------------------------------------------------------


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Compute the neg:pos ratio for XGBoost class imbalance correction.

    This is the exact ratio of negative (non-RI) to positive (RI) labels in
    the training set, as recommended by the XGBoost documentation.

    Args:
        y_train: Binary label array (0/1).

    Returns:
        scale_pos_weight value to pass to XGBClassifier.
    """
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    ratio = n_neg / n_pos
    logger.info(
        f"Class balance — neg: {n_neg:,} | pos: {n_pos:,} | "
        f"scale_pos_weight: {ratio:.2f}"
    )
    return ratio


def build_model(scale_pos_weight: float) -> xgb.XGBClassifier:
    """Construct the XGBClassifier with the CLAUDE.md hyperparameter spec.

    Args:
        scale_pos_weight: Neg:pos ratio computed from the training set.

    Returns:
        Untrained XGBClassifier instance.
    """
    params = {**XGB_BASE_PARAMS, "scale_pos_weight": scale_pos_weight}
    return xgb.XGBClassifier(**params)


def train_model(
    model: xgb.XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> xgb.XGBClassifier:
    """Fit the XGBClassifier with early stopping monitored on the validation AUC.

    Training stops when validation AUC fails to improve for
    ``early_stopping_rounds`` consecutive rounds.  The model's internal
    ``best_iteration`` is used automatically for subsequent predictions.

    Args:
        model:   Untrained XGBClassifier.
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val:   Validation feature matrix (early stopping only).
        y_val:   Validation labels.

    Returns:
        Fitted XGBClassifier.
    """
    logger.info(
        f"Training XGBoost (max {model.n_estimators} rounds, "
        f"early_stopping={model.early_stopping_rounds}) …"
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iter = model.best_iteration
    best_score = model.best_score
    logger.success(
        f"Training complete — best iteration: {best_iter} | "
        f"val AUC: {best_score:.4f}"
    )
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def brier_skill_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    clim_rate: float,
) -> tuple[float, float, float]:
    """Compute Brier Score and Brier Skill Score relative to climatology.

    BSS = 1 - BS / BS_clim, where:
      BS_clim = mean((p_clim - y_true)^2)  using the training-set RI rate.

    A BSS > 0 means the model outperforms climatological persistence.
    The NHC considers BSS > 0.10–0.15 operationally meaningful.

    Args:
        y_true:    True binary labels.
        y_prob:    Predicted probabilities.
        clim_rate: Training-set RI base rate (climatology reference).

    Returns:
        Tuple of (brier_score, brier_clim, bss).
    """
    bs = float(np.mean((y_prob - y_true) ** 2))
    bs_clim = float(np.mean((clim_rate - y_true) ** 2))
    bss = 1.0 - bs / bs_clim if bs_clim > 0 else float("nan")
    return bs, bs_clim, bss


def pod_and_far(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = RI_PROB_THRESHOLD,
) -> tuple[float, float, int, int, int, int]:
    """Compute meteorological POD and FAR at a probability threshold.

    Definitions (consistent with NHC verification reports and Kaplan & DeMaria 2003):
      POD = TP / (TP + FN)  — fraction of RI events correctly forecast
      FAR = FP / (TP + FP)  — fraction of RI forecasts that were false alarms
                               (False Alarm Ratio, not False Alarm Rate)

    Args:
        y_true:    True binary labels.
        y_prob:    Predicted probabilities.
        threshold: Probability cutoff for a positive (RI) forecast.

    Returns:
        Tuple of (pod, far, tp, fp, fn, tn).
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    return float(pod), float(far), int(tp), int(fp), int(fn), int(tn)


def evaluate(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, float]:
    """Run the full evaluation suite on the held-out test set.

    Computes and prints:
      - AUC-ROC
      - Brier Score and Brier Skill Score vs. climatology
      - POD and FAR at RI_PROB_THRESHOLD
      - Comparison against SHIPS-RII published benchmarks
      - Feature importance table (top TOP_N_FEATURES by gain)

    Args:
        model:   Trained XGBClassifier.
        X_test:  Test feature matrix.
        y_test:  Test labels.
        y_train: Training labels (used for climatological BSS baseline).

    Returns:
        Dict mapping metric name to float value.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    clim_rate = float(y_train.mean())

    auc = roc_auc_score(y_test, y_prob)
    bs, bs_clim, bss = brier_skill_score(y_test, y_prob, clim_rate)
    pod, far, tp, fp, fn, tn = pod_and_far(y_test, y_prob, RI_PROB_THRESHOLD)

    metrics = {"auc": auc, "bss": bss, "pod": pod, "far": far, "bs": bs}

    # --- Feature importance ---
    importances = model.feature_importances_
    imp_df = (
        pd.DataFrame({"feature": ALL_FEATURE_COLUMNS, "gain": importances})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    imp_df["gain_pct"] = 100.0 * imp_df["gain"] / imp_df["gain"].sum()

    # --- Print report ---
    bench = SHIPS_RII_BENCHMARKS
    w = 60

    print()
    print("=" * w)
    print("  XGBoost RI Model — Test-Set Evaluation (2018–2023)")
    print("=" * w)
    print(f"  Test observations   : {len(y_test):,}")
    print(f"  RI events in test   : {int(y_test.sum()):,}  "
          f"({y_test.mean() * 100:.1f}%)")
    print(f"  Training clim. rate : {clim_rate * 100:.1f}%")
    print(f"  Best XGB iteration  : {model.best_iteration}")
    print()
    print(f"  {'Metric':<26}  {'Model':>8}  {'SHIPS-RII':>9}  {'Delta':>8}")
    print(f"  {'-'*26}  {'-'*8}  {'-'*9}  {'-'*8}")

    def _row(name: str, key: str, fmt: str = ".4f", higher_better: bool = True) -> None:
        val = metrics[key]
        ref = bench.get(key, float("nan"))
        delta = val - ref
        arrow = "+" if (delta > 0) == higher_better else "-"
        print(
            f"  {name:<26}  {val:{fmt}}  {ref:{fmt}}  "
            f"{arrow}{abs(delta):{fmt}}"
        )

    _row("AUC-ROC", "auc", higher_better=True)
    _row("Brier Skill Score (BSS)", "bss", higher_better=True)
    _row(f"POD  (threshold={RI_PROB_THRESHOLD:.0%})", "pod", higher_better=True)
    _row(f"FAR  (threshold={RI_PROB_THRESHOLD:.0%})", "far", higher_better=False)

    print()
    print(f"  Brier Score         : {bs:.5f}")
    print(f"  Brier (climatology) : {bs_clim:.5f}")
    print()
    print(f"  Confusion matrix @ {RI_PROB_THRESHOLD:.0%} threshold:")
    print(f"    TP (hit)          : {tp:,}")
    print(f"    FP (false alarm)  : {fp:,}")
    print(f"    FN (miss)         : {fn:,}")
    print(f"    TN (correct neg.) : {tn:,}")

    print()
    print(f"  Top {TOP_N_FEATURES} features by gain importance:")
    print(f"  {'Rank':<6}  {'Feature':<22}  {'Gain %':>8}")
    print(f"  {'-'*6}  {'-'*22}  {'-'*8}")
    for rank, row in imp_df.head(TOP_N_FEATURES).iterrows():
        print(f"  {rank + 1:<6}  {row['feature']:<22}  {row['gain_pct']:>7.2f}%")

    print("=" * w)
    print()

    return metrics


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------


def save_artifacts(
    model: xgb.XGBClassifier,
    test_df: pd.DataFrame,
    y_prob_test: np.ndarray,
) -> Path:
    """Persist the trained model and test-set predictions.

    Saves:
      - model/artifacts/xgb_model_{timestamp}.pkl  — timestamped archive copy
      - model/artifacts/xgb_model_latest.pkl       — always points to latest run
      - model/artifacts/xgb_test_preds.parquet     — storm_id, datetime,
                                                       ri_label, xgb_proba

    The test predictions parquet is consumed by ensemble.py to blend XGBoost
    and LSTM scores.

    Args:
        model:       Fitted XGBClassifier.
        test_df:     Test partition DataFrame (provides storm_id, datetime, ri_label).
        y_prob_test: Predicted RI probabilities for the test set.

    Returns:
        Path to the timestamped model file.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    # --- Model ---
    timestamped_path = ARTIFACTS_DIR / f"xgb_model_{timestamp}.pkl"
    with open(timestamped_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.copy2(timestamped_path, ARTIFACTS_DIR / "xgb_model_latest.pkl")
    logger.success(
        f"Model saved → {timestamped_path.name}  "
        f"(also copied to xgb_model_latest.pkl)"
    )

    # --- Test predictions ---
    preds_df = pd.DataFrame(
        {
            "storm_id": test_df["storm_id"].values,
            "datetime": test_df["datetime"].values,
            "ri_label": test_df["ri_label"].values,
            "xgb_proba": y_prob_test.astype(np.float32),
        }
    )
    preds_df.to_parquet(TEST_PREDS_PATH, index=False)
    logger.success(
        f"Test predictions saved → {TEST_PREDS_PATH.name}  "
        f"({len(preds_df):,} rows)"
    )

    return timestamped_path


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def train_xgboost(
    data_path: Path = TRAINING_DATA_PATH,
) -> xgb.XGBClassifier:
    """Run the full XGBoost training pipeline end-to-end.

    Steps:
      1. Load training_data.parquet.
      2. Temporal train / val / test split.
      3. Build NumPy feature matrices.
      4. Compute scale_pos_weight from training set.
      5. Train XGBClassifier with early stopping.
      6. Evaluate on held-out test set (2018–2023).
      7. Save model + test predictions.

    Args:
        data_path: Path to training_data.parquet.

    Returns:
        The fitted XGBClassifier.
    """
    df = load_training_data(data_path)

    train_df, val_df, test_df = temporal_split(df)

    X_train, y_train, X_val, y_val, X_test, y_test = build_matrices(
        train_df, val_df, test_df
    )

    spw = compute_scale_pos_weight(y_train)
    model = build_model(scale_pos_weight=spw)
    model = train_model(model, X_train, y_train, X_val, y_val)

    metrics = evaluate(model, X_test, y_test, y_train)

    y_prob_test = model.predict_proba(X_test)[:, 1]
    save_artifacts(model, test_df, y_prob_test)

    logger.info(
        f"Final scores — AUC: {metrics['auc']:.4f} "
        f"(SHIPS-RII: {SHIPS_RII_BENCHMARKS['auc']:.2f}) | "
        f"BSS: {metrics['bss']:.4f} "
        f"(SHIPS-RII: {SHIPS_RII_BENCHMARKS['bss']:.2f})"
    )
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<8} | {message}")

    train_xgboost()
