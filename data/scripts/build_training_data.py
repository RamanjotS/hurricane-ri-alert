"""
build_training_data.py — Merge HURDAT2 labels with SHIPS features into the
final training DataFrame.

Loads data/processed/hurdat2_labeled.parquet and data/processed/ships_clean.parquet,
joins them on (storm_id, datetime) via an inner join, engineers three additional
features, applies quality filters, imputes remaining nulls with per-feature
training-set medians (saved to model/artifacts/imputer_medians.json for inference
reuse), and writes data/processed/training_data.parquet.

Feature engineering
-------------------
wind_change_6h     — max_wind_kt delta vs. previous 6-hourly obs (momentum proxy).
                     NaN when there is no adjacent 6-hour step (first obs per storm
                     or a track gap after the inner join).
hours_since_ts     — hours elapsed since the storm first reached 34 kt (TS strength).
                     Negative for the handful of pre-TS obs that survive the join.
                     NaN for the rare storm that never appears at ≥ 34 kt.
intensity_pct_vmpi — max_wind_kt / VMPI, fraction of maximum potential intensity.
                     NaN when VMPI is null or zero.

Quality filters (applied before imputation)
--------------------------------------------
1. Drop rows where ri_label is null.
2. Drop rows where more than 3 of the 8 SHIPS feature columns are null.

Imputation
----------
Remaining NaN values in ALL_FEATURE_COLUMNS → per-feature median computed on the
surviving training rows.  Medians saved as model/artifacts/imputer_medians.json.

Usage:
    python data/scripts/build_training_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

HURDAT2_LABELED_PATH = PROCESSED_DIR / "hurdat2_labeled.parquet"
SHIPS_CLEAN_PATH = PROCESSED_DIR / "ships_clean.parquet"
OUTPUT_PATH = PROCESSED_DIR / "training_data.parquet"

ARTIFACTS_DIR = REPO_ROOT / "model" / "artifacts"
IMPUTER_MEDIANS_PATH = ARTIFACTS_DIR / "imputer_medians.json"

# The 8 SHIPS predictors — imported from fetch_ships constants for consistency
SHIPS_FEATURE_COLUMNS: list[str] = [
    "SHRD",
    "RSST",
    "RHLO",
    "RHMD",
    "PSLV",
    "OHCL",
    "VMPI",
    "VVAV",
]

# All model-input feature columns (SHIPS + engineered).
# This exact list must be imported by feature_builder.py and train scripts.
ALL_FEATURE_COLUMNS: list[str] = SHIPS_FEATURE_COLUMNS + [
    "wind_change_6h",
    "hours_since_ts",
    "intensity_pct_vmpi",
]

# Quality filter threshold: drop row if strictly more than this many SHIPS
# feature columns are null.
MAX_SHIPS_NULLS: int = 3

# Tropical storm wind threshold (kt) — used to compute hours_since_ts
TS_THRESHOLD_KT: float = 34.0


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the labeled HURDAT2 and clean SHIPS Parquet files.

    Returns:
        Tuple of (hurdat2_labeled, ships_clean) DataFrames.

    Raises:
        FileNotFoundError: If either source file is absent.
    """
    for path in (HURDAT2_LABELED_PATH, SHIPS_CLEAN_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}\n"
                "Run the upstream scripts first:\n"
                "  python data/scripts/fetch_hurdat2.py\n"
                "  python data/scripts/label_ri_events.py\n"
                "  python data/scripts/fetch_ships.py"
            )

    logger.info(f"Loading {HURDAT2_LABELED_PATH.name} …")
    h2 = pd.read_parquet(HURDAT2_LABELED_PATH)
    logger.info(f"  → {len(h2):,} rows | {h2['storm_id'].nunique():,} storms")

    logger.info(f"Loading {SHIPS_CLEAN_PATH.name} …")
    ships = pd.read_parquet(SHIPS_CLEAN_PATH)
    logger.info(f"  → {len(ships):,} rows | {ships['storm_id'].nunique():,} storms")

    return h2, ships


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------


def inner_join(h2: pd.DataFrame, ships: pd.DataFrame) -> pd.DataFrame:
    """Inner-join HURDAT2 labels with SHIPS features on (storm_id, datetime).

    The SHIPS DataFrame carries its own ``basin`` column (a duplicate of the
    one already in HURDAT2).  It is dropped before the merge to avoid
    collision suffixes.

    Args:
        h2:    Labeled HURDAT2 DataFrame.
        ships: Clean SHIPS feature DataFrame.

    Returns:
        Merged DataFrame containing all HURDAT2 columns and the 8 SHIPS
        feature columns.
    """
    # Drop redundant 'basin' from SHIPS (already in HURDAT2 side)
    ships_features = ships.drop(columns=["basin"], errors="ignore")

    n_before_h2 = len(h2)
    n_before_ships = len(ships_features)

    merged = pd.merge(
        h2,
        ships_features,
        on=["storm_id", "datetime"],
        how="inner",
        validate="one_to_one",
    )

    logger.info(
        f"Inner join: {n_before_h2:,} HURDAT2 rows × "
        f"{n_before_ships:,} SHIPS rows → {len(merged):,} matched rows "
        f"({len(merged) / n_before_h2 * 100:.1f}% of HURDAT2 retained)"
    )
    return merged


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def add_wind_change_6h(df: pd.DataFrame) -> pd.DataFrame:
    """Add wind_change_6h: wind speed delta from the previous 6-hourly timestep.

    Computed within each storm group (sorted by datetime).  Set to NaN when
    the preceding obs is absent (first row per storm) or when the actual time
    gap is not exactly 6 hours (track gap after the inner join removed rows).

    This captures short-term intensity momentum: storms that have already been
    rapidly intensifying are more likely to continue doing so.

    Args:
        df: Merged DataFrame, sorted by (storm_id, datetime).

    Returns:
        DataFrame with the ``wind_change_6h`` column appended (float32).
    """
    grp = df.groupby("storm_id", sort=False)

    wind_prev: pd.Series = grp["max_wind_kt"].shift(1)
    time_prev: pd.Series = grp["datetime"].shift(1)

    # Only valid when the gap to the previous obs is exactly 6 hours
    time_gap_h: pd.Series = (
        df["datetime"] - time_prev
    ).dt.total_seconds() / 3600.0
    valid_gap: pd.Series = time_gap_h == 6.0

    wind_change = (df["max_wind_kt"] - wind_prev).where(valid_gap, other=np.nan)
    df["wind_change_6h"] = wind_change.astype(np.float32)

    n_valid = valid_gap.sum()
    n_gap = len(df) - n_valid
    logger.info(
        f"wind_change_6h: {n_valid:,} valid | "
        f"{n_gap:,} NaN (first obs per storm or track gap)"
    )
    return df


def add_hours_since_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Add hours_since_ts: hours elapsed since the storm first reached 34 kt.

    The first-TS datetime is determined from the joined data (rows present in
    both HURDAT2 and SHIPS).  Rows before TS onset will have a negative value;
    storms that never appear at ≥ 34 kt in the joined data will be NaN and
    subsequently imputed.

    This captures storm age: mature, long-lived storms behave differently from
    freshly intensifying systems.

    Args:
        df: Merged DataFrame.

    Returns:
        DataFrame with the ``hours_since_ts`` column appended (float32).
    """
    ts_mask = df["max_wind_kt"] >= TS_THRESHOLD_KT
    first_ts: pd.Series = (
        df[ts_mask]
        .groupby("storm_id")["datetime"]
        .min()
        .rename("_first_ts_dt")
    )

    df = df.join(first_ts, on="storm_id")
    df["hours_since_ts"] = (
        (df["datetime"] - df["_first_ts_dt"]).dt.total_seconds() / 3600.0
    ).astype(np.float32)
    df.drop(columns=["_first_ts_dt"], inplace=True)

    n_nan = df["hours_since_ts"].isna().sum()
    if n_nan:
        logger.warning(
            f"hours_since_ts: {n_nan:,} NaN "
            f"(storms that never appeared at ≥ {TS_THRESHOLD_KT:.0f} kt in joined data)"
        )
    return df


def add_intensity_pct_vmpi(df: pd.DataFrame) -> pd.DataFrame:
    """Add intensity_pct_vmpi: current wind speed as a fraction of VMPI.

    intensity_pct_vmpi = max_wind_kt / VMPI.

    Set to NaN when VMPI is null, zero, or negative (physically invalid).
    Values > 1 are possible when measured intensity exceeds the theoretical
    maximum (VMPI inaccuracy) and are left unchanged.

    This captures how close to its thermodynamic ceiling a storm is: storms
    already near their MPI have less room to intensify rapidly.

    Args:
        df: Merged DataFrame containing ``max_wind_kt`` and ``VMPI``.

    Returns:
        DataFrame with the ``intensity_pct_vmpi`` column appended (float32).
    """
    vmpi_valid = df["VMPI"].notna() & (df["VMPI"] > 0.0)
    ratio = np.where(
        vmpi_valid,
        df["max_wind_kt"] / df["VMPI"],
        np.nan,
    )
    df["intensity_pct_vmpi"] = ratio.astype(np.float32)

    n_nan = (~vmpi_valid).sum()
    logger.info(
        f"intensity_pct_vmpi: {len(df) - n_nan:,} valid | "
        f"{n_nan:,} NaN (VMPI null or ≤ 0)"
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in the correct dependency order.

    Args:
        df: Inner-joined DataFrame, not yet sorted.

    Returns:
        DataFrame with wind_change_6h, hours_since_ts, and intensity_pct_vmpi
        appended.
    """
    # Sort before any group-wise shift operation
    df = df.sort_values(["storm_id", "datetime"]).reset_index(drop=True)

    df = add_wind_change_6h(df)
    df = add_hours_since_ts(df)
    df = add_intensity_pct_vmpi(df)

    return df


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------


def apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with undefined labels or excessive SHIPS feature missingness.

    Two filters applied in order:
      1. Drop rows where ri_label is null (label undefined → unusable for training).
      2. Drop rows where more than MAX_SHIPS_NULLS (3) of the 8 SHIPS feature
         columns are null (too much environmental context is missing to be reliable).

    Args:
        df: DataFrame after feature engineering.

    Returns:
        Filtered DataFrame; index is reset.
    """
    n_start = len(df)

    # Filter 1: undefined RI label
    null_label_mask = df["ri_label"].isna()
    n_null_label = null_label_mask.sum()
    df = df[~null_label_mask].reset_index(drop=True)
    logger.info(f"Quality filter 1 — null ri_label: dropped {n_null_label:,} rows")

    # Filter 2: excessive SHIPS feature nulls
    ships_null_count: pd.Series = df[SHIPS_FEATURE_COLUMNS].isna().sum(axis=1)
    too_sparse_mask = ships_null_count > MAX_SHIPS_NULLS
    n_too_sparse = too_sparse_mask.sum()
    df = df[~too_sparse_mask].reset_index(drop=True)
    logger.info(
        f"Quality filter 2 — >{ MAX_SHIPS_NULLS} SHIPS nulls: "
        f"dropped {n_too_sparse:,} rows"
    )

    logger.info(
        f"Quality filters total: dropped {n_start - len(df):,} rows, "
        f"{len(df):,} rows remain"
    )
    return df


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------


def impute_and_save_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Impute remaining nulls in ALL_FEATURE_COLUMNS with per-feature medians.

    Medians are computed on the surviving training rows (after quality filters)
    and written to IMPUTER_MEDIANS_PATH as JSON so that the real-time inference
    pipeline can apply identical imputation to live feature vectors.

    Args:
        df: Quality-filtered DataFrame.

    Returns:
        DataFrame with zero nulls in ALL_FEATURE_COLUMNS.
    """
    medians: dict[str, float] = {}
    null_before = df[ALL_FEATURE_COLUMNS].isna().sum()
    total_nulls_before = int(null_before.sum())

    for col in ALL_FEATURE_COLUMNS:
        median_val = float(df[col].median())
        medians[col] = median_val
        if df[col].isna().any():
            df[col] = df[col].fillna(median_val)

    # Verify zero nulls remain
    null_after = df[ALL_FEATURE_COLUMNS].isna().sum()
    assert null_after.sum() == 0, (
        f"Imputation incomplete — nulls remain:\n{null_after[null_after > 0]}"
    )

    # Persist medians for inference
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    IMPUTER_MEDIANS_PATH.write_text(
        json.dumps(medians, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.success(
        f"Imputed {total_nulls_before:,} nulls across {len(ALL_FEATURE_COLUMNS)} "
        f"feature columns; medians saved → {IMPUTER_MEDIANS_PATH}"
    )
    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(df: pd.DataFrame) -> None:
    """Print a concise dataset summary after all processing steps.

    Reports:
      - Total rows and RI event rate
      - Feature count (model-input columns)
      - Null count per column (should be zero for all feature columns after
        imputation)

    Args:
        df: Final training DataFrame.
    """
    total = len(df)
    ri_events = int(df["ri_label"].sum())
    ri_rate = 100.0 * ri_events / total if total else 0.0
    feature_count = len(ALL_FEATURE_COLUMNS)

    print()
    print("=" * 60)
    print("  Training Dataset — Final Summary")
    print("=" * 60)
    print(f"  Total rows          : {total:,}")
    print(f"  RI events (label=1) : {ri_events:,}  ({ri_rate:.1f}%)")
    print(f"  Non-RI    (label=0) : {total - ri_events:,}  ({100 - ri_rate:.1f}%)")
    print(f"  Feature columns     : {feature_count}")
    print(f"  Storms              : {df['storm_id'].nunique():,}")
    years = df["datetime"].dt.year.min(), df["datetime"].dt.year.max()
    print(f"  Years               : {years[0]}–{years[1]}")
    print()

    # Null counts across all columns — highlight any non-zero
    print(f"  {'Column':<22}  {'Nulls':>8}")
    print(f"  {'-'*22}  {'-'*8}")
    all_cols = ALL_FEATURE_COLUMNS + ["ri_label", "wind_change_24h"]
    for col in all_cols:
        nulls = int(df[col].isna().sum()) if col in df.columns else -1
        flag = "  ← WARNING" if nulls > 0 else ""
        print(f"  {col:<22}  {nulls:>8,}{flag}")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def build_training_data(
    hurdat2_path: Path = HURDAT2_LABELED_PATH,
    ships_path: Path = SHIPS_CLEAN_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Run the full merge, feature-engineering, and imputation pipeline.

    Idempotent — safe to re-run; always rewrites the output Parquet from the
    source files.

    Args:
        hurdat2_path: Path to hurdat2_labeled.parquet.
        ships_path:   Path to ships_clean.parquet.
        output_path:  Destination path for training_data.parquet.

    Returns:
        The final training DataFrame.
    """
    h2, ships = load_inputs()

    df = inner_join(h2, ships)
    df = engineer_features(df)
    df = apply_quality_filters(df)
    df = impute_and_save_medians(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.success(
        f"Saved → {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)"
    )

    print_summary(df)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<8} | {message}")

    df = build_training_data()
