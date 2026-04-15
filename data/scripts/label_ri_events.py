"""
label_ri_events.py — Compute binary Rapid Intensification (RI) labels for HURDAT2.

Loads data/processed/hurdat2_clean.parquet, restricts to synoptic-time
observations (00/06/12/18 UTC), computes per-storm 24-hour forward wind change
using a shift(-4) within each storm group, and writes a labeled Parquet file.

RI definition (NHC operational):
    max sustained wind increases by >= 30 kt over the next 24 hours
    within the same storm.  30 kt ≈ 34.5 mph (colloquially "35 mph").

Output columns added:
    wind_change_24h  — float32, kt change from current obs to obs 24 h later
    ri_label         — int8,    1 if wind_change_24h >= RI_THRESHOLD_KT, else 0

Usage:
    python data/scripts/label_ri_events.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Paths & exported constants
# (feature_builder.py and train scripts import these)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
INPUT_PATH = PROCESSED_DIR / "hurdat2_clean.parquet"
OUTPUT_PATH = PROCESSED_DIR / "hurdat2_labeled.parquet"

# NHC operational RI threshold: +30 kt in 24 hours (≈ 34.5 mph / "35 mph")
RI_THRESHOLD_KT: float = 30.0

# Number of 6-hourly steps that equal 24 hours
RI_STEPS_FORWARD: int = 4

# Training window (years, inclusive)
TRAIN_YEAR_START: int = 1980
TRAIN_YEAR_END: int = 2023

# Exported column name constants — import these in feature_builder.py / train scripts
RI_LABEL_COLUMN: str = "ri_label"
WIND_CHANGE_COLUMN: str = "wind_change_24h"

# Synoptic hours used in HURDAT2 main deck (UTC)
SYNOPTIC_HOURS: frozenset[int] = frozenset({0, 6, 12, 18})


# ---------------------------------------------------------------------------
# Core labeling logic
# ---------------------------------------------------------------------------


def filter_synoptic(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only standard 6-hourly synoptic observations.

    HURDAT2 includes "special" advisory entries at non-standard times
    (e.g., 03:00, 21:00, 09:30) identified by the record-identifier field.
    Mixing these into a shift(-4) calculation would produce incorrect 24-hour
    windows, so we restrict to the regular deck: minute=0, hour in {0,6,12,18}.

    Args:
        df: Raw HURDAT2 DataFrame with a datetime column.

    Returns:
        Filtered DataFrame; index is reset.
    """
    mask = (df["datetime"].dt.minute == 0) & (
        df["datetime"].dt.hour.isin(SYNOPTIC_HOURS)
    )
    dropped = (~mask).sum()
    if dropped:
        logger.info(
            f"Dropped {dropped:,} non-synoptic observations "
            f"(kept {mask.sum():,} standard 6-hourly rows)"
        )
    return df[mask].reset_index(drop=True)


def compute_ri_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add wind_change_24h and ri_label columns to a synoptic HURDAT2 DataFrame.

    Strategy:
      1. Sort by storm_id then datetime (must be pre-filtered to synoptic times).
      2. Within each storm group, shift max_wind_kt and datetime backward by
         RI_STEPS_FORWARD (4) rows to obtain the values at t+24h.
      3. Validate that the time gap to the shifted row is exactly 24 hours —
         this catches tracks with gaps (e.g., storm went off-record then resumed).
      4. Compute wind_change_24h = wind_future - wind_current.
      5. Set ri_label = 1 where wind_change_24h >= RI_THRESHOLD_KT.
      6. Drop rows where wind_change_24h is NaN (final 4 obs of each storm and
         any rows where the 24-h window has a gap); these cannot be used as
         training examples since the label is undefined.

    Args:
        df: Synoptic-only HURDAT2 DataFrame (output of filter_synoptic).

    Returns:
        DataFrame with WIND_CHANGE_COLUMN and RI_LABEL_COLUMN added and NaN
        label rows removed.
    """
    df = df.sort_values(["storm_id", "datetime"]).reset_index(drop=True)

    # Shift wind speed and datetime forward within each storm group
    grp = df.groupby("storm_id", sort=False)
    wind_future: pd.Series = grp["max_wind_kt"].shift(-RI_STEPS_FORWARD)
    time_future: pd.Series = grp["datetime"].shift(-RI_STEPS_FORWARD)

    # Validate time gap is exactly 24 hours (catches track discontinuities)
    time_gap_h: pd.Series = (
        time_future - df["datetime"]
    ).dt.total_seconds() / 3600.0
    valid_window: pd.Series = time_gap_h == float(RI_STEPS_FORWARD * 6)

    # wind_change_24h: NaN where window is invalid or wind_future is NaN
    wind_change = wind_future - df["max_wind_kt"]
    wind_change = wind_change.where(valid_window, other=np.nan)

    df[WIND_CHANGE_COLUMN] = wind_change.astype(np.float32)

    # ri_label: 1/0 int8 — only defined where wind_change_24h is not NaN
    ri_label = (wind_change >= RI_THRESHOLD_KT).astype(np.int8)
    df[RI_LABEL_COLUMN] = ri_label.where(wind_change.notna(), other=np.nan)

    # Count rows dropped for transparency
    n_before = len(df)
    df = df.dropna(subset=[WIND_CHANGE_COLUMN]).reset_index(drop=True)
    df[RI_LABEL_COLUMN] = df[RI_LABEL_COLUMN].astype(np.int8)
    logger.info(
        f"Dropped {n_before - len(df):,} rows with undefined 24-h window "
        f"(last {RI_STEPS_FORWARD} obs per storm or track gaps); "
        f"{len(df):,} labeled rows remain"
    )
    return df


def filter_training_window(
    df: pd.DataFrame,
    year_start: int = TRAIN_YEAR_START,
    year_end: int = TRAIN_YEAR_END,
) -> pd.DataFrame:
    """Restrict the labeled dataset to the satellite-era training window.

    Pre-1980 data is excluded because:
      - Reconnaissance aircraft coverage was inconsistent.
      - No GOES satellite imagery exists for alignment with GOES-16 features.
      - Wind measurements are less reliable in the pre-Dvorak era.

    Args:
        df:         Labeled DataFrame.
        year_start: First year to include (inclusive).
        year_end:   Last year to include (inclusive).

    Returns:
        Filtered DataFrame; index is reset.
    """
    mask = df["datetime"].dt.year.between(year_start, year_end)
    dropped = (~mask).sum()
    logger.info(
        f"Training window filter ({year_start}–{year_end}): "
        f"dropped {dropped:,} pre-{year_start} rows, "
        f"{mask.sum():,} rows remain"
    )
    return df[mask].reset_index(drop=True)


def print_class_balance(df: pd.DataFrame) -> None:
    """Print a class balance report for the RI label.

    Args:
        df: Labeled and filtered DataFrame containing RI_LABEL_COLUMN.
    """
    total = len(df)
    ri_events = int(df[RI_LABEL_COLUMN].sum())
    non_ri = total - ri_events
    ri_rate = 100.0 * ri_events / total if total > 0 else 0.0

    storms = df["storm_id"].nunique()
    years = (
        df["datetime"].dt.year.min(),
        df["datetime"].dt.year.max(),
    )
    basins = df["basin"].value_counts().to_dict() if "basin" in df.columns else {}

    print()
    print("=" * 52)
    print("  HURDAT2 RI Label — Class Balance Report")
    print("=" * 52)
    print(f"  Training window     : {years[0]}–{years[1]}")
    print(f"  Storms              : {storms:,}")
    if basins:
        for basin, count in sorted(basins.items()):
            print(f"    {basin:<18}: {count:,} obs")
    print(f"  Total observations  : {total:,}")
    print(f"  RI events  (label=1): {ri_events:,}  ({ri_rate:.1f}%)")
    print(f"  Non-RI     (label=0): {non_ri:,}  ({100 - ri_rate:.1f}%)")
    print(f"  Imbalance ratio     : {non_ri / ri_events:.1f}:1  (neg:pos)")
    print(f"  Suggested XGB scale_pos_weight: {non_ri / ri_events:.2f}")
    print(f"  RI threshold used   : >= {RI_THRESHOLD_KT:.0f} kt in 24 h")
    print("=" * 52)
    print()


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def build_labeled_dataset(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Load, label, filter, and save the HURDAT2 training dataset.

    Idempotent — safe to re-run; always overwrites the output file with a
    fresh computation from the source parquet.

    Args:
        input_path:  Path to hurdat2_clean.parquet (output of fetch_hurdat2.py).
        output_path: Destination path for hurdat2_labeled.parquet.

    Returns:
        The final labeled DataFrame.
    """
    logger.info(f"Loading {input_path.name} …")
    df = pd.read_parquet(input_path)
    logger.info(f"  → {len(df):,} rows, {df['storm_id'].nunique():,} storms")

    df = filter_synoptic(df)
    df = compute_ri_labels(df)
    df = filter_training_window(df)

    print_class_balance(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.success(
        f"Saved → {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)"
    )
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<8} | {message}")

    df = build_labeled_dataset()

    logger.info("Sample RI events (label=1):")
    ri_sample = df[df[RI_LABEL_COLUMN] == 1][
        ["storm_id", "name", "datetime", "max_wind_kt", WIND_CHANGE_COLUMN, RI_LABEL_COLUMN]
    ].head(8)
    print(ri_sample.to_string(index=False))

    logger.info("\nSample non-RI rows (label=0):")
    non_ri_sample = df[df[RI_LABEL_COLUMN] == 0][
        ["storm_id", "name", "datetime", "max_wind_kt", WIND_CHANGE_COLUMN, RI_LABEL_COLUMN]
    ].head(8)
    print(non_ri_sample.to_string(index=False))
