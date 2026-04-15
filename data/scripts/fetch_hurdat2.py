"""
fetch_hurdat2.py — Download and parse the NOAA HURDAT2 hurricane database.

Downloads the Atlantic + East Pacific HURDAT2 files from NHC, parses the
custom fixed-width / comma-delimited format, and writes a clean Parquet file
to data/processed/hurdat2_clean.parquet.

HURDAT2 format reference:
  https://www.nhc.noaa.gov/data/hurdat/hurdat2-format-nov2019.pdf

Usage:
    python data/scripts/fetch_hurdat2.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import requests
from loguru import logger

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "hurdat2"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "hurdat2_clean.parquet"

HURDAT2_URLS: dict[str, str] = {
    "atlantic": "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt",
    "epac": "https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2023-042624.txt",
}

# HURDAT2 status codes (Table 2 in format spec)
VALID_STATUSES = {"TD", "TS", "HU", "EX", "SS", "SD", "LO", "WV", "DB"}

# Sentinel for missing numeric values in HURDAT2
HURDAT2_MISSING = -999

# Columns produced by this script — imported by feature_builder.py
HURDAT2_COLUMNS: list[str] = [
    "storm_id",
    "name",
    "datetime",
    "lat",
    "lon",
    "max_wind_kt",
    "min_pressure_mb",
    "status",
]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_hurdat2(basin: str, url: str, raw_dir: Path) -> Path:
    """Download a HURDAT2 text file if not already cached locally.

    Args:
        basin: Short label used for the local filename (e.g. "atlantic").
        url:   Full HTTPS URL of the HURDAT2 .txt file.
        raw_dir: Directory to write the cached file into.

    Returns:
        Path to the local (possibly freshly downloaded) file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"hurdat2_{basin}.txt"

    if dest.exists():
        logger.info(f"[{basin}] already cached at {dest}, skipping download")
        return dest

    logger.info(f"[{basin}] downloading from {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    logger.success(f"[{basin}] saved {dest.stat().st_size / 1024:.1f} KB → {dest}")
    return dest


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------


def _parse_latlon(raw: str) -> float:
    """Convert a HURDAT2 lat/lon string like '28.0N' or '94.8W' to a float.

    Southern latitudes and western longitudes are returned as negative values.

    Args:
        raw: Raw string from the HURDAT2 file.

    Returns:
        Signed float coordinate.

    Raises:
        ValueError: If the hemisphere character is unrecognised.
    """
    raw = raw.strip()
    hemisphere = raw[-1].upper()
    value = float(raw[:-1])
    if hemisphere in ("S", "W"):
        value = -value
    elif hemisphere not in ("N", "E"):
        raise ValueError(f"Unknown hemisphere character: {hemisphere!r}")
    return value


def _to_nullable_int(raw: str) -> float:
    """Parse a HURDAT2 integer field, converting the -999 sentinel to NaN.

    Args:
        raw: Raw string from the HURDAT2 file.

    Returns:
        Float value (NaN when the original is -999).
    """
    value = int(raw.strip())
    return float("nan") if value == HURDAT2_MISSING else float(value)


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------


def _iter_records(lines: list[str]) -> Iterator[dict]:
    """Iterate over all observation records in a HURDAT2 file.

    HURDAT2 layout:
      - Header line: ``<storm_id>, <name>, <n_records>,``
      - Followed by exactly <n_records> data lines, each with 20 comma-
        separated fields.

    Args:
        lines: All text lines from the HURDAT2 file.

    Yields:
        One dict per valid observation with keys matching HURDAT2_COLUMNS.
    """
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        parts = [p.strip() for p in line.split(",")]

        # Header line: first field is a 8-char storm ID (e.g. "AL011851")
        if len(parts[0]) == 8 and parts[0][:2].isalpha():
            storm_id = parts[0]
            name = parts[1].strip()
            n_records = int(parts[2])
            i += 1

            for _ in range(n_records):
                if i >= len(lines):
                    break
                data_line = lines[i].strip()
                i += 1
                if not data_line:
                    continue

                dp = [p.strip() for p in data_line.split(",")]
                if len(dp) < 8:
                    logger.warning(f"Short data line for {storm_id}: {data_line!r}")
                    continue

                # Field layout (0-indexed):
                # 0: YYYYMMDD  1: HHMM  2: record_id  3: status
                # 4: lat       5: lon   6: max_wind   7: min_pressure
                try:
                    dt = pd.Timestamp(
                        year=int(dp[0][:4]),
                        month=int(dp[0][4:6]),
                        day=int(dp[0][6:8]),
                        hour=int(dp[1][:2]),
                        minute=int(dp[1][2:4]),
                    )
                    lat = _parse_latlon(dp[4])
                    lon = _parse_latlon(dp[5])
                    max_wind = _to_nullable_int(dp[6])
                    min_pressure = _to_nullable_int(dp[7])
                    status = dp[3].strip()
                except (ValueError, IndexError) as exc:
                    logger.warning(
                        f"Could not parse data line for {storm_id}: {data_line!r} — {exc}"
                    )
                    continue

                yield {
                    "storm_id": storm_id,
                    "name": name,
                    "datetime": dt,
                    "lat": lat,
                    "lon": lon,
                    "max_wind_kt": max_wind,
                    "min_pressure_mb": min_pressure,
                    "status": status,
                }
        else:
            # Not a header — skip (shouldn't happen in valid files)
            i += 1


def parse_hurdat2(path: Path) -> pd.DataFrame:
    """Parse a single HURDAT2 .txt file into a tidy DataFrame.

    Args:
        path: Local path to the HURDAT2 text file.

    Returns:
        DataFrame with columns defined in HURDAT2_COLUMNS.
    """
    logger.info(f"Parsing {path.name} …")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    records = list(_iter_records(lines))
    df = pd.DataFrame(records, columns=HURDAT2_COLUMNS)

    # Enforce dtypes
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["lat"] = df["lat"].astype(np.float32)
    df["lon"] = df["lon"].astype(np.float32)
    df["max_wind_kt"] = pd.to_numeric(df["max_wind_kt"], errors="coerce").astype(
        np.float32
    )
    df["min_pressure_mb"] = pd.to_numeric(
        df["min_pressure_mb"], errors="coerce"
    ).astype(np.float32)
    df["storm_id"] = df["storm_id"].astype("string")
    df["name"] = df["name"].astype("string")
    df["status"] = df["status"].astype("category")

    logger.success(
        f"  → {len(df):,} observations across "
        f"{df['storm_id'].nunique():,} storms"
    )
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_hurdat2_dataset(
    raw_dir: Path = RAW_DIR,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Download, parse, merge, and save the HURDAT2 dataset.

    Combines the Atlantic and East Pacific basins into a single DataFrame
    and writes it to Parquet.  The function is idempotent — re-running it
    will re-download only files that are missing from the cache.

    Args:
        raw_dir:     Directory where raw .txt files are cached.
        output_path: Destination path for the clean Parquet file.

    Returns:
        The final merged DataFrame.
    """
    frames: list[pd.DataFrame] = []

    for basin, url in HURDAT2_URLS.items():
        raw_path = download_hurdat2(basin, url, raw_dir)
        df_basin = parse_hurdat2(raw_path)
        df_basin["basin"] = basin
        frames.append(df_basin)

    df = pd.concat(frames, ignore_index=True)

    # Sort chronologically within each storm
    df.sort_values(["storm_id", "datetime"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Log summary statistics
    year_range = (
        df["datetime"].dt.year.min(),
        df["datetime"].dt.year.max(),
    )
    ri_window = df[df["datetime"].dt.year.between(1980, 2023)]
    logger.info(
        f"Full dataset: {len(df):,} rows | "
        f"years {year_range[0]}–{year_range[1]} | "
        f"{df['storm_id'].nunique():,} storms"
    )
    logger.info(
        f"Training window (1980–2023): {len(ri_window):,} rows | "
        f"{ri_window['storm_id'].nunique():,} storms"
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.success(f"Saved → {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<8} | {message}")

    df = build_hurdat2_dataset()

    # Quick sanity check
    logger.info("\nSample rows:")
    print(df.head(10).to_string(index=False))
    logger.info(f"\nColumn dtypes:\n{df.dtypes}")
    logger.info(f"\nNull counts:\n{df.isnull().sum()}")
