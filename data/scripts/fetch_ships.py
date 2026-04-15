"""
fetch_ships.py — Download and parse the SHIPS developmental dataset.

Downloads the consolidated SHIPS lsdiag files for the Atlantic and Eastern
Pacific basins from RAMMB/CIRA (Colorado State University) and extracts the
key predictor values at t=0 (current observation time) for each advisory.

SHIPS file format notes (confirmed empirically):
  - Each advisory block starts with a HEAD line, followed by ~70 predictor rows.
  - Every row uses 5-character fixed-width integer fields:
      field[0] = t=-12h, field[1] = t=-6h, field[2] = t=0 ← we want this one
  - Field is blank (not 9999) when that time step has no data.
  - Missing-data sentinel: 9999  (CLAUDE.md mentions -999/-9999, but actual
    files use 9999 for all missing values).
  - Variable name appears as the last uppercase-alpha token on each line.
  - The ocean-heat-content column is named COHC in the file; CLAUDE.md calls
    the same variable OHCL.  We output it as OHCL to match the spec.

Data coverage note:
  SHIPS developmental data begins in 1982 — there is no SHIPS data for 1980
  or 1981.  The output will cover 1982-2023 for both basins.

Scale factors (divide raw integer by scale to get physical units):
  SHRD  ÷ 10  → kt        (850–200 hPa wind shear magnitude)
  RSST  ÷ 10  → °C        (Reynolds sea-surface temperature)
  RHLO  ÷  1  → %         (850–700 hPa relative humidity)
  RHMD  ÷  1  → %         (500–300 hPa relative humidity)
  PSLV  ÷  1  → raw       (200 hPa divergence, ~10⁻⁷ Pa/s units)
  COHC  ÷  1  → kJ/cm²   (column ocean heat content → output col OHCL)
  VMPI  ÷  1  → kt        (maximum potential intensity)
  VVAV  ÷  1  → raw       (850–200 hPa average vertical velocity)

File sizes (approximate, one-time download):
  Atlantic 7-day (1982-2023): ~359 MB
  EPac     7-day (1982-2023): ~439 MB

Usage:
    python data/scripts/fetch_ships.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Generator, Iterator

import numpy as np
import pandas as pd
import requests
from loguru import logger

# ---------------------------------------------------------------------------
# Paths & public constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "ships"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "ships_clean.parquet"

# Consolidated 7-day files: cover 1982-2023 for both basins
SHIPS_URLS: dict[str, str] = {
    "atlantic": (
        "https://rammb-data.cira.colostate.edu/ships/data/AL/"
        "lsdiaga_1982_2023_sat_ts_7day.txt"
    ),
    "epac": (
        "https://rammb-data.cira.colostate.edu/ships/data/EP/"
        "lsdiage_1982_2023_sat_ts_7day.txt"
    ),
}

# Fixed-width format constants
_FIELD_WIDTH: int = 5          # each time-step column is 5 chars wide
_T0_FIELD_INDEX: int = 2       # t=0 is the 3rd field (0-indexed)
_T0_START: int = _T0_FIELD_INDEX * _FIELD_WIDTH   # char offset 10
_T0_END: int = _T0_START + _FIELD_WIDTH            # char offset 15
_MISSING_SENTINEL: int = 9999  # value that means "no data"

# Map: SHIPS file variable name → output DataFrame column name
# COHC in the file = OHCL in CLAUDE.md spec
_VARIABLE_MAP: dict[str, str] = {
    "SHRD": "SHRD",  # 850-200 hPa wind shear magnitude
    "RSST": "RSST",  # Reynolds SST
    "RHLO": "RHLO",  # 850-700 hPa relative humidity
    "RHMD": "RHMD",  # 500-300 hPa relative humidity
    "PSLV": "PSLV",  # 200 hPa divergence
    "COHC": "OHCL",  # column OHC → renamed to match CLAUDE.md spec
    "VMPI": "VMPI",  # maximum potential intensity
    "VVAV": "VVAV",  # 850-200 hPa average vertical velocity
}
_TARGET_VARS: frozenset[str] = frozenset(_VARIABLE_MAP)

# Scale factors: divide raw integer by this value to get physical units
_SCALE_FACTORS: dict[str, float] = {
    "SHRD": 10.0,
    "RSST": 10.0,
    "RHLO": 1.0,
    "RHMD": 1.0,
    "PSLV": 1.0,
    "COHC": 1.0,
    "VMPI": 1.0,
    "VVAV": 1.0,
}

# Exported constant — import this in feature_builder.py and train scripts
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


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_ships(basin: str, url: str, raw_dir: Path) -> Path:
    """Stream-download a SHIPS lsdiag file, skipping if already cached.

    The file is written atomically via a temporary file so a failed mid-download
    never leaves a corrupt cache entry.

    Args:
        basin:   Short label for the file (e.g. "atlantic").
        url:     Full HTTPS URL of the SHIPS .txt file.
        raw_dir: Directory for cached raw files.

    Returns:
        Path to the local cached file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"ships_{basin}_7day.txt"

    if dest.exists():
        size_mb = dest.stat().st_size / 1_048_576
        logger.info(
            f"[{basin}] already cached at {dest.name} ({size_mb:.0f} MB), "
            "skipping download"
        )
        return dest

    logger.info(f"[{basin}] downloading {url}")
    logger.warning(
        f"[{basin}] NOTE: this file is ~360-440 MB — download may take "
        "several minutes on slower connections"
    )

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    # Atomic write via temp file in same directory
    tmp_path: Path | None = None
    try:
        fd, tmp_str = tempfile.mkstemp(dir=raw_dir, suffix=".tmp")
        tmp_path = Path(tmp_str)
        bytes_written = 0
        with open(fd, "wb") as f:
            for chunk in response.iter_content(chunk_size=65_536):
                f.write(chunk)
                bytes_written += len(chunk)
                if bytes_written % (32 * 1_048_576) < 65_536:
                    logger.debug(
                        f"[{basin}] {bytes_written / 1_048_576:.0f} MB downloaded…"
                    )
        tmp_path.rename(dest)
        logger.success(
            f"[{basin}] saved {dest.stat().st_size / 1_048_576:.0f} MB → {dest.name}"
        )
    except Exception:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
        raise

    return dest


# ---------------------------------------------------------------------------
# HEAD line parser
# ---------------------------------------------------------------------------


def _parse_head(tokens: list[str]) -> dict | None:
    """Parse a SHIPS HEAD line into storm metadata.

    HEAD line field layout (split by whitespace, rightmost first):
      tokens[-1] = 'HEAD'
      tokens[-2] = storm_id  e.g. 'AL011982'
      tokens[-3] = MSLP (mb)
      tokens[-4] = longitude (degrees W, positive)
      tokens[-5] = latitude  (degrees N)
      tokens[-6] = VMAX      (kt)
      tokens[-7] = hour      (UTC, 0/6/12/18)
      tokens[-8] = YYMMDD    (2-digit year)
      tokens[-9] = name      (first 4 chars)

    Args:
        tokens: Whitespace-split tokens from the HEAD line.

    Returns:
        Dict with storm_id and datetime, or None on parse failure.
    """
    if len(tokens) < 9:
        return None
    try:
        storm_id = tokens[-2]
        if len(storm_id) != 8:
            return None  # unexpected format

        date_str = tokens[-8]
        if len(date_str) != 6:
            return None

        yy = int(date_str[:2])
        mm = int(date_str[2:4])
        dd = int(date_str[4:6])
        # 2-digit year convention: 50+ → 1900s, <50 → 2000s
        year = 1900 + yy if yy >= 50 else 2000 + yy
        hour = int(tokens[-7])

        dt = pd.Timestamp(year=year, month=mm, day=dd, hour=hour)
        return {"storm_id": storm_id, "datetime": dt}
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Predictor row parser
# ---------------------------------------------------------------------------


def _extract_t0(line: str) -> float | None:
    """Extract the t=0 (current-time) value from a SHIPS predictor row.

    Uses fixed-width parsing: field[2] occupies chars 10–14 (0-indexed).
    Blank fields (no data for that time step) and 9999 both map to None.

    Args:
        line: Raw text line from the SHIPS file.

    Returns:
        Integer value as float, or None if missing/invalid.
    """
    if len(line) < _T0_END:
        return None
    raw = line[_T0_START:_T0_END].strip()
    if not raw:
        return None
    try:
        val = int(raw)
        return None if val == _MISSING_SENTINEL else float(val)
    except ValueError:
        return None


def _label_from_line(line: str) -> str | None:
    """Return the SHIPS variable label at the end of a line, or None.

    Labels are the rightmost uppercase-alpha token (e.g. 'VMAX', 'SHRD').
    Lines ending in 'HEAD' or 'TIME' return those strings directly.
    Trailing digit flags after the label (e.g. 'RSST    0') are ignored.

    Args:
        line: Raw text line from the SHIPS file.

    Returns:
        Uppercase string label, or None if not found.
    """
    for tok in reversed(line.split()):
        if tok.isalpha() and tok.isupper():
            return tok
    return None


# ---------------------------------------------------------------------------
# Advisory block iterator
# ---------------------------------------------------------------------------


def _iter_advisories(
    lines: Iterator[str],
    basin: str,
) -> Generator[dict, None, None]:
    """Stream-parse a SHIPS lsdiag file, yielding one dict per advisory.

    Each dict contains storm_id, datetime, basin, and one key per
    SHIPS_FEATURE_COLUMNS.  Features are NaN if absent from that block.

    Args:
        lines:  Iterator over raw text lines from the SHIPS file.
        basin:  Basin label string (e.g. 'atlantic').

    Yields:
        One record dict per advisory observation.
    """
    empty_features: dict[str, float] = {col: np.nan for col in SHIPS_FEATURE_COLUMNS}
    current_meta: dict | None = None
    current_features: dict[str, float] = dict(empty_features)
    parse_errors = 0
    advisories_seen = 0

    for raw_line in lines:
        line = raw_line.rstrip("\n\r")
        if not line.strip():
            continue

        label = _label_from_line(line)
        if label is None:
            continue

        if label == "HEAD":
            # Flush completed advisory block before starting a new one
            if current_meta is not None:
                advisories_seen += 1
                yield {**current_meta, "basin": basin, **current_features}

            tokens = line.split()
            meta = _parse_head(tokens)
            if meta is None:
                parse_errors += 1
                if parse_errors <= 5:
                    logger.warning(
                        f"[{basin}] Could not parse HEAD line: {line[:80]!r}"
                    )
                current_meta = None
            else:
                current_meta = meta
                current_features = dict(empty_features)

        elif label == "TIME":
            pass  # time-offset row; not needed since we use fixed-width

        elif label in _TARGET_VARS and current_meta is not None:
            raw_val = _extract_t0(line)
            if raw_val is not None:
                out_col = _VARIABLE_MAP[label]
                scale = _SCALE_FACTORS[label]
                current_features[out_col] = raw_val / scale

    # Flush the final advisory block
    if current_meta is not None:
        advisories_seen += 1
        yield {**current_meta, "basin": basin, **current_features}

    logger.info(
        f"[{basin}] parsed {advisories_seen:,} advisories"
        + (f" ({parse_errors} HEAD parse errors)" if parse_errors else "")
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def parse_ships_file(path: Path, basin: str) -> pd.DataFrame:
    """Parse a cached SHIPS lsdiag .txt file into a tidy DataFrame.

    Args:
        path:  Local path to the raw SHIPS text file.
        basin: Basin label for the 'basin' column.

    Returns:
        DataFrame with columns: storm_id, datetime, basin,
        and one column per SHIPS_FEATURE_COLUMNS.
    """
    logger.info(f"[{basin}] parsing {path.name} ({path.stat().st_size / 1_048_576:.0f} MB)…")

    cols = ["storm_id", "datetime", "basin"] + SHIPS_FEATURE_COLUMNS

    with open(path, "r", encoding="latin-1", errors="replace") as fh:
        records = list(_iter_advisories(fh, basin))

    if not records:
        logger.warning(f"[{basin}] no records parsed — returning empty DataFrame")
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(records, columns=cols)

    # Enforce dtypes
    df["storm_id"] = df["storm_id"].astype("string")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["basin"] = df["basin"].astype("string")
    for col in SHIPS_FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    df.sort_values(["storm_id", "datetime"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    storms = df["storm_id"].nunique()
    years = df["datetime"].dt.year.min(), df["datetime"].dt.year.max()
    logger.success(
        f"[{basin}] {len(df):,} advisories | "
        f"{storms:,} storms | years {years[0]}–{years[1]}"
    )
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print a dataset summary: years covered, total rows, null rate per feature.

    Args:
        df: Final merged SHIPS DataFrame.
    """
    years = df["datetime"].dt.year.min(), df["datetime"].dt.year.max()
    storms = df["storm_id"].nunique()

    print()
    print("=" * 58)
    print("  SHIPS Developmental Dataset — Summary")
    print("=" * 58)
    print(f"  Years covered       : {years[0]}–{years[1]}")
    print(f"    NOTE: No SHIPS data exists before 1982.")
    if "basin" in df.columns:
        for basin, grp in df.groupby("basin"):
            print(
                f"    {basin:<18}: {len(grp):,} advisories  "
                f"| {grp['storm_id'].nunique():,} storms"
            )
    print(f"  Total advisories    : {len(df):,}")
    print(f"  Total storms        : {storms:,}")
    print()
    print(f"  {'Feature':<8}  {'Non-null':>10}  {'Null':>8}  {'Null %':>7}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}")
    for col in SHIPS_FEATURE_COLUMNS:
        non_null = df[col].notna().sum()
        null_n = df[col].isna().sum()
        null_pct = 100.0 * null_n / len(df) if len(df) else 0.0
        col_min = df[col].min() if non_null else float("nan")
        col_max = df[col].max() if non_null else float("nan")
        print(
            f"  {col:<8}  {non_null:>10,}  {null_n:>8,}  {null_pct:>6.1f}%"
            f"  {col_min:>8.2f}  {col_max:>8.2f}"
        )
    print("=" * 58)
    print()


def build_ships_dataset(
    raw_dir: Path = RAW_DIR,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Download, parse, merge, and save the SHIPS feature dataset.

    Idempotent: raw files are cached and skipped if already present.
    The output Parquet is always rewritten from the cached raw files.

    Args:
        raw_dir:     Directory for caching raw SHIPS .txt files.
        output_path: Destination Parquet path.

    Returns:
        Final merged DataFrame.
    """
    frames: list[pd.DataFrame] = []

    for basin, url in SHIPS_URLS.items():
        try:
            raw_path = download_ships(basin, url, raw_dir)
        except requests.RequestException as exc:
            logger.error(
                f"[{basin}] download failed: {exc} — skipping this basin"
            )
            continue

        try:
            df_basin = parse_ships_file(raw_path, basin)
        except Exception as exc:
            logger.error(
                f"[{basin}] parse failed: {exc} — skipping this basin"
            )
            continue

        if not df_basin.empty:
            frames.append(df_basin)

    if not frames:
        raise RuntimeError(
            "No SHIPS data parsed successfully — cannot write output."
        )

    df = pd.concat(frames, ignore_index=True)
    df.sort_values(["storm_id", "datetime"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print_summary(df)

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

    df = build_ships_dataset()

    logger.info("Sample rows (random advisory with all features present):")
    complete = df.dropna(subset=SHIPS_FEATURE_COLUMNS)
    if not complete.empty:
        print(
            complete.sample(min(5, len(complete)), random_state=42)[
                ["storm_id", "datetime", "basin"] + SHIPS_FEATURE_COLUMNS
            ].to_string(index=False)
        )
