"""
fetch_goes16.py — Extract GOES-16 Band 13 brightness-temperature features
for each storm observation in data/processed/training_data.parquet.

GOES-16 ABI-L2-CMIPF (Cloud and Moisture Imagery Full Disk, Band 13, 10.3 µm)
provides clean longwave IR brightness temperature at ~2 km nadir resolution
every 10 minutes (Mode 6 from 2019-04-02) or 15 minutes (Mode 3, earlier).

For each row in training_data.parquet this script:
  1. Finds the nearest ABI-L2-CMIPF C13 scan within ±30 minutes on S3.
  2. Caches the granule to data/raw/goes16/ (skipped if already present).
  3. Extracts a ±3° lat/lon box centred on the storm using the GOES-R
     geostationary forward projection (no external geo library required).
  4. Computes 5 scalar brightness-temperature features plus coverage fraction.
  5. Saves results to data/processed/goes16_features.parquet.

GOES-16 data availability: GOES-16 was declared operational 2017-12-18.
Observations before 2018-01-01 receive all-NaN features without any S3 access.

Feature definitions
-------------------
std_bt          Standard deviation of BT in the storm box (K).
area_deep_conv  Fraction of valid pixels with BT < 220 K (deep convective tops).
min_bt          Coldest pixel in the box (K) — proxy for highest cloud top.
sym_index       Std-dev of per-quadrant (NW/NE/SW/SE) mean BT — convective
                asymmetry; high = lopsided convection, low = symmetric.
ot_count        Fraction of valid pixels with BT < 200 K (overshooting tops).
goes_coverage   Fraction of box pixels that contained valid (non-fill) data.
                Use goes_coverage < 0.5 as a QC threshold in downstream scripts.

Usage:
    python data/scripts/fetch_goes16.py
"""

from __future__ import annotations

import multiprocessing as mp
import re
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from loguru import logger

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_GOES16_DIR = REPO_ROOT / "data" / "raw" / "goes16"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TRAINING_DATA_PATH = PROCESSED_DIR / "training_data.parquet"
OUTPUT_PATH = PROCESSED_DIR / "goes16_features.parquet"

S3_BUCKET = "noaa-goes16"
S3_PRODUCT = "ABI-L2-CMIPF"
GOES16_BAND = 13  # clean longwave IR, 10.3 µm

# GOES-16 declared operational 2017-12-18; use 2018-01-01 as safe boundary
GOES16_OPERATIONAL_DATE = datetime(2018, 1, 1, tzinfo=timezone.utc)

# Spatial box half-width in degrees (±3° lat/lon around storm centre)
BOX_DEG: float = 3.0

# Temporal search window in minutes
TIME_WINDOW_MIN: int = 30

# Number of parallel worker processes
N_WORKERS: int = 4

# Brightness-temperature thresholds (Kelvin)
DEEP_CONV_THRESHOLD_K: float = 220.0  # deep convective cloud tops
OT_THRESHOLD_K: float = 200.0         # overshooting tops proxy

# Exported constant — import this in feature_builder.py and train scripts
GOES_FEATURE_COLUMNS: list[str] = [
    "std_bt",
    "area_deep_conv",
    "min_bt",
    "sym_index",
    "ot_count",
]

# ---------------------------------------------------------------------------
# Coordinate conversion: (lat, lon) → GOES-16 ABI scan angles
# ---------------------------------------------------------------------------


def latlon_to_scanangle(
    lat_deg: float,
    lon_deg: float,
    lon_0_deg: float,
    H_m: float,
    r_eq_m: float,
    r_pol_m: float,
) -> tuple[float | None, float | None]:
    """Convert geographic coordinates to GOES-16 ABI scan angles (radians).

    Implements the forward geostationary projection defined in NOAA's GOES-R
    ABI L1b Product User Guide (NOAA-GOES-R-PUG-L1B-0123, §4.2).

    Args:
        lat_deg:  Geographic latitude in degrees (positive north).
        lon_deg:  Geographic longitude in degrees (positive east).
        lon_0_deg: Satellite sub-point longitude in degrees (positive east).
        H_m:      Distance from Earth's centre to satellite in metres
                  (= perspective_point_height + semi_major_axis from file).
        r_eq_m:   Semi-major axis of reference ellipsoid in metres.
        r_pol_m:  Semi-minor axis of reference ellipsoid in metres.

    Returns:
        (x, y) scan angles in radians, or (None, None) if the point is
        on the far side of the Earth (not visible to the satellite).
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    lon_0 = np.radians(lon_0_deg)

    # Geocentric latitude (accounts for Earth's oblateness)
    lat_c = np.arctan((r_pol_m / r_eq_m) ** 2 * np.tan(lat))

    # Distance from Earth centre to the surface point on the reference ellipsoid
    r_c = r_pol_m / np.sqrt(
        1.0 - ((r_eq_m**2 - r_pol_m**2) / r_eq_m**2) * np.cos(lat_c) ** 2
    )

    # Vector components from satellite to Earth surface point
    s_x = H_m - r_c * np.cos(lat_c) * np.cos(lon - lon_0)
    s_y = -r_c * np.cos(lat_c) * np.sin(lon - lon_0)
    s_z = r_c * np.sin(lat_c)

    # Visibility check: s_x must be positive (point faces the satellite)
    if s_x < 0.0:
        return None, None

    s_norm = np.sqrt(s_x**2 + s_y**2 + s_z**2)
    y_angle = np.arctan(s_z / s_x)         # elevation angle (N-S)
    x_angle = np.arcsin(-s_y / s_norm)     # azimuth angle (E-W)
    return x_angle, y_angle


def _box_scanangle_bounds(
    lat: float,
    lon: float,
    deg: float,
    lon_0: float,
    H: float,
    r_eq: float,
    r_pol: float,
) -> tuple[float, float, float, float] | None:
    """Compute (x_min, x_max, y_min, y_max) scan angle bounds for a lat/lon box.

    Samples all four corners plus the centre to handle the slight curvature
    introduced by the geostationary projection.  Returns None if the storm
    centre is not visible to the satellite.

    Args:
        lat:  Storm centre latitude (degrees).
        lon:  Storm centre longitude (degrees).
        deg:  Half-width of the box (degrees).
        lon_0: Satellite longitude (degrees, positive east).
        H:    Satellite distance from Earth centre (metres).
        r_eq: Semi-major axis (metres).
        r_pol: Semi-minor axis (metres).

    Returns:
        (x_min, x_max, y_min, y_max) in radians, or None.
    """
    sample_points = [
        (lat - deg, lon - deg),
        (lat - deg, lon + deg),
        (lat + deg, lon - deg),
        (lat + deg, lon + deg),
        (lat, lon),           # centre
    ]
    xs: list[float] = []
    ys: list[float] = []
    center_visible = False

    for i, (la, lo) in enumerate(sample_points):
        xv, yv = latlon_to_scanangle(la, lo, lon_0, H, r_eq, r_pol)
        if xv is not None:
            xs.append(xv)
            ys.append(yv)
            if i == 4:  # centre point
                center_visible = True

    if not center_visible or not xs:
        return None

    return min(xs), max(xs), min(ys), max(ys)


# ---------------------------------------------------------------------------
# GOES-16 filename helpers
# ---------------------------------------------------------------------------

# Matches OR_ABI-L2-CMIPF-M{mode}C{band}_G16_s{start}_e{end}_c{create}.nc
_GOES_FN_RE = re.compile(
    r"OR_ABI-L2-CMIPF-M\dC(?P<band>\d{2})_G16"
    r"_s(?P<year>\d{4})(?P<doy>\d{3})"
    r"(?P<hh>\d{2})(?P<mm>\d{2})(?P<ss>\d{2})\d"
    r"_e\d+_c\d+\.nc$"
)


def _parse_granule_time(basename: str) -> datetime | None:
    """Parse the scan start time encoded in a GOES-16 ABI filename.

    The start-time field uses the format: YYYYJJJHHMMSSs where JJJ is the
    day-of-year and the final character is a tenths-of-seconds digit.

    Args:
        basename: Filename component only (not the full S3 path).

    Returns:
        UTC datetime, or None if the pattern does not match.
    """
    m = _GOES_FN_RE.search(basename)
    if not m:
        return None
    year = int(m.group("year"))
    doy = int(m.group("doy"))
    hh = int(m.group("hh"))
    mm_val = int(m.group("mm"))
    ss = int(m.group("ss"))
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=doy - 1, hours=hh, minutes=mm_val, seconds=ss
    )


def _s3_hour_prefix(dt: datetime) -> str:
    """Build the S3 key prefix (without bucket) for the hour containing *dt*.

    Args:
        dt: UTC datetime.

    Returns:
        Prefix string, e.g. ``ABI-L2-CMIPF/2020/001/00/``.
    """
    doy = dt.timetuple().tm_yday
    return f"{S3_PRODUCT}/{dt.year}/{doy:03d}/{dt.hour:02d}/"


# ---------------------------------------------------------------------------
# S3 access & local caching
# ---------------------------------------------------------------------------


def _open_s3() -> s3fs.S3FileSystem:
    """Return an anonymous S3FileSystem pointed at the public NOAA bucket."""
    return s3fs.S3FileSystem(anon=True)


def find_nearest_granule(
    fs: s3fs.S3FileSystem,
    dt: datetime,
    window_minutes: int = TIME_WINDOW_MIN,
) -> tuple[str | None, datetime | None]:
    """Find the S3 key of the nearest Band 13 full-disk granule.

    Lists the hour bucket for *dt* and, when *dt* falls within
    *window_minutes* of an hour boundary, also the adjacent hour bucket.
    Selects the scan whose start time is closest to *dt* and within the
    allowed window.

    Args:
        fs:             Anonymous s3fs filesystem.
        dt:             Target UTC datetime (timezone-aware or naive UTC).
        window_minutes: Maximum allowed offset from *dt* in minutes.

    Returns:
        (s3_key, scan_time) of the best match, or (None, None) if none found.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    hour_start = dt.replace(minute=0, second=0, microsecond=0)
    minutes_past_hour = (dt - hour_start).total_seconds() / 60.0

    hours_to_check: list[datetime] = [hour_start]
    if minutes_past_hour < window_minutes:
        hours_to_check.append(hour_start - timedelta(hours=1))
    if minutes_past_hour > 60 - window_minutes:
        hours_to_check.append(hour_start + timedelta(hours=1))

    candidates: list[tuple[str, datetime]] = []
    for hour_dt in hours_to_check:
        prefix = f"{S3_BUCKET}/{_s3_hour_prefix(hour_dt)}"
        try:
            entries = fs.ls(prefix, detail=False)
        except FileNotFoundError:
            continue
        for s3_path in entries:
            basename = s3_path.rsplit("/", 1)[-1]
            if f"C{GOES16_BAND:02d}" not in basename:
                continue
            scan_time = _parse_granule_time(basename)
            if scan_time is None:
                continue
            delta_min = abs((scan_time - dt).total_seconds()) / 60.0
            if delta_min <= window_minutes:
                candidates.append((s3_path, scan_time))

    if not candidates:
        return None, None

    best_key, best_time = min(
        candidates, key=lambda kv: abs((kv[1] - dt).total_seconds())
    )
    return best_key, best_time


def cache_granule(
    fs: s3fs.S3FileSystem,
    s3_key: str,
    cache_dir: Path,
) -> tuple[Path, bool]:
    """Download a GOES-16 granule to the local cache, if not already present.

    Mirrors the S3 path structure under *cache_dir*.  Writes atomically via
    a temporary file so an interrupted download never leaves a corrupt entry.

    Args:
        fs:        Anonymous s3fs filesystem.
        s3_key:    Full S3 key including bucket, e.g. ``noaa-goes16/ABI-…``.
        cache_dir: Root cache directory.

    Returns:
        (local_path, was_downloaded) — was_downloaded is True only when a
        fresh network download occurred; False means the file was already cached.
    """
    relative = s3_key.removeprefix(f"{S3_BUCKET}/")
    local_path = cache_dir / relative
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        return local_path, False  # cache hit — no download needed

    logger.debug(f"Downloading {s3_key.split('/')[-1]} …")
    tmp_path: Path | None = None
    try:
        fd, tmp_str = tempfile.mkstemp(dir=local_path.parent, suffix=".tmp")
        tmp_path = Path(tmp_str)
        with open(fd, "wb") as fout, fs.open(s3_key, "rb") as fin:
            while True:
                chunk = fin.read(65_536)
                if not chunk:
                    break
                fout.write(chunk)
        tmp_path.rename(local_path)
    except Exception:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    return local_path, True  # freshly downloaded


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_bt_features(
    nc_path: Path,
    storm_lat: float,
    storm_lon: float,
    box_deg: float = BOX_DEG,
) -> dict[str, float]:
    """Compute brightness-temperature features from a cached GOES-16 granule.

    Opens the NetCDF file with lazy I/O (only the needed spatial slice is
    pulled off disk), extracts the ±*box_deg* window around the storm centre,
    and computes five scalar features plus a coverage fraction.

    Args:
        nc_path:   Local path to the cached ABI-L2-CMIPF .nc file.
        storm_lat: Storm centre latitude (degrees).
        storm_lon: Storm centre longitude (degrees, positive east).
        box_deg:   Half-width of the spatial extraction box (degrees).

    Returns:
        Dict with keys matching GOES_FEATURE_COLUMNS + ``goes_coverage``.
        All values are plain Python floats; NaN on any failure.
    """
    nan_result: dict[str, float] = {
        "std_bt": float("nan"),
        "area_deep_conv": float("nan"),
        "min_bt": float("nan"),
        "sym_index": float("nan"),
        "ot_count": float("nan"),
        "goes_coverage": float("nan"),
    }

    try:
        ds = xr.open_dataset(nc_path, engine="netcdf4")
    except Exception as exc:
        logger.warning(f"Cannot open {nc_path.name}: {exc}")
        return nan_result

    try:
        proj = ds["goes_imager_projection"]
        lon_0 = float(proj.longitude_of_projection_origin)
        # H = satellite height above surface + semi-major axis = dist from Earth centre
        H = float(proj.perspective_point_height) + float(proj.semi_major_axis)
        r_eq = float(proj.semi_major_axis)
        r_pol = float(proj.semi_minor_axis)

        # Load coordinate arrays (small 1-D, fast)
        x_arr = ds["x"].values.astype(np.float64)  # E-W scan angle, ascending
        y_arr = ds["y"].values.astype(np.float64)  # N-S scan angle, descending
    except (KeyError, AttributeError) as exc:
        logger.warning(f"Projection/coordinate variable missing in {nc_path.name}: {exc}")
        ds.close()
        return nan_result

    # Compute scan-angle bounding box for the storm's spatial window
    bounds = _box_scanangle_bounds(
        storm_lat, storm_lon, box_deg, lon_0, H, r_eq, r_pol
    )
    if bounds is None:
        ds.close()
        return nan_result  # storm centre not visible to this satellite

    x_min_sa, x_max_sa, y_min_sa, y_max_sa = bounds

    # x_arr is stored west→east (ascending)
    ix0 = int(np.searchsorted(x_arr, x_min_sa, side="left"))
    ix1 = int(np.searchsorted(x_arr, x_max_sa, side="right"))

    # y_arr is stored north→south (descending) in GOES-16 files
    # Larger y (more north) → smaller index; so invert to use searchsorted
    if y_arr[0] > y_arr[-1]:  # descending — expected case
        iy0 = int(np.searchsorted(-y_arr, -y_max_sa, side="left"))
        iy1 = int(np.searchsorted(-y_arr, -y_min_sa, side="right"))
    else:  # ascending — defensive fallback
        iy0 = int(np.searchsorted(y_arr, y_min_sa, side="left"))
        iy1 = int(np.searchsorted(y_arr, y_max_sa, side="right"))

    ny, nx = ds["CMI"].shape
    ix0 = max(0, min(ix0, nx))
    ix1 = max(0, min(ix1, nx))
    iy0 = max(0, min(iy0, ny))
    iy1 = max(0, min(iy1, ny))

    if ix1 <= ix0 or iy1 <= iy0:
        ds.close()
        return nan_result

    try:
        # Lazy-load only the spatial slice — avoids reading the full ~468 MB image
        bt = (
            ds["CMI"]
            .isel(y=slice(iy0, iy1), x=slice(ix0, ix1))
            .values.astype(np.float32)
        )
    except Exception as exc:
        logger.warning(f"CMI slice failed in {nc_path.name}: {exc}")
        ds.close()
        return nan_result

    ds.close()

    # Coverage fraction — fraction of box pixels with valid (non-NaN) BT
    total_pixels = bt.size
    valid_mask = np.isfinite(bt)
    n_valid = int(valid_mask.sum())
    goes_coverage = n_valid / total_pixels if total_pixels > 0 else 0.0

    if n_valid == 0:
        return {**nan_result, "goes_coverage": 0.0}

    bt_valid = bt[valid_mask]

    # Feature 1: std_bt — spatial standard deviation of BT across the box
    std_bt = float(np.std(bt_valid))

    # Feature 2: area_deep_conv — fraction of pixels colder than 220 K
    area_deep_conv = float(np.mean(bt_valid < DEEP_CONV_THRESHOLD_K))

    # Feature 3: min_bt — coldest pixel (proxy for tallest cloud top)
    min_bt = float(np.min(bt_valid))

    # Feature 4: sym_index — std of per-quadrant mean BT (NW/NE/SW/SE)
    #   The pixel grid is split at its spatial midpoint, which closely
    #   approximates the geographic quadrant split at the storm's lat/lon.
    rows, cols = bt.shape
    row_mid = rows // 2
    col_mid = cols // 2
    quadrant_slices = [
        (slice(0, row_mid), slice(0, col_mid)),         # NW (top-left)
        (slice(0, row_mid), slice(col_mid, cols)),      # NE (top-right)
        (slice(row_mid, rows), slice(0, col_mid)),      # SW (bottom-left)
        (slice(row_mid, rows), slice(col_mid, cols)),   # SE (bottom-right)
    ]
    quadrant_means: list[float] = []
    for r_sl, c_sl in quadrant_slices:
        quad = bt[r_sl, c_sl]
        q_valid = quad[np.isfinite(quad)]
        if q_valid.size >= 4:  # require at least a few pixels per quadrant
            quadrant_means.append(float(np.mean(q_valid)))

    sym_index = (
        float(np.std(quadrant_means)) if len(quadrant_means) >= 2 else float("nan")
    )

    # Feature 5: ot_count — fraction of pixels colder than 200 K (overshooting tops)
    ot_count = float(np.mean(bt_valid < OT_THRESHOLD_K))

    return {
        "std_bt": std_bt,
        "area_deep_conv": area_deep_conv,
        "min_bt": min_bt,
        "sym_index": sym_index,
        "ot_count": ot_count,
        "goes_coverage": goes_coverage,
    }


# ---------------------------------------------------------------------------
# Worker (module-level so it is picklable by the 'spawn' multiprocessing context)
# ---------------------------------------------------------------------------


class _WorkItem(NamedTuple):
    """Carries all data needed by a worker process for one storm observation."""

    storm_id: str
    datetime_utc: pd.Timestamp
    lat: float
    lon: float
    cache_dir: Path


def _process_one(item: _WorkItem) -> dict:
    """Fetch, cache, and extract GOES-16 features for one storm observation.

    Creates its own s3fs connection (connections are not picklable).
    Any retrieval or extraction error is caught and returns NaN features so
    the parent process can always collect a result row.

    Args:
        item: Work item with storm metadata.

    Returns:
        Dict with storm_id, datetime, GOES_FEATURE_COLUMNS, goes_coverage,
        plus private tracking keys _status, _was_downloaded, _file_bytes
        (stripped by the parent before storing rows).
    """
    nan_features = {col: float("nan") for col in GOES_FEATURE_COLUMNS}
    base = {
        "storm_id": item.storm_id,
        "datetime": item.datetime_utc,
        **nan_features,
        "goes_coverage": float("nan"),
        # private tracking fields — stripped in build_goes16_features
        "_status": "fail",
        "_was_downloaded": False,
        "_file_bytes": 0,
    }

    dt: datetime = item.datetime_utc.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    try:
        fs = _open_s3()
        s3_key, _scan_time = find_nearest_granule(fs, dt)
    except Exception as exc:
        logger.warning(f"[{item.storm_id}/{dt.date()}] S3 listing error: {exc}")
        return base

    if s3_key is None:
        base["_status"] = "no_granule"
        return base  # no scan within ±30 min

    try:
        local_path, was_downloaded = cache_granule(fs, s3_key, item.cache_dir)
        base["_was_downloaded"] = was_downloaded
        base["_file_bytes"] = local_path.stat().st_size
    except Exception as exc:
        logger.warning(
            f"[{item.storm_id}/{dt.date()}] Download failed "
            f"({s3_key.rsplit('/', 1)[-1]}): {exc}"
        )
        return base

    features = extract_bt_features(local_path, item.lat, item.lon)
    if not np.isnan(features.get("goes_coverage", float("nan"))):
        base["_status"] = "success"
    else:
        base["_status"] = "extract_fail"
    return {**base, **features}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_goes16_features(
    training_path: Path = TRAINING_DATA_PATH,
    cache_dir: Path = RAW_GOES16_DIR,
    output_path: Path = OUTPUT_PATH,
    n_workers: int = N_WORKERS,
) -> pd.DataFrame:
    """Fetch GOES-16 BT features for all observations in the training dataset.

    Pre-GOES-16-era observations (before 2018-01-01) receive all-NaN features
    without any S3 access.  GOES-16-era observations are dispatched to a
    multiprocessing pool; already-cached granules are reused.  Safe to re-run
    after interruptions.

    Args:
        training_path: Path to training_data.parquet.
        cache_dir:     Root directory for granule cache.
        output_path:   Destination Parquet path.
        n_workers:     Number of parallel worker processes.

    Returns:
        DataFrame with columns: storm_id, datetime, GOES_FEATURE_COLUMNS,
        goes_coverage (one row per training observation).
    """
    if not training_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {training_path}\n"
            "Run data/scripts/build_training_data.py first."
        )

    logger.info(f"Loading {training_path.name} …")
    df_train = pd.read_parquet(
        training_path, columns=["storm_id", "datetime", "lat", "lon"]
    )
    df_train["datetime"] = pd.to_datetime(df_train["datetime"])
    # Ensure tz-aware for GOES-16 era comparison
    if df_train["datetime"].dt.tz is None:
        df_train["datetime"] = df_train["datetime"].dt.tz_localize("UTC")

    n_total = len(df_train)
    logger.info(
        f"  → {n_total:,} observations | "
        f"{df_train['storm_id'].nunique():,} storms"
    )

    # Split on GOES-16 availability
    goes_mask = df_train["datetime"] >= pd.Timestamp(GOES16_OPERATIONAL_DATE)
    df_goes = df_train[goes_mask].reset_index(drop=True)
    df_pre = df_train[~goes_mask].reset_index(drop=True)

    logger.info(
        f"  GOES-16 era (≥ 2018): {len(df_goes):,} observations | "
        f"pre-GOES-16: {len(df_pre):,} (will be NaN)"
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build pre-GOES NaN rows (no S3 access needed)
    nan_feature_vals = {col: float("nan") for col in GOES_FEATURE_COLUMNS}
    pre_goes_records = [
        {
            "storm_id": str(row.storm_id),
            "datetime": row.datetime,
            **nan_feature_vals,
            "goes_coverage": float("nan"),
        }
        for row in df_pre.itertuples(index=False)
    ]

    # Build work items for GOES-16 era
    work_items = [
        _WorkItem(
            storm_id=str(row.storm_id),
            datetime_utc=row.datetime,
            lat=float(row.lat),
            lon=float(row.lon),
            cache_dir=cache_dir,
        )
        for row in df_goes.itertuples(index=False)
    ]

    logger.info(
        f"Dispatching {len(work_items):,} GOES-16-era observations "
        f"to {n_workers} worker(s) …"
    )

    goes_records: list[dict] = []

    # Progress tracking counters
    _n_success    = 0
    _n_skip       = 0   # cache hits (no download needed)
    _n_no_granule = 0   # no GOES scan within ±30 min window
    _n_fail       = 0   # download or extraction errors
    _bytes_dl     = 0   # bytes freshly downloaded
    _t0           = time.perf_counter()

    _PROGRESS_INTERVAL = 50  # log every N completed work items
    _total = len(work_items)

    def _log_progress(i: int) -> None:
        elapsed = time.perf_counter() - _t0
        rate    = i / elapsed if elapsed > 0 else 0.0
        eta_sec = (_total - i) / rate if rate > 0 else 0.0
        eta_str = (
            f"{eta_sec / 3600:.1f} h"
            if eta_sec >= 3600
            else f"{eta_sec / 60:.0f} min"
        )
        gb_done = _bytes_dl / 1e9
        logger.info(
            f"  [{i:,}/{_total:,}]  "
            f"done={i}  remaining={_total - i}  "
            f"downloaded={gb_done:.2f} GB  ETA={eta_str}  |  "
            f"success={_n_success}  skip={_n_skip}  "
            f"no_granule={_n_no_granule}  fail={_n_fail}"
        )

    if n_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for i, result in enumerate(
                pool.imap_unordered(_process_one, work_items, chunksize=4),
                start=1,
            ):
                status        = result.pop("_status", "fail")
                was_dl        = result.pop("_was_downloaded", False)
                file_bytes    = result.pop("_file_bytes", 0)

                if status == "success":
                    _n_success += 1
                elif status == "no_granule":
                    _n_no_granule += 1
                else:
                    _n_fail += 1

                if was_dl:
                    _bytes_dl += file_bytes
                else:
                    _n_skip += 1

                goes_records.append(result)
                if i % _PROGRESS_INTERVAL == 0 or i == _total:
                    _log_progress(i)
    else:
        for i, item in enumerate(work_items, start=1):
            result        = _process_one(item)
            status        = result.pop("_status", "fail")
            was_dl        = result.pop("_was_downloaded", False)
            file_bytes    = result.pop("_file_bytes", 0)

            if status == "success":
                _n_success += 1
            elif status == "no_granule":
                _n_no_granule += 1
            else:
                _n_fail += 1

            if was_dl:
                _bytes_dl += file_bytes
            else:
                _n_skip += 1

            goes_records.append(result)
            if i % _PROGRESS_INTERVAL == 0 or i == _total:
                _log_progress(i)

    # Combine and enforce dtypes
    df_out = pd.DataFrame(pre_goes_records + goes_records)
    df_out["storm_id"] = df_out["storm_id"].astype("string")
    df_out["datetime"] = pd.to_datetime(df_out["datetime"])
    for col in GOES_FEATURE_COLUMNS + ["goes_coverage"]:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce").astype(np.float32)

    df_out.sort_values(["storm_id", "datetime"], inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    df_out.to_parquet(output_path, index=False)
    logger.success(
        f"Saved → {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)"
    )
    return df_out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(df: pd.DataFrame) -> None:
    """Print extraction statistics for the complete output DataFrame.

    Reports attempted count (GOES-16 era only), successful retrievals,
    mean goes_coverage, and mean value per feature column.

    Args:
        df: Full output DataFrame including pre-GOES-16 NaN rows.
    """
    # "Attempted" = rows where a GOES-16 scan was within reach (coverage != NaN
    # or the date was in the GOES era regardless of outcome).
    # Proxy: rows where datetime >= GOES16_OPERATIONAL_DATE.
    dt_col = df["datetime"]
    if dt_col.dt.tz is None:
        dt_col = dt_col.dt.tz_localize("UTC")
    goes_era_mask = dt_col >= pd.Timestamp(GOES16_OPERATIONAL_DATE)
    n_attempted = int(goes_era_mask.sum())
    n_successful = int(df.loc[goes_era_mask, "goes_coverage"].notna().sum())
    mean_coverage = float(df["goes_coverage"].mean(skipna=True))

    print()
    print("=" * 62)
    print("  GOES-16 Feature Extraction — Summary")
    print("=" * 62)
    print(f"  Total observations in dataset  : {len(df):,}")
    print(f"  GOES-16-era observations       : {n_attempted:,}  (≥ 2018-01-01)")
    print(f"  Successful retrievals          : {n_successful:,}"
          f"  ({100 * n_successful / max(n_attempted, 1):.1f}%)")
    print(f"  Mean goes_coverage             : {mean_coverage:.3f}")
    print()
    print(f"  {'Feature':<18}  {'Mean (non-NaN rows)':>20}")
    print(f"  {'-'*18}  {'-'*20}")
    for col in GOES_FEATURE_COLUMNS:
        mean_val = float(df[col].mean(skipna=True))
        unit_hint = " K" if "bt" in col else ""
        print(f"  {col:<18}  {mean_val:>18.4f}{unit_hint}")
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # freeze_support() is required for Windows multiprocessing with 'spawn'
    mp.freeze_support()

    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<8} | {message}")

    df_result = build_goes16_features()
    print_summary(df_result)
