"""
train_lstm.py — Train a two-layer LSTM for Rapid Intensification prediction.

Loads training_data.parquet and goes16_features.parquet, joins them into a
combined feature set, builds rolling 48-hour (8 × 6-hourly timestep) sequences
per storm, and trains an LSTM binary classifier with BCEWithLogitsLoss.

Architecture (from CLAUDE.md spec)
------------------------------------
    LSTM(input_size=N, hidden_size=128, num_layers=2, dropout=0.3)
    → Linear(128, 64) → ReLU → Dropout(0.3) → Linear(64, 1)

Training uses BCEWithLogitsLoss with pos_weight, AdamW optimiser, and cosine
annealing LR.  Best model (lowest val loss) is saved at the end.

Temporal split
--------------
Identical to train_xgboost.py: storms first observed before 2018 → training
pool; 2018–2023 → test.  Last 15 % of training-pool rows by time → validation.

Output
------
    model/artifacts/lstm_model_{timestamp}.pt   — PyTorch state_dict archive
    model/artifacts/lstm_model_latest.pt        — always latest
    model/artifacts/lstm_test_preds.parquet     — storm_id, datetime,
                                                   ri_label, lstm_proba

Usage:
    python model/train_lstm.py
"""

from __future__ import annotations

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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Repo-root path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.scripts.build_training_data import (  # noqa: E402
    ALL_FEATURE_COLUMNS,
    GOES_FEATURE_COLUMNS,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TRAINING_DATA_PATH = PROCESSED_DIR / "training_data.parquet"
GOES16_FEATURES_PATH = PROCESSED_DIR / "goes16_features.parquet"

ARTIFACTS_DIR = REPO_ROOT / "model" / "artifacts"
TEST_PREDS_PATH = ARTIFACTS_DIR / "lstm_test_preds.parquet"

# ALL_FEATURE_COLUMNS already includes GOES-16 features (added by build_training_data.py).
# LSTM uses the full 16-feature set.
LSTM_FEATURE_COLUMNS: list[str] = ALL_FEATURE_COLUMNS

# Sequence length: 8 steps × 6-hourly = 48-hour rolling window
SEQ_LEN: int = 8

# Temporal split year (same as XGBoost)
TEST_YEAR: int = 2018
VAL_FRAC: float = 0.15

# Training hyperparameters
BATCH_SIZE: int = 512
MAX_EPOCHS: int = 60
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
EARLY_STOP_PATIENCE: int = 8

# LSTM architecture
HIDDEN_SIZE: int = 128
NUM_LAYERS: int = 2
LSTM_DROPOUT: float = 0.3

# Evaluation threshold
RI_PROB_THRESHOLD: float = 0.40


# ---------------------------------------------------------------------------
# Data loading & feature join
# ---------------------------------------------------------------------------


def load_and_join(
    training_path: Path = TRAINING_DATA_PATH,
    goes_path: Path = GOES16_FEATURES_PATH,
) -> pd.DataFrame:
    """Load training data and optionally join GOES-16 features.

    If goes16_features.parquet does not exist the LSTM falls back to the
    ALL_FEATURE_COLUMNS set (SHIPS + engineered) without satellite columns,
    logging a warning.

    Args:
        training_path: Path to training_data.parquet.
        goes_path:     Path to goes16_features.parquet.

    Returns:
        DataFrame with LSTM_FEATURE_COLUMNS (or ALL_FEATURE_COLUMNS fallback),
        storm_id, datetime, ri_label.
    """
    if not training_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {training_path}\n"
            "Run: python data/scripts/build_training_data.py"
        )

    logger.info(f"Loading {training_path.name} …")
    df = pd.read_parquet(training_path)
    logger.info(
        f"  → {len(df):,} rows | {df['storm_id'].nunique():,} storms | "
        f"RI rate {df['ri_label'].mean() * 100:.1f}%"
    )

    # GOES-16 features are already merged into training_data.parquet by
    # build_training_data.py.  Only fall back to a separate join if they are
    # somehow absent (e.g. old parquet built before the GOES update).
    goes_missing = [c for c in GOES_FEATURE_COLUMNS if c not in df.columns]
    if goes_missing and goes_path.exists():
        logger.warning(
            "GOES-16 columns missing from training_data — doing late join from "
            f"{goes_path.name} …"
        )
        goes_df = pd.read_parquet(goes_path)
        goes_df["datetime"] = pd.to_datetime(goes_df["datetime"])
        if goes_df["datetime"].dt.tz is not None:
            goes_df["datetime"] = goes_df["datetime"].dt.tz_localize(None)
        df = pd.merge(df, goes_df, on=["storm_id", "datetime"], how="left")
        # Impute with GOES-era medians
        goes_era_mask = df[GOES_FEATURE_COLUMNS[0]].notna()
        for col in GOES_FEATURE_COLUMNS:
            if df[col].isna().any():
                fill_val = float(df.loc[goes_era_mask, col].median())
                df[col] = df[col].fillna(fill_val)
    elif goes_missing:
        logger.warning(
            f"{goes_path.name} not found and GOES columns absent from "
            "training_data — GOES features will use imputed medians (0.0). "
            "Run fetch_goes16.py + build_training_data.py to add satellite features."
        )
        for col in goes_missing:
            df[col] = np.float32(0.0)
    else:
        n_nan = int(df[GOES_FEATURE_COLUMNS].isna().any(axis=1).sum())
        if n_nan:
            logger.warning(f"  {n_nan:,} rows still have NaN GOES features after load — check imputation.")
        logger.info(
            f"  GOES-16 features present in training_data "
            f"({len(GOES_FEATURE_COLUMNS)} columns)"
        )

    feature_cols = LSTM_FEATURE_COLUMNS
    logger.info(f"Feature columns: {len(feature_cols)}")
    return df, feature_cols


# ---------------------------------------------------------------------------
# Temporal split (mirrors train_xgboost.py)
# ---------------------------------------------------------------------------


def temporal_split(
    df: pd.DataFrame,
    test_year: int = TEST_YEAR,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal train/val/test split identical to train_xgboost.py.

    Args:
        df:        Full DataFrame.
        test_year: First year of test window.
        val_frac:  Fraction of training pool rows reserved for validation.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    storm_first_year = df.groupby("storm_id")["datetime"].min().dt.year
    train_pool_storms = storm_first_year[storm_first_year < test_year].index
    test_storms = storm_first_year[storm_first_year >= test_year].index

    train_pool = df[df["storm_id"].isin(train_pool_storms)].copy()
    test_df = df[df["storm_id"].isin(test_storms)].copy()

    sorted_pool = train_pool.sort_values("datetime")
    val_cutoff_idx = int(len(sorted_pool) * (1.0 - val_frac))
    val_cutoff_dt = sorted_pool.iloc[val_cutoff_idx]["datetime"]

    train_df = train_pool[train_pool["datetime"] < val_cutoff_dt].copy()
    val_df = train_pool[train_pool["datetime"] >= val_cutoff_dt].copy()

    logger.info("Temporal split:")
    logger.info(
        f"  Train : {len(train_df):>7,} rows | {train_df['storm_id'].nunique()} storms | "
        f"years {train_df['datetime'].dt.year.min()}–{train_df['datetime'].dt.year.max()} | "
        f"RI {train_df['ri_label'].mean() * 100:.1f}%"
    )
    logger.info(
        f"  Val   : {len(val_df):>7,} rows | {val_df['storm_id'].nunique()} storms | "
        f"RI {val_df['ri_label'].mean() * 100:.1f}%"
    )
    logger.info(
        f"  Test  : {len(test_df):>7,} rows | {test_df['storm_id'].nunique()} storms | "
        f"years {test_df['datetime'].dt.year.min()}–{test_df['datetime'].dt.year.max()} | "
        f"RI {test_df['ri_label'].mean() * 100:.1f}%"
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Sequence dataset
# ---------------------------------------------------------------------------


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build fixed-length look-back sequences for the LSTM.

    For each row where at least ``seq_len`` consecutive 6-hourly observations
    exist within the same storm, a (seq_len × n_features) sequence is created
    whose label is the RI label at the final (most recent) timestep.

    A valid sequence requires:
      - seq_len rows in the same storm ordered by datetime.
      - Each consecutive pair is exactly 6 hours apart (no track gaps).
      - The target row (last in the sequence) has a defined ri_label.

    Args:
        df:           DataFrame sorted by (storm_id, datetime).
        feature_cols: Ordered list of feature column names.
        seq_len:      Number of 6-hourly timesteps in the look-back window.

    Returns:
        Tuple of:
          - X: float32 array of shape (N, seq_len, n_features)
          - y: int8 array of shape (N,)
          - meta: DataFrame with storm_id, datetime, ri_label for each sequence
                  (indexed to the *last* row of each sequence).
    """
    df = df.sort_values(["storm_id", "datetime"]).reset_index(drop=True)
    feat_arr = df[feature_cols].to_numpy(dtype=np.float32)
    labels = df["ri_label"].to_numpy(dtype=np.int8)
    storm_ids = df["storm_id"].to_numpy()
    datetimes = df["datetime"].to_numpy()

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    meta_rows: list[dict] = []

    # Group by storm for efficient sequential scanning
    for storm_id, grp in df.groupby("storm_id", sort=False):
        idx = grp.index.to_numpy()
        if len(idx) < seq_len:
            continue

        times = df.loc[idx, "datetime"].to_numpy()
        time_gaps_h = np.diff(times.astype("datetime64[s]")).astype(np.float64) / 3600.0

        for end in range(seq_len - 1, len(idx)):
            start = end - (seq_len - 1)
            # All consecutive gaps in the window must be exactly 6h
            gaps = time_gaps_h[start:end]
            if not np.all(gaps == 6.0):
                continue

            target_idx = idx[end]
            lbl = labels[target_idx]
            if np.isnan(lbl):
                continue

            seq = feat_arr[idx[start:end + 1]]  # shape (seq_len, n_features)
            X_list.append(seq)
            y_list.append(int(lbl))
            meta_rows.append({
                "storm_id": storm_ids[target_idx],
                "datetime": datetimes[target_idx],
                "ri_label": int(lbl),
            })

    X = np.stack(X_list, axis=0)       # (N, seq_len, n_features)
    y = np.array(y_list, dtype=np.int8)
    meta_df = pd.DataFrame(meta_rows).reset_index(drop=True)

    logger.info(
        f"Sequences built: {len(X):,} | seq_len={seq_len} | "
        f"features={X.shape[2]} | RI rate {y.mean() * 100:.1f}%"
    )
    return X, y, meta_df


class RISequenceDataset(Dataset):
    """PyTorch Dataset wrapping (X, y) sequence arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)           # float32
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RILSTMModel(nn.Module):
    """Two-layer LSTM binary classifier for Rapid Intensification.

    Architecture (CLAUDE.md spec):
        LSTM(input → 128, 2 layers, dropout=0.3)
        → take hidden state of last timestep
        → Linear(128 → 64) → ReLU → Dropout(0.3) → Linear(64 → 1)

    Output is a raw logit; apply sigmoid externally for probabilities.

    Args:
        n_features:   Number of input features per timestep.
        hidden_size:  LSTM hidden dimension.
        num_layers:   Number of stacked LSTM layers.
        dropout:      Dropout rate applied between LSTM layers and in the head.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
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
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, n_features).

        Returns:
            Logit tensor of shape (batch, 1).
        """
        _, (h_n, _) = self.lstm(x)
        # h_n shape: (num_layers, batch, hidden_size) — take last layer
        last_hidden = h_n[-1]   # (batch, hidden_size)
        return self.head(last_hidden)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def compute_pos_weight(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    """Compute BCEWithLogitsLoss pos_weight from training label distribution.

    Args:
        y_train: Binary label array.
        device:  Target torch device.

    Returns:
        Scalar tensor = n_neg / n_pos.
    """
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pw = n_neg / n_pos
    logger.info(
        f"Class balance — neg: {n_neg:,} | pos: {n_pos:,} | pos_weight: {pw:.2f}"
    )
    return torch.tensor([pw], dtype=torch.float32, device=device)


def train_epoch(
    model: RILSTMModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch; return mean loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model: RILSTMModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate loss and AUC on a DataLoader; return (loss, auc)."""
    model.eval()
    total_loss = 0.0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(X_batch)
        all_logits.append(logits.cpu().numpy().squeeze(-1))
        all_labels.append(y_batch.cpu().numpy().squeeze(-1))

    mean_loss = total_loss / len(loader.dataset)
    probs = torch.sigmoid(torch.from_numpy(np.concatenate(all_logits))).numpy()
    labels = np.concatenate(all_labels)
    auc = roc_auc_score(labels, probs) if labels.mean() > 0 else 0.0
    return mean_loss, float(auc)


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
    device: torch.device,
) -> RILSTMModel:
    """Train the LSTM with early stopping on validation loss.

    Args:
        X_train:    Training sequences (N, seq_len, n_features).
        y_train:    Training labels (N,).
        X_val:      Validation sequences.
        y_val:      Validation labels.
        n_features: Input feature count.
        device:     Torch device.

    Returns:
        Best model (lowest val loss across epochs).
    """
    train_ds = RISequenceDataset(X_train, y_train)
    val_ds = RISequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)

    model = RILSTMModel(n_features=n_features).to(device)
    pw = compute_pos_weight(y_train, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS
    )

    best_val_loss = float("inf")
    best_state: dict | None = None
    patience_counter = 0

    logger.info(
        f"Training LSTM — {n_features} features | {len(X_train):,} train seqs | "
        f"{MAX_EPOCHS} max epochs | device={device}"
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            tag = "*"
        else:
            patience_counter += 1
            tag = ""

        if epoch % 5 == 0 or epoch == 1 or tag == "*":
            logger.info(
                f"  Epoch {epoch:3d}/{MAX_EPOCHS} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_AUC={val_auc:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} {tag}"
            )

        if patience_counter >= EARLY_STOP_PATIENCE:
            logger.info(
                f"Early stopping triggered at epoch {epoch} "
                f"(no improvement for {EARLY_STOP_PATIENCE} epochs)"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.success(f"Loaded best model state (val_loss={best_val_loss:.4f})")

    return model


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------


def save_artifacts(
    model: RILSTMModel,
    n_features: int,
    feature_cols: list[str],
    test_meta: pd.DataFrame,
    y_prob_test: np.ndarray,
) -> Path:
    """Persist the LSTM model and test-set predictions.

    Saves:
      - model/artifacts/lstm_model_{timestamp}.pt   — state_dict + metadata
      - model/artifacts/lstm_model_latest.pt        — always latest
      - model/artifacts/lstm_test_preds.parquet     — storm_id, datetime,
                                                       ri_label, lstm_proba

    Args:
        model:        Trained RILSTMModel.
        n_features:   Number of input features.
        feature_cols: Feature column names (for inference reconstruction).
        test_meta:    DataFrame with storm_id, datetime, ri_label for test seqs.
        y_prob_test:  Predicted RI probabilities for test sequences.

    Returns:
        Path to the timestamped model file.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    checkpoint = {
        "state_dict": model.state_dict(),
        "n_features": n_features,
        "feature_cols": feature_cols,
        "seq_len": SEQ_LEN,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": LSTM_DROPOUT,
    }

    timestamped_path = ARTIFACTS_DIR / f"lstm_model_{timestamp}.pt"
    torch.save(checkpoint, timestamped_path)
    shutil.copy2(timestamped_path, ARTIFACTS_DIR / "lstm_model_latest.pt")
    logger.success(
        f"Model saved → {timestamped_path.name}  "
        f"(also copied to lstm_model_latest.pt)"
    )

    preds_df = pd.DataFrame(
        {
            "storm_id": test_meta["storm_id"].values,
            "datetime": test_meta["datetime"].values,
            "ri_label": test_meta["ri_label"].values,
            "lstm_proba": y_prob_test.astype(np.float32),
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


def train_lstm(
    training_path: Path = TRAINING_DATA_PATH,
    goes_path: Path = GOES16_FEATURES_PATH,
) -> RILSTMModel:
    """Run the full LSTM training pipeline end-to-end.

    Steps:
      1. Load and join training + GOES-16 feature data.
      2. Temporal train/val/test split.
      3. Build 8-timestep rolling sequences.
      4. Train LSTM with early stopping.
      5. Evaluate on test set.
      6. Save model + test predictions.

    Args:
        training_path: Path to training_data.parquet.
        goes_path:     Path to goes16_features.parquet.

    Returns:
        Trained RILSTMModel.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    df, feature_cols = load_and_join(training_path, goes_path)
    n_features = len(feature_cols)

    train_df, val_df, test_df = temporal_split(df)

    logger.info("Building training sequences …")
    X_train, y_train, _ = build_sequences(train_df, feature_cols)

    logger.info("Building validation sequences …")
    X_val, y_val, _ = build_sequences(val_df, feature_cols)

    logger.info("Building test sequences …")
    X_test, y_test, test_meta = build_sequences(test_df, feature_cols)

    model = train_lstm_model(X_train, y_train, X_val, y_val, n_features, device)

    # Evaluate on test set
    test_ds = RISequenceDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    pw = compute_pos_weight(y_train, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    test_loss, test_auc = eval_epoch(model, test_loader, criterion, device)

    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_test).to(device)
        logits = model(X_t).cpu().numpy().squeeze(-1)
    y_prob_test = torch.sigmoid(torch.from_numpy(logits)).numpy()

    logger.success(
        f"Test set — loss: {test_loss:.4f} | AUC: {test_auc:.4f} | "
        f"RI rate: {y_test.mean() * 100:.1f}%  "
        f"(SHIPS-RII benchmark: AUC ~0.78)"
    )

    save_artifacts(model, n_features, feature_cols, test_meta, y_prob_test)
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<8} | {message}")

    train_lstm()
