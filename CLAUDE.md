# Hurricane Rapid Intensification (RI) Alert System
## CLAUDE.md — Project Context & Vibe Coding Guide

---

## What this project is

A real-time machine learning system that predicts the probability of **Rapid Intensification (RI)** in active Atlantic and Eastern Pacific hurricanes — hours before the National Hurricane Center (NHC) issues an advisory. RI is defined as a 35+ mph increase in max sustained winds within 24 hours and is the most dangerous, least-solved problem in operational tropical meteorology.

This project directly replicates and improves upon the NHC's operational **SHIPS-RII** model (a 1990s-era logistic regression) by replacing it with a modern ML ensemble trained on the same public NOAA/NHC data sources, augmented with GOES-16 satellite imagery features that the operational model does not use.

**Live output:** A deployed React dashboard showing all active storms, color-coded by RI probability, with trend charts and NHC comparison.

---

## Project owner background

- Computer Engineering & Computer Science student, transferring into Cornell Atmospheric Sciences
- Former computer hardware engineer at NASA
- Incoming Environmental Computing & AI Researcher at Oakridge National Lab
- Strong Python, systems, and ML background — atmospheric domain knowledge is being built alongside the project

---

## Project structure

```
hurricane-ri-alert/
├── CLAUDE.md                  ← you are here
├── README.md
├── requirements.txt
├── .env.example
│
├── data/
│   ├── raw/                   # downloaded source files, never committed
│   │   ├── hurdat2/
│   │   ├── ships/
│   │   └── goes16/
│   ├── processed/             # cleaned, labeled, feature-engineered
│   └── scripts/
│       ├── fetch_hurdat2.py
│       ├── fetch_ships.py
│       ├── fetch_goes16.py
│       └── label_ri_events.py
│
├── model/
│   ├── train_xgboost.py
│   ├── train_lstm.py
│   ├── ensemble.py
│   ├── calibrate.py
│   ├── evaluate.py
│   └── artifacts/             # saved model files (.pkl, .pt), not committed
│
├── pipeline/
│   ├── ingestor.py            # real-time NHC + GOES polling
│   ├── feature_builder.py     # assembles live feature vector per storm
│   └── inference.py           # runs ensemble on live feature vector
│
├── api/
│   ├── main.py                # FastAPI app
│   ├── routes/
│   │   ├── storms.py          # GET /storms — all active storms + RI scores
│   │   └── history.py         # GET /storms/{id}/history
│   └── scheduler.py           # APScheduler polling every 30 min
│
└── frontend/
    ├── package.json
    ├── src/
    │   ├── App.jsx
    │   ├── components/
    │   │   ├── StormMap.jsx    # Leaflet map with RI-scored storm markers
    │   │   ├── StormCard.jsx   # sidebar card per active storm
    │   │   └── RITrendChart.jsx# recharts probability timeline
    │   └── lib/
    │       └── api.js          # fetch wrappers for backend
    └── public/
```

---

## Data sources

### 1. HURDAT2 (training backbone)
- **What:** North Atlantic and East Pacific hurricane database, 6-hourly observations back to 1851
- **Fields used:** date/time, lat, lon, max sustained winds (kt), min central pressure (mb), storm status
- **Download:** https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt
- **Format:** custom fixed-width text — see `fetch_hurdat2.py` for parser
- **Training window:** 1980–2023 (consistent aircraft recon era)
- **RI label:** computed as `wind_t+4 - wind_t >= 35 mph` using pandas shift on 6-hourly rows (4 steps = 24 hours)
- **Class imbalance:** ~15% RI cases — address with `scale_pos_weight` in XGBoost or SMOTE

### 2. SHIPS developmental dataset (tabular features)
- **What:** ~50 pre-computed atmospheric environment predictors matched to each HURDAT2 entry
- **Key features:**
  - `SHRD` — 850–200 hPa wind shear magnitude (kt) — most important single predictor
  - `RSST` — Reynolds sea surface temperature (°C)
  - `RHLO` — 850–700 hPa relative humidity (%)
  - `RHMD` — 500–300 hPa relative humidity (%)
  - `PSLV` — 200 hPa divergence
  - `OHCL` — Ocean heat content (kJ/cm²)
  - `VMPI` — Maximum potential intensity (kt)
  - `VVAV` — 850–200 hPa average vertical velocity
- **Download:** https://rammb.cira.colostate.edu/research/tropical_cyclones/ships/developmental_data.asp
- **Format:** space-delimited .txt files, one per basin per year — see `fetch_ships.py`
- **Join key:** storm ID + time step → matches HURDAT2 rows

### 3. GOES-16 (GOES-East) satellite imagery
- **What:** NOAA's geostationary satellite, covers Atlantic + Gulf in real time
- **Band used:** Band 13 — clean longwave IR (10.3 µm), brightness temperature of cloud tops
  - Cold pixels (< 220K) = deep convective towers = RI precursor
- **Access:** AWS S3 open data bucket — `s3://noaa-goes16/ABI-L2-CMIPF/` (no auth required)
- **Python access:** `boto3` with unsigned config or `s3fs`
- **Format:** NetCDF4 (.nc) files — use `xarray` + `netCDF4`
- **Tile extraction:** pull ±3° lat/lon box centered on storm eye, remap to brightness temp grid
- **Derived features (extract, don't use raw pixels as input):**
  - `std_bt` — standard deviation of brightness temp in storm box (cloud top turbulence)
  - `area_deep_conv` — pixel count < 220K (deep convection extent)
  - `min_bt` — coldest pixel (max convective tower height proxy)
  - `sym_index` — brightness temp asymmetry (lopsided convection = bad sign)
  - `ot_count` — overshooting top count (pixels < 200K)

### 4. NHC advisory RSS feed (real-time)
- **What:** Machine-readable XML advisory for every active storm, issued every 6 hours
- **URL pattern:** `https://www.nhc.noaa.gov/xml/TCPAT{1-5}.xml` (AL storms 1–5) and equivalent for EP
- **Alternative:** `https://www.nhc.noaa.gov/nhc_at1.xml` (current advisories index)
- **Fields needed:** storm ID, name, lat, lon, max winds (kt), movement, intensity forecast
- **Parse with:** `feedparser` or `xmltodict`

---

## ML architecture

### Model 1: XGBoost classifier (tabular)
- **Input:** SHIPS features + satellite-derived features (all scalar, current timestep)
- **Output:** RI probability [0–1]
- **Config:**
  ```python
  xgb.XGBClassifier(
      n_estimators=400,
      max_depth=6,
      learning_rate=0.05,
      subsample=0.8,
      colsample_bytree=0.8,
      scale_pos_weight=5.5,   # approx neg/pos ratio
      eval_metric='auc',
      use_label_encoder=False
  )
  ```

### Model 2: LSTM (temporal sequence)
- **Input:** rolling 48-hour window of SHIPS + satellite features (8 timesteps × N features)
- **Output:** RI probability [0–1]
- **Architecture:**
  ```python
  nn.LSTM(input_size=N_FEATURES, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
  nn.Linear(128, 64) → ReLU → Dropout(0.3) → Linear(64, 1) → Sigmoid
  ```
- **Framework:** PyTorch
- **Training:** BCELoss with pos_weight, AdamW optimizer, cosine annealing LR

### Ensemble
- Simple average: `p_final = 0.5 * p_xgb + 0.5 * p_lstm`
- Explore learned stacking (logistic meta-learner) after both models are working

### Calibration
- Apply **isotonic regression** to final ensemble output
- Evaluate with **Brier Skill Score** (BSS) — this is what NHC uses operationally, cite it explicitly
- Target: BSS > 0.15 over climatology is considered meaningful in the literature

### Evaluation metrics
- AUC-ROC (primary)
- Brier Skill Score vs. climatology
- Probability of Detection (POD) at 40% threshold
- False Alarm Rate (FAR) at 40% threshold
- Compare against SHIPS-RII baseline: POD ~0.42, FAR ~0.72 (published in NHC verification reports)

---

## Backend (FastAPI)

- Python 3.11+
- FastAPI + Uvicorn
- APScheduler for polling (every 30 min during active season)
- Store latest storm states and RI history in SQLite (simple) or Redis (if we want fast)
- Key endpoints:
  - `GET /storms` → list of active storms with current RI probability, trend, NHC forecast
  - `GET /storms/{storm_id}/history` → last N probability readings for trend chart
  - `GET /health` → ping

---

## Frontend (React)

- Vite + React
- **Map:** `react-leaflet` with OpenStreetMap tiles
- **Charts:** `recharts` for RI probability timeline
- **Storm markers:** color-coded circle markers
  - Green border: RI probability < 20%
  - Yellow border: 20–40%
  - Orange border: 40–60%
  - Red border + pulse animation: > 60%
- **Sidebar:** StormCard per active storm showing name, current intensity, RI score, 24h NHC forecast vs model forecast
- **Auto-refresh:** poll `/storms` every 5 minutes in the UI
- **Deploy:** Vercel (free tier, `vercel deploy`)

---

## Environment variables

```bash
# .env (never commit)
AWS_DEFAULT_REGION=us-east-1
NOAA_GOES16_BUCKET=noaa-goes16
NHC_FEED_BASE=https://www.nhc.noaa.gov
MODEL_ARTIFACTS_PATH=./model/artifacts
DATABASE_URL=sqlite:///./hurricane_ri.db
```

---

## Key Python dependencies

```
xarray>=2024.1
netCDF4>=1.6
boto3>=1.34
s3fs>=2024.1
xgboost>=2.0
torch>=2.2
scikit-learn>=1.4
imbalanced-learn>=0.12
pandas>=2.2
numpy>=1.26
fastapi>=0.110
uvicorn>=0.29
apscheduler>=3.10
feedparser>=6.0
xmltodict>=0.13
```

---

## Coding conventions

- Python: type hints everywhere, docstrings on all public functions
- Use `pathlib.Path` not `os.path`
- Logging via `loguru`, not `print`
- All data scripts should be idempotent (safe to re-run)
- Feature names must match exactly between training and inference — define them as a constant list in `feature_builder.py` and import everywhere
- Never hardcode thresholds — put them in a `config.py` dataclass
- Model artifacts: save with timestamp in filename, keep latest symlinked as `model_latest.pkl`

---

## Phase-by-phase build order

1. **Data pipeline** — `fetch_hurdat2.py` → `fetch_ships.py` → `label_ri_events.py` → merged training DataFrame
2. **Tabular baseline** — `train_xgboost.py` on SHIPS features only → get first AUC number
3. **Satellite features** — `fetch_goes16.py` → tile extractor → append satellite columns to training data
4. **Full ensemble** — `train_lstm.py` → `ensemble.py` → `calibrate.py` → `evaluate.py`
5. **Real-time pipeline** — `ingestor.py` → `feature_builder.py` → `inference.py`
6. **API** — FastAPI app wiring inference to HTTP endpoints
7. **Frontend** — map + storm cards + trend charts
8. **Deploy** — backend on Railway/Render (free tier), frontend on Vercel

---

## Scientific context (useful for comments and README)

- RI threshold (NHC definition): +35 kt in 24 hours (≈ +40 mph)
- Climatological RI base rate in Atlantic: ~10–15% of all 6-hourly observations
- Current operational benchmark: SHIPS-RII (logistic regression, ~20 predictors, circa 1999)
  - Published POD: 0.42 | FAR: 0.72 | BSS: ~0.08
- This project targets: POD > 0.50 | FAR < 0.65 | BSS > 0.15
- Key papers to cite if writing this up:
  - Kaplan & DeMaria (2003) — original SHIPS-RII paper
  - Zheng et al. (2023) — deep learning for RI prediction
  - NHC Tropical Cyclone Forecast Verification reports (annual, nhc.noaa.gov)
- NHC 5-year strategic plan (2020–2025) lists RI as top scientific priority

---

## Version control & workflow

- **Commit early, commit often.** As you make progress on any feature, bugfix, or experiment, create small, incremental commits so work is never lost.
- **Push regularly to GitHub.** Do not let local work sit unpushed for long periods. Remote history is the source of truth and backup.
- **Write clean, meaningful commit messages.**
  - Use present tense ("Add feature", "Fix bug", "Refactor pipeline").
  - Clearly describe *what* changed and *why*.
  - Keep messages concise but informative.
- **Never batch large unrelated changes into a single commit.** Keep commits logically scoped.
- **Checkpoint before risky changes.** Commit before refactors, large data operations, or model experiments.
- **Working rule:** At any moment, the repo should reflect the latest stable state of progress and be safely recoverable from GitHub.

This discipline ensures we never lose work, can roll back safely, and maintain a clear history of how the system evolves.

---

## What to tell admissions / reviewers

This project replicates and directly challenges an **operational NOAA forecasting tool** using the same publicly available data sources, adds a novel satellite imagery component that the operational model lacks, and deploys as a working live system. It was built independently as a demonstration of how computer engineering skills — satellite data processing, ML pipeline architecture, real-time systems — can be applied directly to unsolved problems in atmospheric science.
