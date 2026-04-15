# Hurricane Rapid Intensification Alert System

A real-time machine learning system that predicts the probability of **Rapid Intensification (RI)** in active Atlantic and Eastern Pacific hurricanes — hours before the National Hurricane Center (NHC) issues an advisory.

> **RI definition (NHC):** a +35 kt increase in max sustained winds within 24 hours.  
> RI is the most dangerous and least-solved problem in operational tropical meteorology.

---

## What this does

This project replicates and improves upon the NHC's operational **SHIPS-RII** model (a 1990s-era logistic regression) by replacing it with a modern ML ensemble trained on the same public NOAA/NHC data sources, augmented with **GOES-16 satellite imagery features** that the operational model does not use.

**Live output:** A deployed React dashboard showing all active storms, color-coded by RI probability, with trend charts and NHC comparison.

---

## Architecture

```
Data sources                   ML models              Live system
─────────────                  ─────────              ───────────
HURDAT2 (1851–2023)  ──┐
SHIPS dev dataset    ──┼──► XGBoost (tabular)  ──┐
GOES-16 IR imagery   ──┘    LSTM (temporal)    ──┴──► Ensemble ──► FastAPI ──► React UI
NHC advisory RSS                                        + calibration
```

### Models

| Model | Input | Target |
|-------|-------|--------|
| XGBoost | SHIPS + satellite features (scalar, current timestep) | RI probability |
| LSTM | 48-hour rolling window (8 timesteps × N features) | RI probability |
| Ensemble | Average of XGBoost + LSTM (+ isotonic calibration) | Calibrated RI probability |

### Operational benchmark

| Metric | SHIPS-RII (NHC baseline) | This project target |
|--------|--------------------------|---------------------|
| AUC-ROC | — | > 0.78 |
| Brier Skill Score | ~0.08 | > 0.15 |
| POD @ 40% | 0.42 | > 0.50 |
| FAR @ 40% | 0.72 | < 0.65 |

---

## Data sources

| Source | What | Access |
|--------|------|--------|
| [HURDAT2](https://www.nhc.noaa.gov/data/hurdat/) | 6-hourly storm observations, 1851–2023 | Public HTTP |
| [SHIPS developmental dataset](https://rammb.cira.colostate.edu/research/tropical_cyclones/ships/developmental_data.asp) | ~50 atmospheric environment predictors | Public HTTP |
| [GOES-16 (AWS)](https://registry.opendata.aws/noaa-goes/) | Band 13 IR brightness temperature | Public S3 (no auth) |
| [NHC advisory RSS](https://www.nhc.noaa.gov/aboutrss.shtml) | Real-time storm position, intensity, forecast | Public XML feed |

---

## Project structure

```
hurricane-ri-alert/
├── data/
│   ├── raw/                   # downloaded source files (gitignored)
│   ├── processed/             # feature-engineered DataFrames (gitignored)
│   └── scripts/
│       ├── fetch_hurdat2.py   # download + parse HURDAT2
│       ├── fetch_ships.py     # download + parse SHIPS dev data
│       ├── fetch_goes16.py    # pull GOES-16 tiles from S3
│       └── label_ri_events.py # compute RI labels from HURDAT2
│
├── model/
│   ├── train_xgboost.py
│   ├── train_lstm.py
│   ├── ensemble.py
│   ├── calibrate.py
│   ├── evaluate.py
│   └── artifacts/             # saved model files (gitignored)
│
├── pipeline/
│   ├── ingestor.py            # real-time NHC + GOES polling
│   ├── feature_builder.py     # assembles live feature vector per storm
│   └── inference.py           # runs ensemble on live feature vector
│
├── api/
│   ├── main.py                # FastAPI app
│   ├── routes/
│   │   ├── storms.py          # GET /storms
│   │   └── history.py         # GET /storms/{id}/history
│   └── scheduler.py           # APScheduler, polls every 30 min
│
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── components/
    │   │   ├── StormMap.jsx       # Leaflet map, RI-scored markers
    │   │   ├── StormCard.jsx      # per-storm sidebar card
    │   │   └── RITrendChart.jsx   # recharts probability timeline
    │   └── lib/
    │       └── api.js
    └── public/
```

---

## Setup

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment variables

Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
```

### Data pipeline (run in order)

```bash
python data/scripts/fetch_hurdat2.py
python data/scripts/fetch_ships.py
python data/scripts/fetch_goes16.py
python data/scripts/label_ri_events.py
```

### Train models

```bash
python model/train_xgboost.py
python model/train_lstm.py
python model/ensemble.py
python model/calibrate.py
python model/evaluate.py
```

### Run the API

```bash
uvicorn api.main:app --reload
```

### Run the frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Key dependencies

```
xarray, netCDF4, boto3, s3fs          # satellite data
xgboost, torch, scikit-learn          # ML
imbalanced-learn                       # SMOTE for class imbalance
pandas, numpy                          # data wrangling
fastapi, uvicorn, apscheduler         # API + scheduling
feedparser, xmltodict                  # NHC RSS parsing
```

See `requirements.txt` for pinned versions.

---

## Scientific context

- RI threshold (NHC): +35 kt / 24 hours
- Atlantic climatological RI rate: ~10–15% of 6-hourly observations
- Operational baseline: SHIPS-RII (Kaplan & DeMaria 2003)
- Key reference: Zheng et al. (2023) — deep learning for RI prediction
- NHC 5-year strategic plan (2020–2025) lists RI as top scientific priority

---

## Build phases

- [x] Phase 0 — Repo setup, CLAUDE.md, project structure
- [ ] Phase 1 — Data pipeline (HURDAT2 + SHIPS + labels)
- [ ] Phase 2 — XGBoost tabular baseline
- [ ] Phase 3 — GOES-16 satellite features
- [ ] Phase 4 — LSTM + ensemble + calibration
- [ ] Phase 5 — Real-time inference pipeline
- [ ] Phase 6 — FastAPI backend
- [ ] Phase 7 — React frontend
- [ ] Phase 8 — Deploy (Railway + Vercel)
