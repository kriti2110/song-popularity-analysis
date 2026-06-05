<div align="center">

<h1>🎵 Song Popularity Analysis & Prediction</h1>

<p>A machine learning pipeline that predicts whether a song will be popular — trained on 114,000+ Spotify tracks with a FastAPI backend and interactive web frontend.</p>

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AD3?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)

![Task](https://img.shields.io/badge/Task-Binary%20Classification-6366f1?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-114K%20Spotify%20Tracks-1DB954?style=flat-square)
![Metric](https://img.shields.io/badge/Metric-Accuracy%20%7C%20F1-2ea44f?style=flat-square)

</div>

---

## Overview

Predicts song popularity (popular / not popular) using audio features extracted from Spotify. A song is labelled **popular** if its popularity score exceeds 50. Handles class imbalance via upsampling and compares Logistic Regression, Random Forest, and XGBoost — XGBoost achieves the best F1 and AUC-ROC.

| Property | Details |
|----------|---------|
| ![](https://img.shields.io/badge/Problem-6366f1?style=flat-square) | Binary classification: popular (score > 50) vs not popular |
| ![](https://img.shields.io/badge/Dataset-1DB954?style=flat-square) | 114,000+ Spotify tracks with 15+ audio features |
| ![](https://img.shields.io/badge/Best_Model-f59e0b?style=flat-square) | XGBoost with SMOTE upsampling |
| ![](https://img.shields.io/badge/Backend-009688?style=flat-square) | FastAPI REST API with `/predict` and `/upload` endpoints |
| ![](https://img.shields.io/badge/Frontend-f97316?style=flat-square) | HTML/JS interface for live single-track predictions |

### Audio Features Used

| Feature | Description |
|---------|-------------|
| `danceability` | How suitable a track is for dancing (0–1) |
| `energy` | Perceptual measure of intensity and activity (0–1) |
| `valence` | Musical positiveness — happy vs sad (0–1) |
| `tempo` | Estimated beats per minute |
| `loudness` | Overall loudness in dB |
| `acousticness` | Confidence the track is acoustic (0–1) |
| `instrumentalness` | Predicts whether a track has no vocals (0–1) |
| `speechiness` | Presence of spoken words (0–1) |
| `liveness` | Detects presence of a live audience (0–1) |
| `duration_ms` | Track duration in milliseconds |

---

## Model Pipeline

```
dataset.csv (114K tracks)
       │
       ▼
┌──────────────────┐
│  Data Cleaning   │  ← drop duplicates, handle nulls
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│  Feature Eng.    │  ← encode genres, scale audio features
└───────┬──────────┘
        │
        ▼
┌──────────────────────────────────┐
│  Model Comparison                │
│  ├─ Logistic Regression          │
│  ├─ Random Forest                │
│  └─ XGBoost          ← best      │
└───────┬──────────────────────────┘
        │
        ▼
  Popularity Prediction
  (popular / not popular + confidence)
```

### Handling Class Imbalance
Popular tracks are a minority class — upsampling via `sklearn.utils.resample` is applied to balance the training set before fitting.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Serve frontend UI |
| `POST` | `/predict` | Predict popularity from audio features |
| `POST` | `/upload` | Upload CSV for batch predictions |
| `GET`  | `/health` | Health check |

### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"danceability": 0.8, "energy": 0.7, "tempo": 120, "loudness": -5.2}'
```

---

## Evaluation

![Accuracy](https://img.shields.io/badge/Accuracy-2ea44f?style=flat-square)
![F1 Score](https://img.shields.io/badge/F1%20Score-3b82f6?style=flat-square)
![Precision](https://img.shields.io/badge/Precision-f97316?style=flat-square)
![Recall](https://img.shields.io/badge/Recall-8b5cf6?style=flat-square)

---

## Repository Structure

```
song-popularity-analysis/
├── analysis.py      # EDA, model training, and evaluation
├── main.py          # FastAPI backend with prediction endpoints
├── index.html       # Web frontend
├── dataset.csv      # 114K Spotify tracks dataset
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# Open http://localhost:8000
```

---

## Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AD3?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=for-the-badge)

</div>

---

## Author

<div align="center">

**Kriti Raj** — B.Tech CSE (AI/ML), KIIT University

[![GitHub](https://img.shields.io/badge/GitHub-kriti2110-181717?style=flat-square&logo=github)](https://github.com/kriti2110)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Kriti%20Raj-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/kriti-raj-5b398236a)

</div>


 
