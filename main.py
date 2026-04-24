from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils import resample
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Spotify Popularity Predictor API", version="1.0.0")
@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model_bundle.pkl"

# ─── Pydantic Schemas ────────────────────────────────────────────────────────

class SongFeatures(BaseModel):
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    duration_ms: int
    time_signature: int
    explicit: int
    track_genre: int = 0   # label-encoded genre (0 if unknown)

class TrainResponse(BaseModel):
    message: str
    results: dict
    feature_importances: dict

# ─── Helpers ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms", "time_signature", "explicit",
    "track_genre",
]

def load_bundle():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model not trained yet. POST /train first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def save_bundle(bundle):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

def preprocess_df(df: pd.DataFrame):
    """Drop meta columns, encode categoricals, create target."""
    drop_cols = ["Unnamed: 0", "track_id", "track_name", "artists", "album_name"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    if "popularity" not in df.columns:
        raise ValueError("CSV must contain a 'popularity' column for training.")

    df["target"] = (df["popularity"] > 50).astype(int)
    df = df.drop(columns=["popularity"])

    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    return df

# ─── Endpoints ───────────────────────────────────────────────────────────────



@app.post("/train", response_model=TrainResponse)
async def train(file: UploadFile = File(...)):
    """Upload dataset.csv → train 3 models → return metrics."""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    try:
        df = preprocess_df(df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    X = df.drop("target", axis=1)
    y = df["target"]

    # Balance
    df_combined = pd.concat([X, y], axis=1)
    majority = df_combined[df_combined.target == 0]
    minority = df_combined[df_combined.target == 1]
    minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    balanced = pd.concat([majority, minority_up])
    X = balanced.drop("target", axis=1)
    y = balanced["target"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric="logloss", random_state=42),
    }

    results = {}
    best_name, best_f1, best_model = None, -1, None

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        f1 = f1_score(y_test, y_pred)
        results[name] = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1": round(f1, 4),
        }
        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, model

    rf = models["Random Forest"]
    importances = {feat: round(float(imp), 4)
                   for feat, imp in zip(feature_names, rf.feature_importances_)}
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    bundle = {
        "scaler": scaler,
        "models": models,
        "best_model_name": best_name,
        "feature_names": feature_names,
        "feature_importances": importances,
        "results": results,
    }
    save_bundle(bundle)

    return TrainResponse(
        message=f"Training complete. Best model: {best_name} (F1={best_f1:.4f})",
        results=results,
        feature_importances=importances,
    )


@app.post("/predict")
def predict_single(features: SongFeatures):
    """Predict popularity for a single song (manual input)."""
    bundle = load_bundle()
    scaler = bundle["scaler"]
    model = bundle["models"][bundle["best_model_name"]]
    feature_names = bundle["feature_names"]

    row = {col: getattr(features, col, 0) for col in feature_names}
    X = pd.DataFrame([row])
    X_s = scaler.transform(X)

    pred = int(model.predict(X_s)[0])
    proba = model.predict_proba(X_s)[0].tolist()

    return {
        "prediction": pred,
        "label": "Popular" if pred == 1 else "Not Popular",
        "confidence": round(max(proba), 4),
        "probabilities": {"not_popular": round(proba[0], 4), "popular": round(proba[1], 4)},
        "model_used": bundle["best_model_name"],
    }


@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    """Batch-predict from an uploaded CSV (no 'popularity' column needed)."""
    bundle = load_bundle()
    scaler = bundle["scaler"]
    model = bundle["models"][bundle["best_model_name"]]
    feature_names = bundle["feature_names"]

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    drop_cols = ["Unnamed: 0", "track_id", "track_name", "artists", "album_name", "popularity"]
    meta_df = df[["track_name", "artists"]].copy() if "track_name" in df.columns and "artists" in df.columns else None
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df.dropna(inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing columns: {missing}")

    X = df[feature_names]
    X_s = scaler.transform(X)
    preds = model.predict(X_s).tolist()
    probas = model.predict_proba(X_s)[:, 1].tolist()

    rows = []
    for i, (pred, prob) in enumerate(zip(preds, probas)):
        row = {"index": i, "prediction": pred,
               "label": "Popular" if pred == 1 else "Not Popular",
               "confidence": round(prob, 4)}
        if meta_df is not None and i < len(meta_df):
            row["track_name"] = meta_df.iloc[i]["track_name"]
            row["artists"] = meta_df.iloc[i]["artists"]
        rows.append(row)

    popular_count = sum(preds)
    return {
        "total": len(preds),
        "popular": popular_count,
        "not_popular": len(preds) - popular_count,
        "model_used": bundle["best_model_name"],
        "predictions": rows,
    }


@app.get("/metrics")
def get_metrics():
    """Return stored training metrics and feature importances."""
    bundle = load_bundle()
    return {
        "results": bundle["results"],
        "best_model": bundle["best_model_name"],
        "feature_importances": bundle["feature_importances"],
    }


@app.get("/model-info")
def model_info():
    """Quick check on whether model is trained."""
    if not os.path.exists(MODEL_PATH):
        return {"trained": False}
    bundle = load_bundle()
    return {
        "trained": True,
        "best_model": bundle["best_model_name"],
        "features": bundle["feature_names"],
    }