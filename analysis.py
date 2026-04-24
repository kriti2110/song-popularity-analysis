# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils import resample
sns.set_style("whitegrid")

# =========================
# 2. LOAD DATA
# =========================
df = pd.read_csv("dataset.csv")

# =========================
# 3. DATA CLEANING
# =========================
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# =========================
# 4. CREATE TARGET
# =========================
df['target'] = df['popularity'].apply(lambda x: 1 if x > 50 else 0)

# =========================
# 5. DROP UNNECESSARY COLUMNS
# =========================
drop_cols = ['Unnamed: 0','track_id','track_name','artists','album_name','popularity']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# =========================
# 6. ENCODE CATEGORICAL
# =========================
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# =========================
# 7. SPLIT FEATURES
# =========================
X = df.drop('target', axis=1)
y = df['target']

# =========================
# 8. HANDLE IMBALANCE
# =========================
df_combined = pd.concat([X, y], axis=1)

df_majority = df_combined[df_combined.target == 0]
df_minority = df_combined[df_combined.target == 1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])

X = df_balanced.drop('target', axis=1)
y = df_balanced['target']

# =========================
# 9. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 10. SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 11. MODELS (TOP 3)
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )
}

# =========================
# 12. TRAIN & STORE RESULTS
# =========================
results_df = pd.DataFrame(columns=["Model","Accuracy","Precision","Recall","F1"])

for name, model in models.items():
    print(f"\nTraining: {name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results_df.loc[len(results_df)] = [name, acc, prec, rec, f1]

print("\n📊 FINAL RESULTS:\n")
print(results_df)


# =========================
# 📊 GRAPH 1: PERFORMANCE COMPARISON
# =========================
results_df.set_index("Model").plot(kind="bar", figsize=(10,6))
plt.title("Algorithm Performance Comparison", fontsize=14)
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# =========================
# 📊 GRAPH 2: CORRELATION HEATMAP
# =========================
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.show()

# =========================
# 📊 GRAPH 3: FEATURE IMPORTANCE
# =========================
rf = models["Random Forest"]
importances = rf.feature_importances_

feat_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x="Importance", y="Feature", data=feat_df)
plt.title("Feature Importance (Random Forest)", fontsize=14)
plt.tight_layout()
plt.show()

# =========================
# 📊 GRAPH 4: TARGET DISTRIBUTION
# =========================
plt.figure(figsize=(8,5))
sns.histplot(y, bins=2, kde=True)
plt.title("Target Distribution (Popular vs Not Popular)", fontsize=14)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()