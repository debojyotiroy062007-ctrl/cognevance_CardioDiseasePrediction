import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

# -------------------------
# Paths & Setup
# -------------------------
DATA_FILE = "cardio_train (1).csv"  # keep CSV in same folder as this script
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -------------------------
# Load & Basic Cleaning
# -------------------------
df = pd.read_csv(DATA_FILE, sep=";") if DATA_FILE.endswith(".csv") else pd.read_csv(DATA_FILE)

# Ensure expected columns exist
expected = {
    "id","age","gender","height","weight","ap_hi","ap_lo",
    "cholesterol","gluc","smoke","alco","active","cardio"
}
missing_cols = expected - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")

# Drop duplicates
df = df.drop_duplicates()

# Age from days → years
df["age_years"] = (df["age"] / 365.25).astype(float)

# Optional engineered feature
height_m = df["height"] / 100.0
df["bmi"] = (df["weight"] / (height_m**2)).replace([np.inf, -np.inf], np.nan)

# Clean unrealistic values (kept conservative)
df = df[
    (df["height"].between(120, 220)) &
    (df["weight"].between(30, 200)) &
    (df["age_years"].between(18, 100)) &
    (df["bmi"].between(10, 80))
]

# Fix swapped/invalid blood pressures where possible
swap_mask = df["ap_hi"] < df["ap_lo"]
df.loc[swap_mask, ["ap_hi", "ap_lo"]] = df.loc[swap_mask, ["ap_lo", "ap_hi"]].values

# Keep reasonable BP ranges
df = df[
    df["ap_hi"].between(80, 240) &
    df["ap_lo"].between(40, 200) &
    (df["ap_hi"] >= df["ap_lo"])
]

# -------------------------
# Features & Target
# -------------------------
target = "cardio"
features = [
    "age_years","gender","height","weight","bmi","ap_hi","ap_lo",
    "cholesterol","gluc","smoke","alco","active"
]

df = df.dropna(subset=features + [target])
X = df[features].copy()
y = df[target].copy()

# -------------------------
# EDA
# -------------------------
# Target balance
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Target Balance: cardio (0=No, 1=Yes)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_target_balance.png"))
plt.close()

# Numeric distributions
num_cols = ["age_years","height","weight","bmi","ap_hi","ap_lo"]
fig, axes = plt.subplots(nrows=int(np.ceil(len(num_cols)/3)), ncols=3, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution: {col}")
for j in range(i+1, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_numeric_distributions.png"))
plt.close()

# Categorical countplots
cat_cols = ["gender","cholesterol","gluc","smoke","alco","active"]
fig, axes = plt.subplots(nrows=int(np.ceil(len(cat_cols)/3)), ncols=3, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    sns.countplot(data=df, x=col)
    axes[i].set_title(f"Count: {col}")
for j in range(i+1, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_categorical_counts.png"))
plt.close()

# Correlation matrix (numeric)
corr_cols = num_cols + ["gender","cholesterol","gluc","smoke","alco","active"]
corr_df = df[corr_cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_df, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_correlation_matrix.png"))
plt.close()

# -------------------------
# Train / Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# -------------------------
# Modeling
# -------------------------
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ]),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=25))
    ]),
    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced_subsample"
    )
}

results = []
best_name, best_model, best_acc = None, None, -1

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    results.append([name, acc, prec, rec, f1])
    if acc > best_acc:
        best_name, best_model, best_acc = name, model, acc

res_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision(macro)","Recall(macro)","F1(macro)"])
res_df = res_df.sort_values(by="Accuracy", ascending=False)
res_df.to_csv(os.path.join(OUTPUT_DIR, "05_model_performance.csv"), index=False)

# -------------------------
# Evaluation of Best Model
# -------------------------
best_preds = best_model.predict(X_test)

report = classification_report(y_test, best_preds, digits=4)
with open(os.path.join(OUTPUT_DIR, "06_classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Best Model: {best_name}\n\n")
    f.write(report)

labels = np.sort(y_test.unique())
cm = confusion_matrix(y_test, best_preds, labels=labels, normalize="true")
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm_df, annot=True, fmt=".2f", cmap="Blues")
plt.title(f"Confusion Matrix (Normalized) - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_confusion_matrix.png"))
plt.close()

# Feature importance if RF
if best_name == "RandomForest":
    importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(7,5))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title("Feature Importance (RandomForest)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_feature_importance.png"))
    plt.close()
    importances.to_csv(os.path.join(OUTPUT_DIR, "08_feature_importance.csv"))

# -------------------------
# Save Best Model
# -------------------------
joblib.dump(best_model, os.path.join(OUTPUT_DIR, f"best_model_{best_name}.joblib"))
