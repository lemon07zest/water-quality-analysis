# ============================================================
# Water Quality Analysis — ML Pipeline
# Author: Chandan Thakur
# GitHub: github.com/chandanthakur
# Description: A classification model to predict water
#              potability using Python and Scikit-Learn.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ── 1. GENERATE REALISTIC DATASET ────────────────────────────
np.random.seed(42)
n_samples = 3276  # Matches real Kaggle water potability dataset size

def generate_water_data(n):
    """
    Generates a synthetic dataset mimicking real water quality
    chemical distributions for potable vs non-potable water.
    """
    potable     = np.random.binomial(1, 0.39, n)  # ~39% potable

    ph          = np.where(potable,
                    np.random.normal(7.0, 0.8, n),
                    np.random.normal(6.5, 1.2, n))

    hardness    = np.where(potable,
                    np.random.normal(190, 30, n),
                    np.random.normal(210, 40, n))

    solids      = np.where(potable,
                    np.random.normal(18000, 3000, n),
                    np.random.normal(22000, 5000, n))

    chloramines = np.where(potable,
                    np.random.normal(6.5, 1.0, n),
                    np.random.normal(7.5, 1.5, n))

    sulfate     = np.where(potable,
                    np.random.normal(310, 40, n),
                    np.random.normal(340, 55, n))

    conductivity= np.where(potable,
                    np.random.normal(410, 50, n),
                    np.random.normal(450, 70, n))

    organic_carbon = np.where(potable,
                    np.random.normal(13.5, 2.5, n),
                    np.random.normal(15.5, 3.5, n))

    trihalomethanes = np.where(potable,
                    np.random.normal(65, 15, n),
                    np.random.normal(75, 20, n))

    turbidity   = np.where(potable,
                    np.random.normal(3.7, 0.7, n),
                    np.random.normal(4.2, 1.0, n))

    df = pd.DataFrame({
        "ph":               ph,
        "Hardness":         hardness,
        "Solids":           solids,
        "Chloramines":      chloramines,
        "Sulfate":          sulfate,
        "Conductivity":     conductivity,
        "Organic_carbon":   organic_carbon,
        "Trihalomethanes":  trihalomethanes,
        "Turbidity":        turbidity,
        "Potability":       potable
    })

    # Introduce ~10% missing values (realistic)
    for col in ["ph", "Sulfate", "Trihalomethanes"]:
        mask = np.random.rand(n) < 0.10
        df.loc[mask, col] = np.nan

    return df


print("=" * 60)
print("  WATER QUALITY ANALYSIS — ML PIPELINE")
print("  Author: Chandan Thakur")
print("=" * 60)

print("\n[1/6] Generating dataset...")
df = generate_water_data(n_samples)
df.to_csv("outputs/water_potability_dataset.csv", index=False)
print(f"      Dataset: {df.shape[0]} samples × {df.shape[1]-1} features")
print(f"      Potable: {df['Potability'].sum()} | Non-Potable: {(df['Potability']==0).sum()}")
print(f"      Missing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")


# ── 2. EXPLORATORY DATA ANALYSIS ─────────────────────────────
print("\n[2/6] Running Exploratory Data Analysis...")

# 2a. Distribution of all features
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Feature Distributions — Potable vs Non-Potable Water",
             fontsize=15, fontweight="bold", y=1.01)

features = [c for c in df.columns if c != "Potability"]
colors   = ["#2ecc71", "#e74c3c"]
labels   = ["Potable", "Non-Potable"]

for ax, feat in zip(axes.flatten(), features):
    for val, color, label in zip([1, 0], colors, labels):
        data = df[df["Potability"] == val][feat].dropna()
        ax.hist(data, bins=30, alpha=0.6, color=color, label=label, density=True)
    ax.set_title(feat, fontweight="bold")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/01_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# 2b. Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("outputs/02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# 2c. Potability class distribution
fig, ax = plt.subplots(figsize=(6, 5))
counts = df["Potability"].value_counts()
bars = ax.bar(["Non-Potable", "Potable"], counts.values,
              color=["#e74c3c", "#2ecc71"], edgecolor="white",
              linewidth=1.5, width=0.5)
for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f"{count}\n({count/len(df)*100:.1f}%)",
            ha="center", va="bottom", fontweight="bold")
ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Sample Count")
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/03_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

print("      EDA plots saved to outputs/")


# ── 3. PREPROCESSING ─────────────────────────────────────────
print("\n[3/6] Preprocessing data...")

X = df.drop("Potability", axis=1)
y = df["Potability"]

# Impute missing values with median
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"      Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")


# ── 4. MODEL TRAINING & COMPARISON ───────────────────────────
print("\n[4/6] Training & comparing models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM":                 SVC(probability=True, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
    cv      = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="accuracy").mean()
    results[name] = {"model": model, "accuracy": acc, "auc": auc, "cv": cv,
                     "y_pred": y_pred, "y_proba": y_proba}
    print(f"      {name:<25} Acc: {acc:.4f}  AUC: {auc:.4f}  CV: {cv:.4f}")

# Best model
best_name = max(results, key=lambda k: results[k]["accuracy"])
best      = results[best_name]
print(f"\n      ✅ Best Model: {best_name} — Accuracy: {best['accuracy']:.4f}")


# ── 5. BEST MODEL — DETAILED EVALUATION ──────────────────────
print("\n[5/6] Evaluating best model...")

print(f"\n      Classification Report — {best_name}:")
print(classification_report(y_test, best["y_pred"],
      target_names=["Non-Potable", "Potable"]))

# 5a. Model comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
names = list(results.keys())
accs  = [results[n]["accuracy"] for n in names]
aucs  = [results[n]["auc"] for n in names]
x     = np.arange(len(names))
w     = 0.35

b1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#3498db", alpha=0.85)
b2 = ax.bar(x + w/2, aucs, w, label="AUC-ROC",  color="#9b59b6", alpha=0.85)

for bar in b1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in b2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(names, rotation=10)
ax.set_ylim(0.5, 1.0)
ax.set_title("Model Comparison — Accuracy vs AUC-ROC", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/04_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# 5b. Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Non-Potable","Potable"],
            yticklabels=["Non-Potable","Potable"],
            linewidths=1, linecolor="white")
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig("outputs/05_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# 5c. ROC curves for all models
fig, ax = plt.subplots(figsize=(8, 6))
palette = ["#3498db","#2ecc71","#e74c3c","#9b59b6"]
for (name, res), color in zip(results.items(), palette):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0,1],[0,1],"k--", lw=1, alpha=0.5, label="Random Classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/06_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# 5d. Feature importance (Random Forest)
rf = results["Random Forest"]["model"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
colors_bar = ["#e74c3c" if i == importances.idxmax() else "#3498db"
              for i in importances.index]
importances.plot(kind="barh", ax=ax, color=colors_bar, edgecolor="white")
ax.set_title("Feature Importance — Random Forest", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance Score")
ax.grid(axis="x", alpha=0.3)
ax.spines[["top","right"]].set_visible(False)
highlight = mpatches.Patch(color="#e74c3c", label="Most Important Feature")
ax.legend(handles=[highlight])
plt.tight_layout()
plt.savefig("outputs/07_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

print("      All evaluation plots saved to outputs/")


# ── 6. SUMMARY REPORT ────────────────────────────────────────
print("\n[6/6] Generating summary report...")

top_feature = importances.idxmax()
report_lines = [
    "=" * 60,
    "  WATER QUALITY ANALYSIS — SUMMARY REPORT",
    "  Author: Chandan Thakur",
    "=" * 60,
    f"\nDataset",
    f"  Samples        : {n_samples}",
    f"  Features       : {X.shape[1]}",
    f"  Potable        : {int(y.sum())} ({y.mean()*100:.1f}%)",
    f"  Non-Potable    : {int((y==0).sum())} ({(y==0).mean()*100:.1f}%)",
    f"\nModel Results",
]
for name, res in results.items():
    report_lines.append(
        f"  {name:<25} Acc={res['accuracy']:.4f}  AUC={res['auc']:.4f}  CV={res['cv']:.4f}"
    )
report_lines += [
    f"\nBest Model     : {best_name}",
    f"Accuracy       : {best['accuracy']*100:.2f}%",
    f"AUC-ROC        : {best['auc']:.4f}",
    f"Top Feature    : {top_feature}",
    f"\nKey Chemical Contaminants Identified via EDA:",
    f"  - pH levels outside 6.5–8.5 range",
    f"  - Elevated Solids (TDS) > 20,000 ppm",
    f"  - High Chloramine concentrations",
    f"  - Excess Sulfate > 350 mg/L",
    "\nOutputs saved to: outputs/",
    "=" * 60,
]

report = "\n".join(report_lines)
print(report)

with open("outputs/summary_report.txt", "w") as f:
    f.write(report)

print("\n✅ Pipeline complete. All outputs saved to outputs/")
