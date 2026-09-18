# -*- coding: utf-8 -*-
"""
Telco Customer Churn - End-to-end analysis
Data cleaning -> EDA -> statistical tests -> feature engineering
-> multiple models (imbalance handling) -> feature importance -> conclusions
All figures are saved to ./analysis_output (no GUI blocking).
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                           f1_score, precision_score, recall_score, accuracy_score,
                           balanced_accuracy_score)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "analysis_output")
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(HERE, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# 1. Data loading & cleaning
# ----------------------------------------------------------------------
print("=" * 70)
print("1. DATA LOADING & CLEANING")
print("=" * 70)
df = pd.read_csv(DATA)
print(f"Raw shape: {df.shape}")
print("Duplicated rows:", df.duplicated().sum())

# TotalCharges is stored as text; convert, then inspect the blanks
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
mask_na = df["TotalCharges"].isna()
print(f"TotalCharges blanks: {mask_na.sum()}")
print("tenure of those rows:", sorted(df.loc[mask_na, "tenure"].unique()))
# All are brand-new customers (tenure == 0) who have not been charged yet -> 0 is correct
df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
print("Missing values after cleaning:", int(df.isnull().sum().sum()))

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = df.drop(columns=["customerID"])

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
print(f"Numerical features: {num_cols}")
print(f"Churn rate: {df['Churn'].mean():.2%}  (churned {df['Churn'].sum()} / {len(df)})")

# ----------------------------------------------------------------------
# 2. EDA visualizations
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# 2.1 target distribution
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
vc = df["Churn"].value_counts().sort_index()
axes[0].bar(["Retain", "Churn"], vc.values, color=["#4C9F70", "#E15759"])
for i, v in enumerate(vc.values):
    axes[0].text(i, v + 40, f"{v}\n({v/len(df):.1%})", ha="center", fontsize=10)
axes[0].set_title("Churn distribution (count)")
axes[0].set_ylabel("customers")
axes[1].pie(vc.values, labels=["Retain", "Churn"], autopct="%1.1f%%",
            colors=["#4C9F70", "#E15759"], startangle=90)
axes[1].set_title("Churn rate")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_churn_distribution.png"), bbox_inches="tight")
plt.close()

# 2.2 numeric features vs churn
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, col in zip(axes, num_cols):
    sns.kdeplot(df.loc[df.Churn == 0, col], ax=ax, label="Retain", fill=True, color="#4C9F70")
    sns.kdeplot(df.loc[df.Churn == 1, col], ax=ax, label="Churn", fill=True, color="#E15759")
    ax.set_title(f"{col} by churn")
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_numeric_by_churn.png"), bbox_inches="tight")
plt.close()

# 2.3 churn rate for key categorical features
key_cats = ["Contract", "InternetService", "PaymentMethod", "OnlineSecurity",
            "TechSupport", "OnlineBackup", "DeviceProtection",
            "PaperlessBilling", "Partner", "Dependents"]
fig, axes = plt.subplots(5, 2, figsize=(14, 22))
for ax, col in zip(axes.ravel(), key_cats):
    rates = df.groupby(col, observed=True)["Churn"].mean().sort_values(ascending=False)
    sns.barplot(x=rates.index, y=rates.values, ax=ax, color="#E15759")
    ax.axhline(df["Churn"].mean(), ls="--", color="gray", label="overall churn rate")
    ax.set_title(f"Churn rate by {col}")
    ax.set_ylabel("churn rate")
    ax.set_ylim(0, 0.75)
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=8)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_churn_rate_by_category.png"), bbox_inches="tight")
plt.close()

# 2.4 tenure histogram by churn
fig, ax = plt.subplots(figsize=(11, 4.2))
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack",
             bins=30, palette=["#4C9F70", "#E15759"], ax=ax)
ax.set_title("Customer count by tenure (stacked by churn)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_tenure_hist.png"), bbox_inches="tight")
plt.close()

# 2.5 MonthlyCharges vs TotalCharges
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df.sample(min(1500, len(df)), random_state=1),
                x="MonthlyCharges", y="TotalCharges", hue="Churn",
                palette=["#4C9F70", "#E15759"], alpha=0.5, ax=ax, s=18)
ax.set_title("MonthlyCharges vs TotalCharges")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "05_charges_scatter.png"), bbox_inches="tight")
plt.close()

# 2.6 correlation heatmap (label encoded)
df_enc = df.copy()
for c in cat_cols:
    df_enc[c] = pd.factorize(df_enc[c])[0]
plt.figure(figsize=(15, 12))
sns.heatmap(df_enc.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
            annot_kws={"size": 7}, cbar_kws={"shrink": 0.6})
plt.title("Feature correlation matrix (factorized)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "06_corr_heatmap.png"), bbox_inches="tight")
plt.close()
print("Figures saved to", OUT)

# ----------------------------------------------------------------------
# 3. Statistical tests
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. STATISTICAL TESTS")
print("=" * 70)

stat_rows = []
print("\n-- Chi-square tests (categorical feature vs churn) --")
for c in cat_cols:
    ct = pd.crosstab(df[c], df["Churn"])
    chi2, p, dof, _ = chi2_contingency(ct)
    # Cramer's V effect size
    n = ct.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    stat_rows.append({"feature": c, "test": "chi2", "statistic": chi2, "p_value": p, "effect_size": v})
    sig = "SIGNIFICANT" if p < 0.05 else "n.s."
    print(f"{c:18s} chi2={chi2:9.2f}  p={p:.2e}  Cramer's V={v:.3f}  {sig}")

print("\n-- Mann-Whitney U tests (numeric feature: churn vs retain) --")
for c in num_cols:
    x = df.loc[df.Churn == 1, c]
    y = df.loc[df.Churn == 0, c]
    u, p = mannwhitneyu(x, y)
    # rank-biserial style effect size
    effect = 2 * u / (len(x) * len(y)) - 1
    stat_rows.append({"feature": c, "test": "mannwhitneyu", "statistic": u,
                     "p_value": p, "effect_size": effect})
    print(f"{c:15s} median churn={x.median():8.2f}  retain={y.median():8.2f}  "
          f"p={p:.2e}  effect={effect:+.3f}")

pd.DataFrame(stat_rows).to_csv(os.path.join(OUT, "statistical_tests.csv"), index=False)

# ----------------------------------------------------------------------
# 4. Feature engineering
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. FEATURE ENGINEERING")
print("=" * 70)

model_df = df.copy()
model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)
# bool dummies -> int (pandas3 returns bool dummies)
for c in model_df.columns:
    if model_df[c].dtype == bool:
        model_df[c] = model_df[c].astype(int)

y = model_df["Churn"].astype(int)
X = model_df.drop(columns=["Churn"])
feature_names = X.columns.tolist()
print("Feature matrix:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print(f"Train: {X_train.shape}  churn rate {y_train.mean():.2%}")
print(f"Test : {X_test.shape}  churn rate {y_test.mean():.2%}")

# scaled copy for scale-sensitive models
scaler = StandardScaler()
scale_cols = [c for c in num_cols if c in X_train.columns]
X_train_s = X_train.copy()
X_test_s = X_test.copy()
X_train_s[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_s[scale_cols] = scaler.transform(X_test[scale_cols])

# SMOTE on scaled training data (only used for one comparison run)
sm = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = sm.fit_resample(X_train_s, y_train)
print(f"After SMOTE train: {X_train_sm.shape}  churn rate {y_train_sm.mean():.2%}")

# ----------------------------------------------------------------------
# 5. Modeling
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. MODEL TRAINING & EVALUATION")
print("=" * 70)

try:
    from xgboost import XGBClassifier
    has_xgb = True
except Exception:
    has_xgb = False

models = [
    ("LogisticRegression", LogisticRegression(max_iter=2000, class_weight="balanced",
                                            random_state=RANDOM_STATE), True),
    ("DecisionTree", DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                           random_state=RANDOM_STATE), False),
    ("RandomForest", RandomForestClassifier(n_estimators=400, max_depth=None,
                                          min_samples_leaf=3, class_weight="balanced",
                                          n_jobs=-1, random_state=RANDOM_STATE), False),
    ("GradientBoosting", GradientBoostingClassifier(random_state=RANDOM_STATE), False),
    ("HistGradientBoosting", HistGradientBoostingClassifier(random_state=RANDOM_STATE), False),
]
if has_xgb:
    models.append(("XGBoost",
                  XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.08,
                                subsample=0.9, colsample_bytree=0.9,
                                eval_metric="logloss", n_jobs=-1,
                                random_state=RANDOM_STATE), False))

rows, fitted = [], {}
pred_store = {}
for name, clf, scaled in models:
    xtr = X_train_s if scaled else X_train
    xte = X_test_s if scaled else X_test
    clf.fit(xtr, y_train)
    pred = clf.predict(xte)
    proba = clf.predict_proba(xte)[:, 1]
    fitted[name] = clf
    pred_store[name] = (pred, proba)
    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "balanced_acc": balanced_accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    })

# Extra comparison: Logistic regression on SMOTE data
lr_smote = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
lr_smote.fit(X_train_sm, y_train_sm)
pred_sm = lr_smote.predict(X_test_s)
proba_sm = lr_smote.predict_proba(X_test_s)[:, 1]
rows.append({
    "model": "LogReg+SMOTE",
    "accuracy": accuracy_score(y_test, pred_sm),
    "balanced_acc": balanced_accuracy_score(y_test, pred_sm),
    "precision": precision_score(y_test, pred_sm),
    "recall": recall_score(y_test, pred_sm),
    "f1": f1_score(y_test, pred_sm),
    "roc_auc": roc_auc_score(y_test, proba_sm),
})
pred_store["LogReg+SMOTE"] = (pred_sm, proba_sm)

res = pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")
print("\nModel comparison (sorted by ROC-AUC):")
print(res.to_string())
res.to_csv(os.path.join(OUT, "model_comparison.csv"))

# confusion matrices for top-3 by f1
top3 = res.sort_values("f1", ascending=False).head(3).index.tolist()
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, name in zip(axes, top3):
    pred, _ = pred_store[name]
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Retain", "Churn"], yticklabels=["Retain", "Churn"])
    ax.set_title(f"{name}\n(by F1)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "07_confusion_matrices.png"), bbox_inches="tight")
plt.close()

best_name = res.sort_values("f1", ascending=False).index[0]
print(f"\nBest model by F1: {best_name}")
print("\nClassification report:")
print(classification_report(y_test, pred_store[best_name][0],
      target_names=["Retain", "Churn"], digits=4))

# ----------------------------------------------------------------------
# 6. Feature importance
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("6. FEATURE IMPORTANCE")
print("=" * 70)

rf = fitted["RandomForest"]
imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
top15 = imp.head(15)
plt.figure(figsize=(9, 6))
sns.barplot(x=top15.values, y=top15.index, color="#4C72B0")
plt.title("Random Forest - top 15 feature importances")
plt.xlabel("importance")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "08_rf_feature_importance.png"), bbox_inches="tight")
plt.close()
print("Random Forest top-15:")
print(top15.to_string(float_format=lambda v: f"{v:.4f}"))

# Logistic regression coefficients (odds ratios)
lr = fitted["LogisticRegression"]
coef = pd.Series(lr.coef_[0], index=feature_names).sort_values(key=abs, ascending=False)
or_top = coef.head(15)
fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#E15759" if v > 0 else "#4C9F70" for v in or_top.values]
sns.barplot(x=or_top.values, y=or_top.index, palette=colors, ax=ax, hue=or_top.index, legend=False)
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Logistic Regression - top 15 coefficients (red: churn risk up)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "09_lr_coefficients.png"), bbox_inches="tight")
plt.close()

imp.to_csv(os.path.join(OUT, "rf_feature_importance.csv"))

print("\n" + "=" * 70)
print("ALL DONE. Artifacts in:", OUT)
print("=" * 70)
