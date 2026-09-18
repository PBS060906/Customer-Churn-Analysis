# -*- coding: utf-8 -*-
"""
Telco Customer Churn - Advanced stage
1) GridSearchCV tuning (n_jobs=1, compact grids, 3-fold stratified CV)
2) OOF probabilities for honest threshold selection (max F1)
3) ROC / PR / KS evaluation on the held-out test set
4) Chinese analysis report (分析报告.md)
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             roc_auc_score, f1_score, precision_score, recall_score,
                             accuracy_score, balanced_accuracy_score, confusion_matrix)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

RS = 42
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "analysis_output")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------
# Data (same pipeline as churn_analysis.py)
# ----------------------------------------------------------------------
df = pd.read_csv(os.path.join(HERE, "WA_Fn-UseC_-Telco-Customer-Churn.csv"))
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = df.drop(columns=["customerID"])
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

model_df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
for c in model_df.columns:
    if model_df[c].dtype == bool:
        model_df[c] = model_df[c].astype(int)
y = model_df["Churn"].astype(int)
X = model_df.drop(columns=["Churn"])
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RS)

scaler = StandardScaler()
X_train_s = X_train.copy()
X_test_s = X_test.copy()
X_train_s[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_s[num_cols] = scaler.transform(X_test[num_cols])
print(f"Data ready: train {X_train.shape}, test {X_test.shape}")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ----------------------------------------------------------------------
# 1. Grid search (n_jobs=1 to avoid loky/resource issues on Windows)
# ----------------------------------------------------------------------
print("\n[1/4] GridSearchCV tuning ...")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RS)

search_space = [
    ("LogisticRegression",
     LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RS),
     {"C": [0.01, 0.1, 1.0, 10.0]},
     True),
    ("RandomForest",
     RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=-1),
     {"max_depth": [6, 10, None], "min_samples_leaf": [2, 5],
      "class_weight": ["balanced", "balanced_subsample"]},
     False),
    ("GradientBoosting",
     GradientBoostingClassifier(random_state=RS),
     {"n_estimators": [100, 200], "max_depth": [2, 3],
      "learning_rate": [0.05, 0.1]},
     False),
]
if HAS_XGB:
    search_space.append((
        "XGBoost",
        XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=RS),
        {"n_estimators": [200, 400], "max_depth": [3, 5],
         "learning_rate": [0.05, 0.1], "subsample": [0.9]},
        False))

best_models, best_params = {}, {}
rows = []
for name, est, grid, scaled in search_space:
    xtr = X_train_s if scaled else X_train
    gs = GridSearchCV(est, grid, cv=cv, scoring="roc_auc",
                      n_jobs=1, refit=True, verbose=0)
    gs.fit(xtr, y_train)
    best_models[name] = gs.best_estimator_
    best_params[name] = gs.best_params_
    rows.append({"model": name, "cv_roc_auc": gs.best_score_,
                 "params": json.dumps(gs.best_params_, ensure_ascii=False)})
    print(f"  {name:20s} CV-AUC={gs.best_score_:.4f}  {gs.best_params_}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "tuned_gridsearch.csv"), index=False)

# ----------------------------------------------------------------------
# 2. OOF probabilities -> honest threshold (maximize F1 on training OOF)
# ----------------------------------------------------------------------
print("\n[2/4] OOF threshold optimization ...")
thresholds, oof_proba = {}, {}
for name, est in best_models.items():
    scaled = name == "LogisticRegression"
    xtr = X_train_s if scaled else X_train
    oof = cross_val_predict(est, xtr, y_train, cv=cv, method="predict_proba",
                            n_jobs=1)[:, 1]
    oof_proba[name] = oof
    ths = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_train, (oof >= t).astype(int)) for t in ths]
    best_t = ths[int(np.argmax(f1s))]
    thresholds[name] = best_t
    print(f"  {name:20s} OOF best threshold={best_t:.2f}  OOF-F1={max(f1s):.4f}")

# ----------------------------------------------------------------------
# 3. Test-set evaluation: default 0.5 vs tuned threshold; ROC / PR / KS
# ----------------------------------------------------------------------
print("\n[3/4] Test-set evaluation ...")
eval_rows, curves = [], {}
for name, est in best_models.items():
    scaled = name == "LogisticRegression"
    xte = X_test_s if scaled else X_test
    proba = est.predict_proba(xte)[:, 1]
    pred05 = (proba >= 0.5).astype(int)
    t = thresholds[name]
    pred_t = (proba >= t).astype(int)

    fpr, tpr, rth = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    ks_idx = np.argmax(tpr - fpr)
    ks = tpr[ks_idx] - fpr[ks_idx]
    ks_t = rth[ks_idx]
    prec, recl, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    curves[name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc,
                    "prec": prec, "recl": recl, "ap": ap,
                    "ks": ks, "ks_t": ks_t, "proba": proba}

    eval_rows.append({"model": name, "threshold": 0.5, "kind": "default",
                      "accuracy": accuracy_score(y_test, pred05),
                      "balanced_acc": balanced_accuracy_score(y_test, pred05),
                      "precision": precision_score(y_test, pred05),
                      "recall": recall_score(y_test, pred05),
                      "f1": f1_score(y_test, pred05),
                      "roc_auc": roc_auc, "ks": ks})
    eval_rows.append({"model": name, "threshold": round(t, 2), "kind": "tuned",
                      "accuracy": accuracy_score(y_test, pred_t),
                      "balanced_acc": balanced_accuracy_score(y_test, pred_t),
                      "precision": precision_score(y_test, pred_t),
                      "recall": recall_score(y_test, pred_t),
                      "f1": f1_score(y_test, pred_t),
                      "roc_auc": roc_auc, "ks": ks})

ev = pd.DataFrame(eval_rows)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")
print(ev.to_string(index=False))
ev.to_csv(os.path.join(OUT, "tuned_evaluation.csv"), index=False)

# ROC curves
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for name, c in curves.items():
    axes[0].plot(c["fpr"], c["tpr"], lw=1.8,
                 label=f"{name} (AUC={c['auc']:.3f}, KS={c['ks']:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC curves (tuned models)")
axes[0].legend(fontsize=8, loc="lower right")

for name, c in curves.items():
    axes[1].plot(c["recl"], c["prec"], lw=1.8,
                 label=f"{name} (AP={c['ap']:.3f})")
axes[1].axhline(y=y_test.mean(), ls="--", color="gray", label="random baseline")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall curves")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "10_roc_pr_curves.png"), bbox_inches="tight")
plt.close()

# Threshold-F1 curves (OOF) + confusion matrix of the final model
final_name = ev[ev.kind == "tuned"].sort_values("f1", ascending=False).iloc[0]["model"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
for name in best_models:
    oof = oof_proba[name]
    ths = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_train, (oof >= t).astype(int)) for t in ths]
    lw = 2.6 if name == final_name else 1.2
    axes[0].plot(ths, f1s, lw=lw, label=name)
    axes[0].axvline(thresholds[name], ls=":", lw=1, alpha=0.7)
axes[0].set_xlabel("decision threshold")
axes[0].set_ylabel("F1 (OOF train)")
axes[0].set_title("Threshold vs F1 (chosen thresholds as dotted lines)")
axes[0].legend(fontsize=8)

cm = confusion_matrix(y_test, (curves[final_name]["proba"] >= thresholds[final_name]).astype(int))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1],
            xticklabels=["Retain", "Churn"], yticklabels=["Retain", "Churn"])
axes[1].set_title(f"Confusion matrix: {final_name}\nthreshold={thresholds[final_name]:.2f}")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "11_threshold_and_cm.png"), bbox_inches="tight")
plt.close()

# Feature importance from tuned RF + tuned LR coefficients
rf = best_models["RandomForest"]
imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
lr = best_models["LogisticRegression"]
coef = pd.Series(lr.coef_[0], index=feature_names).sort_values(key=abs, ascending=False)

# Persist key numbers for the report step
summary = {
    "n_total": int(len(df)), "n_churn": int(y.sum()),
    "churn_rate": float(y.mean()),
    "best_params": {k: {p_: (float(v_) if isinstance(v_, (np.floating, float)) else v_)
                        for p_, v_ in vp.items()}
                    for k, vp in best_params.items()},
    "thresholds": {k: float(v) for k, v in thresholds.items()},
    "final_model": final_name,
    "rf_top10": imp.head(10).round(4).to_dict(),
    "lr_top10_risk": coef[coef > 0].head(8).round(3).to_dict(),
    "lr_top10_protect": (coef[coef < 0].sort_values()).head(8).round(3).to_dict(),
    "test_metrics": ev.round(4).to_dict(orient="records"),
    "ks": {k: round(float(v["ks"]), 4) for k, v in curves.items()},
    "roc_auc": {k: round(float(v["auc"]), 4) for k, v in curves.items()},
}
with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print(f"\nFinal model (best tuned F1): {final_name}, threshold={thresholds[final_name]:.2f}")
print("Artifacts saved.")
