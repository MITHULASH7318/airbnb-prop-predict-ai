"""
Train all ML models:
  1. Price predictor      → XGBoost regression  (predict nightly_price_usd)
  2. ROI classifier       → Random Forest        (Strong Buy / Buy / Hold / Avoid)
  3. Investment scorer    → XGBoost regression   (0-100 investment_score)
  4. Occupancy forecaster → XGBoost regression   (occupancy_rate)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import pickle, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_absolute_error, r2_score,
                              accuracy_score, classification_report)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from utils.preprocess import engineer_features, investment_label, FEATURES

os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ── Load & prep ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/combined_listings.csv")
df = engineer_features(df)
df["investment_label"] = df["roi_pct"].apply(investment_label)

label_map = {"Avoid":0, "Hold":1, "Buy":2, "Strong Buy":3}
df["label_enc"] = df["investment_label"].map(label_map)

X = df[FEATURES]
metrics = {}

# ── 1. Price predictor ────────────────────────────────────────────────────────
print("\n=== Model 1: Price Predictor ===")
y_price = df["nightly_price_usd"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y_price, test_size=0.2, random_state=42)
price_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                random_state=42, verbosity=0)
price_model.fit(X_tr, y_tr)
preds = price_model.predict(X_te)
mae   = round(mean_absolute_error(y_te, preds), 2)
r2    = round(r2_score(y_te, preds), 4)
print(f"MAE: ${mae}  |  R²: {r2}")
metrics["price_predictor"] = {"MAE_usd": mae, "R2": r2}
with open("models/price_model.pkl","wb") as f: pickle.dump(price_model, f)

# ── 2. ROI Classifier ─────────────────────────────────────────────────────────
print("\n=== Model 2: ROI Investment Classifier ===")
y_cls = df["label_enc"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
clf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
clf.fit(X_tr, y_tr)
preds_cls = clf.predict(X_te)
acc = round(accuracy_score(y_te, preds_cls), 4)
print(f"Accuracy: {acc}")
present = sorted(y_te.unique())
names   = [k for k,v in sorted(label_map.items(), key=lambda x:x[1]) if v in present]
print(classification_report(y_te, preds_cls, labels=present, target_names=names))
metrics["roi_classifier"] = {"accuracy": acc}
with open("models/roi_classifier.pkl","wb") as f: pickle.dump(clf, f)
with open("models/label_map.json","w") as f: json.dump(label_map, f)

# ── 3. Investment Score Model ─────────────────────────────────────────────────
print("\n=== Model 3: Investment Score Regressor ===")
y_score = df["investment_score"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y_score, test_size=0.2, random_state=42)
score_model = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1,
                                random_state=42, verbosity=0)
score_model.fit(X_tr, y_tr)
preds_s = score_model.predict(X_te)
mae_s = round(mean_absolute_error(y_te, preds_s), 2)
r2_s  = round(r2_score(y_te, preds_s), 4)
print(f"MAE: {mae_s} pts  |  R²: {r2_s}")
metrics["investment_scorer"] = {"MAE_pts": mae_s, "R2": r2_s}
with open("models/score_model.pkl","wb") as f: pickle.dump(score_model, f)

# ── 4. Occupancy Forecaster ───────────────────────────────────────────────────
print("\n=== Model 4: Occupancy Rate Forecaster ===")
y_occ = df["occupancy_rate"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y_occ, test_size=0.2, random_state=42)
occ_model = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.08,
                               random_state=42, verbosity=0)
occ_model.fit(X_tr, y_tr)
preds_o = occ_model.predict(X_te)
mae_o = round(mean_absolute_error(y_te, preds_o), 4)
r2_o  = round(r2_score(y_te, preds_o), 4)
print(f"MAE: {mae_o}  |  R²: {r2_o}")
metrics["occupancy_forecaster"] = {"MAE": mae_o, "R2": r2_o}
with open("models/occ_model.pkl","wb") as f: pickle.dump(occ_model, f)

# ── Feature importance (for SHAP-style explanation) ───────────────────────────
fi = dict(zip(FEATURES, price_model.feature_importances_.tolist()))
with open("models/feature_importance.json","w") as f:
    json.dump(fi, f, indent=2)

# ── Save metrics ──────────────────────────────────────────────────────────────
with open("models/metrics.json","w") as f: json.dump(metrics, f, indent=2)

# ── Save processed data ───────────────────────────────────────────────────────
df.to_csv("data/processed/listings_featured.csv", index=False)

print("\n✅ All models saved to /models/")
print(json.dumps(metrics, indent=2))
