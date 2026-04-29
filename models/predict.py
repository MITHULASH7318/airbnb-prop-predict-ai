"""
predict.py — loads trained models and exposes prediction functions
used by both the chatbot and the Streamlit app
"""
import pickle, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from utils.preprocess import engineer_features, investment_label, FEATURES

# ── Load models once at import time ───────────────────────────────────────────
# Resolve BASE robustly for both local and Streamlit Cloud
def _find_models_dir():
    candidates = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.getcwd(), "models"),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "price_model.pkl")):
            return c
    return candidates[0]
BASE = _find_models_dir()

def _load(name):
    path = os.path.join(BASE, name)
    with open(path, "rb") as f:
        return pickle.load(f)

price_model = _load("price_model.pkl")
roi_clf     = _load("roi_classifier.pkl")
score_model = _load("score_model.pkl")
occ_model   = _load("occ_model.pkl")

with open(os.path.join(BASE, "label_map.json")) as f:
    label_map = json.load(f)
inv_label_map = {v:k for k,v in label_map.items()}

with open(os.path.join(BASE, "feature_importance.json")) as f:
    feature_importance = json.load(f)

def predict_property(input_dict: dict) -> dict:
    """
    input_dict keys (all optional, will use defaults):
      city, area, property_type, bedrooms, bathrooms,
      review_score, occupancy_rate, distance_to_beach_km,
      distance_to_airport_km, num_reviews, amenities
    Returns dict with predicted price, ROI label, investment score, occupancy
    """
    defaults = {
        "city":"Dubai","area":"JVC","property_type":"Apartment",
        "bedrooms":2,"bathrooms":2,"review_score":4.5,
        "occupancy_rate":0.65,"distance_to_beach_km":2.0,
        "distance_to_airport_km":15.0,"num_reviews":50,
        "amenities":"WiFi|AC|Parking","peak_occupancy":0.80,"off_occupancy":0.45
    }
    row = {**defaults, **input_dict}
    df  = pd.DataFrame([row])
    df  = engineer_features(df)
    X   = df[FEATURES]

    price    = float(price_model.predict(X)[0])
    occ      = float(np.clip(occ_model.predict(X)[0], 0.1, 1.0))
    score    = float(np.clip(score_model.predict(X)[0], 0, 100))
    roi_cls  = int(roi_clf.predict(X)[0])
    roi_lbl  = inv_label_map.get(roi_cls, "Hold")

    annual_rev   = round(price * occ * 365 * 0.72)
    prop_price   = round(annual_rev / 0.08)
    roi_est      = round((annual_rev / prop_price) * 100, 2)

    top_features = sorted(feature_importance.items(), key=lambda x:-x[1])[:3]
    explanation  = f"Top drivers: {', '.join(f[0] for f in top_features)}"

    return {
        "predicted_nightly_price_usd": round(price),
        "predicted_occupancy_rate":    round(occ, 2),
        "predicted_annual_revenue_usd": annual_rev,
        "estimated_property_price_usd": prop_price,
        "estimated_roi_pct":           roi_est,
        "investment_label":            roi_lbl,
        "investment_score":            round(score),
        "explanation":                 explanation
    }

def top_properties(city: str = None, min_roi: float = 5.0,
                   max_price: int = None, top_n: int = 5) -> pd.DataFrame:
    """Return top N properties from processed data filtered by criteria."""
    # Try models/../data or cwd/data
    for base in [os.path.dirname(BASE), os.getcwd()]:
        p = os.path.join(base, "data", "processed", "listings_featured.csv")
        if os.path.exists(p):
            path = p
            break
    else:
        path = os.path.join(os.path.dirname(BASE), "data", "processed", "listings_featured.csv")
    df = pd.read_csv(path)
    if city:
        df = df[df["city"].str.lower() == city.lower()]
    df = df[df["roi_pct"] >= min_roi]
    if max_price:
        df = df[df["property_price_usd"] <= max_price]
    df = df.sort_values("investment_score", ascending=False)
    cols = ["id","city","area","property_type","bedrooms","nightly_price_usd",
            "roi_pct","investment_score","investment_label","review_score","occupancy_rate"]
    return df[cols].head(top_n)

if __name__ == "__main__":
    result = predict_property({"city":"Dubai","area":"Palm Jumeirah",
                                "property_type":"Villa","bedrooms":3,"review_score":4.8})
    print(json.dumps(result, indent=2))
    print("\nTop 5 Dubai properties:")
    print(top_properties("Dubai", min_roi=7, top_n=5).to_string(index=False))
