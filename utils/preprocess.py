import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle, os

FEATURES = [
    "bedrooms","bathrooms","review_score","occupancy_rate",
    "distance_to_beach_km","distance_to_airport_km","num_reviews",
    "amenity_count","has_pool","has_wifi","has_ac","has_parking",
    "city_enc","area_enc","property_type_enc"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Amenity features
    df["amenity_count"]  = df["amenities"].fillna("").apply(lambda x: len(x.split("|")) if x else 0)
    df["has_pool"]       = df["amenities"].fillna("").str.contains("Pool|pool", regex=True).astype(int)
    df["has_wifi"]       = df["amenities"].fillna("").str.contains("WiFi|wifi", regex=True).astype(int)
    df["has_ac"]         = df["amenities"].fillna("").str.contains("AC|ac|Air", regex=True).astype(int)
    df["has_parking"]    = df["amenities"].fillna("").str.contains("Parking|parking", regex=True).astype(int)
    # Fill nulls
    df["distance_to_beach_km"]   = df["distance_to_beach_km"].fillna(df["distance_to_beach_km"].median())
    df["distance_to_airport_km"] = df["distance_to_airport_km"].fillna(df["distance_to_airport_km"].median())
    df["peak_occupancy"] = df["peak_occupancy"].fillna(df["occupancy_rate"])
    df["off_occupancy"]  = df["off_occupancy"].fillna(df["occupancy_rate"] * 0.5)
    # Encode categoricals
    for col in ["city","area","property_type"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
    return df

def get_feature_matrix(df: pd.DataFrame):
    df = engineer_features(df)
    X = df[FEATURES].copy()
    return X, df

def investment_label(roi):
    if roi >= 10:   return "Strong Buy"
    elif roi >= 7:  return "Buy"
    elif roi >= 5:  return "Hold"
    else:           return "Avoid"

if __name__ == "__main__":
    df = pd.read_csv("data/raw/combined_listings.csv")
    df = engineer_features(df)
    df["investment_label"] = df["roi_pct"].apply(investment_label)
    df.to_csv("data/processed/listings_featured.csv", index=False)
    print("Processed:", df.shape)
    print(df["investment_label"].value_counts())
