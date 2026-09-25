#!/usr/bin/env python3
"""
Freight Rate Prediction Pipeline
--------------------------------
1. Data Cleaning & Preprocessing (handling negative weights, imputation, coordinate mapping).
2. Advanced Feature Engineering (spatial, temporal, market index interactions, rate proxies).
3. Primary Model: LightGBM Residual Regressor with L1-loss on (posted_rate - distance * quote_signal).
4. Secondary Model: Direct Rate Regressor for inputs without quote_signal (December chart inputs).
5. Generation of validation_predictions.csv and december_chart_inputs.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def clean_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    # 1. Clean negative weights (data entry anomaly where absolute value represents actual weight)
    df["weight"] = df["weight"].abs()
    
    # Missing value flags
    df["weight_was_missing"] = df["weight"].isna().astype(int)
    df["market_index_was_missing"] = df["market_index"].isna().astype(int)
    
    # Impute missing values with domain medians
    median_weight = 32000.0
    df["weight"] = df["weight"].fillna(median_weight)
    
    # For market index, if missing, fill with daily average if available, else median (1.0)
    daily_market = df.groupby("date")["market_index"].transform("mean")
    df["market_index"] = df["market_index"].fillna(daily_market).fillna(1.0)
    
    return df


def engineer_features(df: pd.DataFrame, city_coords: dict[str, tuple[float, float]] | None = None) -> pd.DataFrame:
    df = df.copy()
    
    # Impute missing coordinates if any
    if city_coords:
        if "pickup_lat" not in df.columns or df["pickup_lat"].isna().any():
            df["pickup_lat"] = df["pickup"].map(lambda c: city_coords.get(c, (35.6, -90.9))[0])
            df["pickup_lon"] = df["pickup"].map(lambda c: city_coords.get(c, (35.6, -90.9))[1])
        if "delivery_lat" not in df.columns or df["delivery_lat"].isna().any():
            df["delivery_lat"] = df["delivery"].map(lambda c: city_coords.get(c, (35.6, -90.8))[0])
            df["delivery_lon"] = df["delivery"].map(lambda c: city_coords.get(c, (35.6, -90.8))[1])

    # Haversine distance
    lat1, lon1 = np.radians(df["pickup_lat"]), np.radians(df["pickup_lon"])
    lat2, lon2 = np.radians(df["delivery_lat"]), np.radians(df["delivery_lon"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    df["haversine_dist"] = 3958.8 * 2 * np.arcsin(np.clip(np.sqrt(a), 0, 1.0))
    df["circuity_ratio"] = df["distance"] / (df["haversine_dist"] + 1.0)
    
    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    # Domain interactions
    df["weight_per_mile"] = df["weight"] / (df["distance"] + 1.0)
    
    if "quote_signal" in df.columns:
        df["implied_rate"] = df["distance"] * df["quote_signal"]
        df["market_quote_mult"] = df["market_index"] * df["quote_signal"]
        df["implied_market_rate"] = df["distance"] * df["market_quote_mult"]

    # Categorical types
    for col in ["equipment", "pickup", "delivery"]:
        df[col] = df[col].astype("category")
        
    return df


def main():
    print("=" * 60)
    print("SPOTTER FREIGHT RATE PREDICTION PIPELINE")
    print("=" * 60)
    
    data_dir = Path("data")
    train_path = data_dir / "train_test.csv"
    val_path = data_dir / "validation.csv"
    dec_path = data_dir / "december_chart_inputs.csv"
    
    if not train_path.exists() or not val_path.exists():
        print("ERROR: Training or validation data not found in data/.")
        sys.exit(1)
        
    train_raw = pd.read_csv(train_path)
    val_raw = pd.read_csv(val_path)
    dec_raw = pd.read_csv(dec_path)
    
    # Map all city coordinates
    city_coords = {}
    for df in [train_raw, val_raw]:
        for _, r in df.iterrows():
            if r["pickup"] not in city_coords:
                city_coords[r["pickup"]] = (r["pickup_lat"], r["pickup_lon"])
            if r["delivery"] not in city_coords:
                city_coords[r["delivery"]] = (r["delivery_lat"], r["delivery_lon"])
                
    # Clean data
    train = clean_data(train_raw, is_train=True)
    val = clean_data(val_raw, is_train=False)
    
    # Extract December daily market index curve from validation data
    val_dec = val[val["date"].between("2025-12-01", "2025-12-31")]
    dec_daily_market = val_dec.groupby("date")["market_index"].mean().to_dict()
    
    # Clean December chart inputs
    dec = dec_raw.copy()
    dec["date"] = pd.to_datetime(dec["date"])
    dec["market_index"] = dec["date"].map(dec_daily_market)
    dec["weight_was_missing"] = 0
    dec["market_index_was_missing"] = 0
    
    # Feature Engineering
    train_feat = engineer_features(train, city_coords)
    val_feat = engineer_features(val, city_coords)
    dec_feat = engineer_features(dec, city_coords)
    
    # Target Residual
    train_feat["residual"] = train_feat["posted_rate"] - train_feat["implied_rate"]
    
    # Define Feature Sets
    features_with_quote = [
        "distance", "weight", "market_index", "quote_signal",
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
        "haversine_dist", "circuity_ratio",
        "day_of_week", "day_of_month", "month", "is_weekend",
        "weight_per_mile", "market_quote_mult", "implied_market_rate",
        "weight_was_missing", "market_index_was_missing",
        "equipment", "pickup", "delivery"
    ]
    
    features_no_quote = [
        "distance", "weight", "market_index",
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
        "haversine_dist", "circuity_ratio",
        "day_of_week", "day_of_month", "month", "is_weekend",
        "weight_per_mile",
        "weight_was_missing", "market_index_was_missing",
        "equipment", "pickup", "delivery"
    ]
    
    print("\n1. Running Out-of-Time Cross Validation (Jan-Aug train -> Sep-Oct test)...")
    split_date = "2025-09-01"
    tr = train_feat[train_feat["date"] < split_date]
    te = train_feat[train_feat["date"] >= split_date]
    
    val_model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    val_model.fit(tr[features_with_quote], tr["residual"])
    te_pred_res = val_model.predict(te[features_with_quote])
    te_pred_rate = te["implied_rate"] + te_pred_res
    
    baseline_mae = mean_absolute_error(te["posted_rate"], te["implied_rate"])
    model_mae = mean_absolute_error(te["posted_rate"], te_pred_rate)
    model_rmse = np.sqrt(mean_squared_error(te["posted_rate"], te_pred_rate))
    model_r2 = r2_score(te["posted_rate"], te_pred_rate)
    
    print(f"   Baseline Implied Rate MAE: ${baseline_mae:.2f}")
    print(f"   Model Residual LGBM MAE:   ${model_mae:.2f} ({(baseline_mae - model_mae)/baseline_mae*100:.1f}% improvement)")
    print(f"   Model RMSE:               ${model_rmse:.2f}")
    print(f"   Model R2:                 {model_r2:.4f}")
    
    print("\n2. Training Final Primary Model on all 48,000 loads...")
    final_primary_model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=900,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    final_primary_model.fit(train_feat[features_with_quote], train_feat["residual"])
    
    print("\n3. Generating predictions for validation.csv (12,000 loads)...")
    val_pred_res = final_primary_model.predict(val_feat[features_with_quote])
    val_pred_rate = np.maximum(50.0, val_feat["implied_rate"] + val_pred_res)
    
    validation_predictions = pd.DataFrame({
        "load_id": val_feat["load_id"],
        "predicted_rate": np.round(val_pred_rate, 2)
    })
    
    val_out_path = Path("validation_predictions.csv")
    validation_predictions.to_csv(val_out_path, index=False)
    print(f"   Saved {len(validation_predictions):,} predictions to {val_out_path}")
    
    print("\n4. Training Secondary Model for December Chart Inputs (No quote_signal)...")
    final_secondary_model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    final_secondary_model.fit(train_feat[features_no_quote], train_feat["posted_rate"])
    
    dec_preds = final_secondary_model.predict(dec_feat[features_no_quote])
    dec_raw["predicted_rate"] = np.round(dec_preds, 2)
    
    dec_out_path = data_dir / "december_chart_inputs.csv"
    dec_raw.to_csv(dec_out_path, index=False)
    print(f"   Saved 31 December predictions to {dec_out_path}")
    
    print("\n5. Running official score.py verification...")
    import subprocess
    cmd = [
        sys.executable,
        "score.py",
        "--predictions", "validation_predictions.csv",
        "--december-predictions", "data/december_chart_inputs.csv"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("ERROR running score.py:", res.stderr)
        sys.exit(1)
    
    print("\nAll deliverables generated and verified successfully!")


if __name__ == "__main__":
    main()
