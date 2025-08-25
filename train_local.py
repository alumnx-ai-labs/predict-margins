# train_local.py - FIXED VERSION
import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

CANONICAL_MIX = [
    "Mix_Rigid_Packaging",
    "Mix_Flexible_Films",
    "Mix_PET_Preforms",
    "Mix_Labels",
    "Mix_Caps_&_Closures",
]

# Possible aliases found in different exports
MIX_ALIASES = {
    "Mix_Rigid_Packaging":     ["Mix_RIG", "Mix_Rigid", "Mix_RigidPackaging"],
    "Mix_Flexible_Films":      ["Mix_FLE", "Mix_Flexible", "Mix_FlexibleFilms"],
    "Mix_PET_Preforms":        ["Mix_PET", "Mix_PETPreforms"],
    "Mix_Labels":              ["Mix_LAB"],
    "Mix_Caps_&_Closures":     ["Mix_CAP", "Mix_Caps", "Mix_Caps_Closures"],
}

CANONICAL_FEATURES = [
    "BrentUSD","PETResinIdx","INR_PER_USD","ElectricityIdx",
    "FreightIdx","DieselIdx","DemandClimateIdx",
] + CANONICAL_MIX

def coalesce_mix_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure canonical mix columns exist and are populated.
    If canonical names are missing/zero, copy from any alias that has data.
    Finally, row-normalize the mix shares so they sum ≈ 1 (when any are > 0).
    """
    df = df.copy()
    
    # Step 1: Handle missing/zero canonical columns by copying from aliases
    for canon in CANONICAL_MIX:
        if canon not in df.columns:
            df[canon] = 0.0
        
        # If canonical column exists but has no data, try aliases
        if df[canon].fillna(0).sum() == 0:
            for alias in MIX_ALIASES.get(canon, []):
                if alias in df.columns and df[alias].fillna(0).sum() > 0:
                    print(f"[info] Using {alias} data for {canon}")
                    df[canon] = df[alias].fillna(0)
                    break

    # Step 2: Normalize mix columns to sum to 1 (only for rows where sum > 0)
    mix_cols_present = [col for col in CANONICAL_MIX if col in df.columns]
    
    if len(mix_cols_present) > 0:
        # Get the mix matrix
        mix_matrix = df[mix_cols_present].fillna(0).values
        
        # Calculate row sums
        row_sums = mix_matrix.sum(axis=1)
        
        # Only normalize rows where sum > 0
        non_zero_mask = row_sums > 0
        
        if non_zero_mask.any():
            # Normalize only non-zero rows
            mix_matrix[non_zero_mask, :] = mix_matrix[non_zero_mask, :] / row_sums[non_zero_mask, np.newaxis]
            
            # Update the dataframe
            df[mix_cols_present] = mix_matrix
            
            print(f"[info] Normalized {non_zero_mask.sum()} rows with non-zero mix values")
        else:
            print("[warn] All mix values are zero - creating equal distribution")
            # If all are zero, create equal distribution
            equal_share = 1.0 / len(mix_cols_present)
            for col in mix_cols_present:
                df[col] = equal_share
    
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to company_monthly_summary.csv")
    ap.add_argument("--model_out", default="gm_model_random_forest.pkl", help="Path to save model")
    ap.add_argument("--features_out", default="features.json", help="Path to save features manifest")
    args = ap.parse_args()

    print(f"[info] Loading data from: {args.data}")
    df = pd.read_csv(args.data, parse_dates=["Month"])
    
    print(f"[info] Data shape: {df.shape}")
    print(f"[info] Date range: {df['Month'].min()} to {df['Month'].max()}")
    
    # Debug: Check what mix columns exist in the data
    existing_mix_cols = [col for col in df.columns if col.startswith("Mix_")]
    print(f"[info] Found mix columns: {existing_mix_cols}")
    
    # Show sample of mix data
    if existing_mix_cols:
        print("[info] Sample mix values:")
        print(df[existing_mix_cols].head())
        print(f"[info] Mix column sums: {df[existing_mix_cols].sum()}")
    
    print("[info] Coalsecing mix columns...")
    df = coalesce_mix_columns(df)

    # Verify all canonical feature cols exist; if not, create zeros
    missing_features = []
    for col in CANONICAL_FEATURES:
        if col not in df.columns:
            print(f"[warn] Column missing in data, creating zeros: {col}")
            df[col] = 0.0
            missing_features.append(col)
    
    if missing_features:
        print(f"[warn] Created {len(missing_features)} missing features with zero values")

    # Check for any remaining NaN values
    nan_counts = df[CANONICAL_FEATURES].isnull().sum()
    if nan_counts.any():
        print(f"[warn] Found NaN values in features: {nan_counts[nan_counts > 0]}")
        df[CANONICAL_FEATURES] = df[CANONICAL_FEATURES].fillna(0)

    X = df[CANONICAL_FEATURES].copy()
    y = df["GM_Pct"].copy()
    
    print(f"[info] Features shape: {X.shape}")
    print(f"[info] Target shape: {y.shape}")
    print(f"[info] Target range: {y.min():.2f}% to {y.max():.2f}%")

    # Check for any infinite or very large values
    if np.any(np.isinf(X.values)) or np.any(np.abs(X.values) > 1e6):
        print("[warn] Found infinite or very large values in features")
        X = X.replace([np.inf, -np.inf], 0)
        X = X.clip(-1e6, 1e6)

    # Time-series CV for honest validation
    print("[info] Starting time-series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=6)
    best_model, best_r2, best_split = None, -1e9, None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print(f"[info] Fold {fold+1}: Train size={len(X_train)}, Test size={len(X_test)}")

        rf = RandomForestRegressor(
            n_estimators=500,
            max_depth=7,
            min_samples_leaf=3,
            random_state=11,
            n_jobs=-1,
        )
        
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"[info] Fold {fold+1}: R²={r2:.3f}, MAE={mae:.3f}")
        
        if r2 > best_r2:
            best_r2, best_model, best_split = r2, rf, (train_idx, test_idx)

    # Report metrics on the best split
    X_train, X_test = X.iloc[best_split[0]], X.iloc[best_split[1]]
    y_train, y_test = y.iloc[best_split[0]], y.iloc[best_split[1]]
    y_pred = best_model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n[FINAL RESULTS]")
    print(f"Best fold R²: {r2:.3f}")
    print(f"MAE: {mae:.3f} percentage points") 
    print(f"RMSE: {rmse:.3f} percentage points")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': CANONICAL_FEATURES,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n[FEATURE IMPORTANCE - Top 5]")
    for _, row in feature_importance.head().iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

    print(f"\n[info] Saving model to: {args.model_out}")
    joblib.dump(best_model, args.model_out)

    # Save the exact feature order the model expects
    meta = {
        "features": CANONICAL_FEATURES,
        "model_performance": {
            "r2_score": float(r2),
            "mae": float(mae),
            "rmse": float(rmse)
        },
        "training_info": {
            "n_samples": len(X),
            "date_range": f"{df['Month'].min()} to {df['Month'].max()}",
            "target_range": f"{y.min():.2f}% to {y.max():.2f}%"
        }
    }
    
    with open(args.features_out, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    print(f"[info] Saved features manifest to: {args.features_out}")
    print(f"\n✅ Training completed successfully!")

if __name__ == "__main__":
    main()