# predict_local.py
import argparse
import json
import numpy as np
import pandas as pd
import joblib

from train_local import coalesce_mix_columns, CANONICAL_FEATURES

def apply_crude_shock(xrow: pd.Series, shock_pct: float) -> pd.Series:
    """Apply a crude shock and pass-through to related indices (demo values)."""
    x = xrow.copy()
    x["BrentUSD"]    = x["BrentUSD"]   * (1 + shock_pct/100.0)
    x["PETResinIdx"] = x["PETResinIdx"]* (1 + 0.6*shock_pct/100.0)   # demo pass-through
    x["DieselIdx"]   = x["DieselIdx"]  * (1 + 0.7*shock_pct/100.0)   # demo pass-through
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Path to company_monthly_summary.csv (required for --month mode)")
    ap.add_argument("--model", required=True, help="Path to gm_model_random_forest.pkl")
    ap.add_argument("--features", required=True, help="Path to features.json saved at training")
    ap.add_argument("--month", help="YYYY-MM-01 style month to pull from CSV")
    ap.add_argument("--shock", type=float, default=0.0, help="Crude shock in percent (e.g., 2.0 for +2%)")
    ap.add_argument("--input_json", help="JSON string with all features (alternative to --month)")
    args = ap.parse_args()

    model = joblib.load(args.model)
    feats = json.loads(open(args.features).read())["features"]

    if args.input_json:
        # Manual input mode
        sample = json.loads(args.input_json)
        # Make sure all features are present
        for f in feats:
            if f not in sample:
                sample[f] = 0.0
        x_base = pd.Series(sample)[feats]
    else:
        if not args.data or not args.month:
            raise SystemExit("For --month mode, provide --data and --month.")
        df = pd.read_csv(args.data, parse_dates=["Month"])
        df = coalesce_mix_columns(df)
        row = df[df["Month"] == pd.to_datetime(args.month)]
        if row.empty:
            raise SystemExit(f"Month {args.month} not found in data.")
        x_base = row.iloc[0][feats]

    x_shock = apply_crude_shock(x_base, args.shock) if args.shock != 0 else x_base

    base_pred  = float(model.predict(pd.DataFrame([x_base], columns=feats))[0])
    shock_pred = float(model.predict(pd.DataFrame([x_shock], columns=feats))[0])

    print(f"Baseline GM%: {base_pred:.2f}")
    if args.shock != 0:
        print(f"Shocked GM% (+{args.shock:.2f}% crude): {shock_pred:.2f}")
        print(f"Delta (pp): {shock_pred - base_pred:.2f}")

if __name__ == "__main__":
    main()
