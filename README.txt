
Packaging AI Demo Dataset & Model
=================================

Files
-----
- sku_monthly_panel.csv : SKU-Region-Customer monthly transactions with prices, costs, and margins (2018-01 to 2025-07).
- company_monthly_summary.csv : Aggregated P&L and mixes by month with macro indices.
- macro_indices.csv : Crude, PET resin, FX, Electricity, Freight, Diesel, and Demand climate indices by month.
- gm_model_random_forest.pkl : Trained RandomForestRegressor that predicts company-level Gross Margin % from drivers.
- app.py : Streamlit demo app to run what-if scenarios (e.g., +2% crude).

Model (company-level GM%)
-------------------------
- Algorithm: RandomForestRegressor
- Features: ['CrudeIdx', 'PETResinIdx', 'FX_INRUSD', 'ElectricityIdx', 'FreightIdx', 'DieselIdx', 'DemandClimateIdx', 'Mix_Rigid_Packaging', 'Mix_Flexible_Films', 'Mix_PET_Preforms', 'Mix_Labels', 'Mix_Caps_&_Closures']
- Test R²: 0.649
- Test MAE: 0.401
- Test RMSE: 0.503

Quick Start
-----------
1) Install:
   pip install streamlit scikit-learn pandas numpy joblib

2) Run the demo:
   streamlit run app.py

3) Use the slider to apply a crude index shock (e.g., +2%), then see predicted GM% vs baseline.

Notes
-----
- Data are synthetic but constructed with realistic relationships:
  PET Resin index lags Crude/Naphtha by ~2 months, FX impacts import-linked costs,
  Electricity & Freight affect production/logistics, and category mix shifts seasonally.
- The panel can be used to build SKU-level models, demand forecasting, pricing optimization, etc.
