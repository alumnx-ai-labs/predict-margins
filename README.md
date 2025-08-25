# Predict Commodity Price Impact on Sales Margins 🎯

## AI-powered real-time gross margin predictor for packaging manufacturing

> Transform commodity price volatility from business risk into competitive advantage through instant margin forecasting.

## 🚀 What It Does

THis project predicts your company's **gross margin percentage** in real-time based on:

- **Commodity prices** (Brent crude, PET resin, diesel, electricity)
- **Currency fluctuations** (USD/INR exchange rates)
- **Market conditions** (freight costs, demand climate)  
- **Product portfolio mix** (rigid packaging, flexible films, PET preforms, labels, caps & closures)

## 🛠️ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_local.py --data company_monthly_summary.csv
```

### 3. Test Predictions

```bash
python predict_local.py \
  --model gm_model_random_forest.pkl \
  --features features.json \
  --input_json '{"BrentUSD":85,"PETResinIdx":120,"INR_PER_USD":83,"ElectricityIdx":135,"FreightIdx":130,"DieselIdx":125,"DemandClimateIdx":105,"Mix_Rigid_Packaging":0.2,"Mix_Flexible_Films":0.2,"Mix_PET_Preforms":0.2,"Mix_Labels":0.2,"Mix_Caps_&_Closures":0.2}' \
  --shock 2
```

### 4. Launch Interactive Demo

```bash
streamlit run app.py
```

## 📁 Repository Structure

```text
predict-margin/
├── train_local.py              # Model training script
├── predict_local.py            # Command-line prediction tool
├── app.py                      # Streamlit investor demo
├── company_monthly_summary.csv # Historical business data (248 months)
├── sku_monthly_panel.csv       # Detailed SKU-level transactions
├── macro_indices.csv           # Commodity price indices
├── gm_model_random_forest.pkl  # Trained model (generated)
├── features.json               # Model metadata (generated)
└── README.md                   # This file
```

## 🎪 Demo Features

### Interactive Sliders

- **Commodity prices**: Brent crude, PET resin, electricity, freight, diesel
- **Currency rate**: USD/INR exchange rate
- **Product mix**: 5 category proportions (auto-normalized)

### Real-Time Predictions  

- **Main display**: Current margin prediction with confidence
- **Change indicator**: Impact vs baseline scenario
- **Quick scenarios**: One-click crisis simulations

### Historical Validation

- **12-month overlay**: Actual vs predicted margins
- **Accuracy metrics**: R², MAE, error distribution
- **Confidence bands**: Prediction uncertainty visualization

## 🧮 Model Details

### Algorithm

- **Random Forest Regressor** (500 trees, max depth 7)
- **Time-series validation** (prevents data leakage)
- **Feature importance** analysis included

### Key Features (12 total)

**Macro Drivers (7)**:

- `BrentUSD` - Brent crude oil price ($/barrel)
- `PETResinIdx` - PET resin price index (base 100)
- `INR_PER_USD` - Exchange rate (₹ per $1)
- `ElectricityIdx` - Power cost index
- `FreightIdx` - Shipping cost proxy (Baltic Dry Index)
- `DieselIdx` - Diesel fuel cost index
- `DemandClimateIdx` - Seasonal demand factor

**Product Mix (5)**:

- `Mix_Rigid_Packaging` - Share of rigid container sales
- `Mix_Flexible_Films` - Share of flexible packaging sales  
- `Mix_PET_Preforms` - Share of bottle preform sales
- `Mix_Labels` - Share of label/sticker sales
- `Mix_Caps_&_Closures` - Share of cap/closure sales

### Target Variable

- `GM_Pct` - Gross Margin Percentage (Net Sales - COGS) / Net Sales × 100

## 🔧 Command Line Usage

### Training

```bash
python train_local.py \
  --data company_monthly_summary.csv \
  --model_out gm_model_random_forest.pkl \
  --features_out features.json
```

### Prediction (Historical Month)

```bash
python predict_local.py \
  --data company_monthly_summary.csv \
  --model gm_model_random_forest.pkl \
  --features features.json \
  --month "2024-06-01" \
  --shock 5.0
```

### Prediction (Manual Input)

```bash
python predict_local.py \
  --model gm_model_random_forest.pkl \
  --features features.json \
  --input_json '{"BrentUSD":90,"PETResinIdx":125,...}' \
  --shock 2.0
```

## 📊 Model Performance

- **Accuracy**: 80.4% (R² = 0.804)
- **Prediction Error**: ±0.82 percentage points (MAE)
- **Training Data**: 248 months (2005-2025)
- **Validation**: Time-series cross-validation (6 folds)

**Key Finding**: PET Resin Index drives 76% of margin sensitivity (not crude oil directly).

## 🎯 Business Value

### Immediate ROI

- **Eliminate manual forecasting**: Save 2-3 FTEs ($200K+ annually)
- **Faster decision making**: Weeks → Seconds
- **Better hedging decisions**: $500K-2M annual risk mitigation
- **Margin optimization**: 1-2% improvement = $2-4M annually

### Strategic Advantages

- **Proactive pricing** based on cost forecasts
- **Real-time scenario planning** for board meetings  
- **Data-driven contract negotiations**
- **Early warning system** for margin compression
