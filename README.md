# Spotter Freight Rate Prediction Assessment

Production-grade Machine Learning solution for the Spotter Freight Rate Prediction Challenge.

---

## 1. Executive Summary & Architecture

The objective is to accurately predict the `posted_rate` ($) for freight loads given logistical, geographic, equipment, and market conditions:
- **Development Data (`data/train_test.csv`)**: 48,000 historic loads spanning January 1, 2025 to October 31, 2025.
- **Validation Data (`data/validation.csv`)**: 12,000 loads spanning November 1, 2025 to December 31, 2025.
- **December Chart Inputs (`data/december_chart_inputs.csv`)**: 31 daily loads for a fixed lane (Lexington -> Fort Wayne, 360 miles, Dry Van, 32,000 lbs) throughout December 2025.

### Model Formulation: Residual Gradient Boosting
Exploratory data analysis revealed that `distance * quote_signal` serves as a baseline market expectation (MAE = \$232.26). However, supply-demand surges, equipment premiums (Reefer and Flatbed vs. Dry Van), route circuity, and macro-market volatility introduce substantial deviations. 

Instead of predicting total rate directly (where large distances dominate the loss gradient), the primary model predicts the **Rate Residual**:

```
residual = posted_rate - (distance * quote_signal)
```

The final prediction is reconstructed as:

```
predicted_rate = max(50.0, distance * quote_signal + predicted_residual)
```

Trained with an **L1 (MAE) objective**, this formulation achieved a **53.2% error reduction** over the baseline in out-of-time evaluation, reducing MAE from **\$246.02** to **\$115.02**.

---

## 2. Key Exploratory Findings & Data Quality Handling

1. **Negative Weight Artifacts**:
   - Exactly 292 rows in train and 145 rows in validation contained negative weights (e.g. `-36559.0` lbs).
   - Analysis proved these were sign-flip data-entry artifacts: taking the absolute value `abs(weight)` aligns them with realistic trailer capacities (10,000 – 47,500 lbs).
2. **Missing Value Imputation**:
   - `weight` was missing in 300 train rows and 165 validation rows; imputed with domain median (32,000 lbs) alongside an indicator flag `weight_was_missing`.
   - `market_index` was missing in 374 train rows and 249 validation rows; imputed with daily market cross-sectional means, falling back to 1.0.
3. **Equipment Discrepancies**:
   - Dry Van exhibits near 1.0 ratio with quote signal baseline year-round.
   - Reefer and Flatbed experience massive seasonal premiums (+25% to +35%) during produce/construction peak seasons (April–May, July, October).
4. **Spatial Circuity**:
   - Comparing actual road distance with Haversine great-circle distance revealed route circuity (average ratio ~1.19), capturing geographical detours and mountain passes.

---

## 3. Validation Strategy

We implemented a strict **Out-of-Time (Temporal) Validation** to prevent look-ahead bias and mirror production conditions:
- **Train Window**: January 1, 2025 – August 31, 2025 (38,477 loads)
- **Validation Window**: September 1, 2025 – October 31, 2025 (9,523 loads)

| Strategy / Model | Validation MAE ($) | RMSE ($) | R² |
| :--- | :---: | :---: | :---: |
| Naive Quote Baseline (`distance * quote_signal`) | \$246.02 | \$674.57 | 0.801 |
| Direct LightGBM (`posted_rate`) | \$166.07 | \$652.10 | 0.814 |
| Rate-Per-Mile LightGBM (`rpm * distance`) | \$155.05 | \$648.30 | 0.819 |
| **Residual LightGBM (L1 Loss) [Selected]** | **\$115.02** | **\$635.66** | **0.826** |

---

## 4. Setup & Reproducibility

### Prerequisites
- Python 3.10+ (macOS, Linux, or Windows)
- OpenMP runtime (`brew install libomp` on macOS if compiling native packages)

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Execution
Run the end-to-end pipeline:
```bash
python solution.py
```
This script will:
1. Clean and engineer all spatial/temporal features.
2. Run out-of-time validation.
3. Train the primary model on all 48,000 training loads.
4. Generate `validation_predictions.csv` (12,000 predictions).
5. Train the secondary model and generate `data/december_chart_inputs.csv` (31 predictions).
6. Automatically run `score.py` to validate outputs and generate `scorer_results/candidate_december.png`.

---

## 5. Deliverables Verification
- `validation_predictions.csv`: 12,000 rows (`load_id,predicted_rate`), verified by `score.py`.
- `data/december_chart_inputs.csv`: 31 daily rows filled with realistic rate predictions.
- `scorer_results/candidate_december.png`: Generated chart reflecting intra-week seasonality and end-of-year market dynamics.
- `REPORT.md` / `report.pdf` / `report.docx`: Comprehensive documentation of methodology, model design trade-offs, and production considerations.
