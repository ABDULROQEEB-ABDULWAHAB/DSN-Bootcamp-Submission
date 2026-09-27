# DSN AI Bootcamp Qualification Hackathon: Retail Sales Prediction

This repository contains the complete end-to-end machine learning solution for the **Data Science Nigeria (DSN) AI Bootcamp Qualification Hackathon (2026)**.

The objective is to predict `total_sales` for retail store-product combinations across diverse outlet formats in Nigeria, evaluated on Root Mean Squared Error (RMSE).

---

## Key Solution Highlights

1. **Assortment Distribution Discovery**:
   - Diagnostic checks revealed that all test rows are *unseen product-store pairings*.
   - Applying empirical store-level multiplier calibrations introduced distribution shift; relying directly on structural feature representations (price ratios, format volume indices, store location tier) yielded superior generalization.
2. **Dual-Objective Ensembling**:
   - **Direct Nominal Sales Regressor**: Minimizes RMSE directly on `total_sales`.
   - **Units Demand Regressor**: Models volume `units = total_sales / product_price`, recovering sales at inference via multiplication with product price.
3. **Multi-Seed Bagging**:
   - 5 distinct random seeds across 5-fold cross-validation (50 total CatBoost fits), smoothed via convex blend optimization.
   - Out-of-fold RMSE: **1071.73**.

---

## Repository Structure

```
├── data/                               # Dataset directory (train.csv, test.csv)
├── notebooks/
│   └── dsn_mart_sales_prediction.ipynb # Complete end-to-end walk-through notebook
├── src/
│   ├── features.py                     # Feature engineering and preprocessing logic
│   └── utils.py                        # Helper functions and metric routines
├── scripts/
│   ├── train_clean_best.py             # Script generating top-performing submission
│   └── generate_notebook.py            # Reproducible notebook generator
├── submissions/                        # Generated prediction outputs
│   └── latest_submission.csv           # Top competitive submission file
├── requirements.txt                    # Project python dependencies
└── README.md                           # Documentation
```

---

## Getting Started

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/<your-username>/dsn-bootcamp-retail-sales.git
cd dsn-bootcamp-retail-sales
pip install -r requirements.txt
```

### 2. Running the Complete Notebook

Open the interactive notebook in Jupyter:

```bash
jupyter notebook notebooks/dsn_mart_sales_prediction.ipynb
```

The notebook covers:
- Exploratory Data Analysis & visualising zero-visibility anomalies
- Deterministic imputation strategies
- Relative pricing benchmarks & store hierarchy features
- Cross-validation routines and multi-objective model training
- Generation of the submission file

### 3. Reproducing Top Submission via CLI

To train the full 5-seed bagged ensemble directly from the command line:

```bash
python scripts/train_clean_best.py
```

The final predictions will be automatically verified for diagnostics and saved to `submissions/latest_submission.csv`.

---

## Methodology Summary

| Component | Approach |
| :--- | :--- |
| **Missing Values** | Item-level mean lookup for `product_weight_kg` and `shelf_visibility`; domain-mapped size for known outlets. |
| **Categoricals** | Target encoding & label encoding handling high-cardinality categories. |
| **Feature Engineering** | Price-to-group z-scores, outlet capacity multipliers, product presence counts, category-format interaction features. |
| **Models** | Multi-seed CatBoost Regressors with early stopping on validation folds. |
| **Validation** | 5-Fold Cross Validation repeated across 5 seeds (25 folds per objective). |
