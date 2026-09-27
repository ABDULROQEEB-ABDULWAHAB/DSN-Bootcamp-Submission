# Technical Report: DSN Mart Sales Prediction Hackathon
## DSN AI Bootcamp 2026 Qualification Assessment — Machine Learning Track
**Author:** Abdulroqeeb Abdulwahab  
**Date:** September 2026  
**Metric:** Root Mean Squared Error (RMSE)  

---

## 1. UNDERSTAND: Business Context & Problem Formulation

### 1.1 The Operational Context
DSN Mart operates a diversified retail footprint across Nigeria spanning multiple store formats:
- **Flagship Hypermarkets** (large-scale destinations in Tier 3 commercial centers)
- **Superstores**
- **Standard Supermarkets** (high frequency, urban centers in Tier 1 and state capitals in Tier 2)
- **Corner Shops** (neighborhood convenience stores in Tier 1 and Tier 3)

In retail supply chains, stockouts lead to lost revenue while overstocking leads to working capital lockup and perishability losses. Accurate product-level sales forecasting allows DSN Mart to optimize replenishment schedules, tailor store-format inventory, and make data-backed capital allocation decisions.

### 1.2 Mathematical Formulation
The objective is to predict $y_i = \text{total\_sales}$ for each item $i$ in test data across 10 distinct retail outlets.
The evaluation metric is Root Mean Squared Error (RMSE):
$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

Because RMSE penalizes large errors quadratically, accurate estimation of high-velocity, high-price products is disproportionately important.

---

## 2. ANALYSE: Exploratory Data Analysis & Empirical Discoveries

### 2.1 The Physical Relationship of Retail Demand
In retail accounting:
$$\text{Total Sales} = \text{Units Sold} \times \text{Product Price}$$

Our empirical analysis revealed:
1. **Product Price Distribution**: `product_price` exhibits a correlation of $r = 0.57$ with `total_sales`.
2. **Unit Demand Stability**: Decomposing sales into $\text{Units Sold} = \frac{\text{total\_sales}}{\text{product\_price}}$ reveals that unit demand has a mean of 15.44 and median of 14.88 across the dataset.
3. **Format Disparity**:
   - **Corner Shops** (`STORE-T5G`, `STORE-JOR`): Mean units sold is **2.42 units** (Mean sales: ~336).
   - **Standard Supermarkets**: Mean units sold is **16.39 units** (Mean sales: ~2,316).
   - **Superstores**: Mean units sold is **13.85 units** (Mean sales: ~1,972).
   - **Flagship Hypermarkets** (`STORE-7WS`): Mean units sold is **26.42 units** (Mean sales: ~3,660).

### 2.2 Data Quality & Domain Recovery
1. **Cross-Store Product Weight Recovery**:
   - `product_weight_kg` had 1,225 missing entries in train and 306 in test.
   - Because a physical product's weight is an intrinsic invariant across outlets, constructing a cross-store lookup dictionary on `product_code` resolved **1,527 of 1,531 missing values** (99.7% recovery). The remaining 4 cases were imputed using the category-level median.
2. **Store Size Imputation**:
   - 3 outlets lacked `store_size` (`STORE-DKU`, `STORE-OYG`, `STORE-JOR`).
   - Matching against identical format and tier peers revealed that `STORE-JOR` (Tier 3 Corner Shop) mirrors `STORE-T5G` (`Small`), while `STORE-DKU` and `STORE-OYG` (Tier 2 Standard Supermarkets) mirror `STORE-9RG` (`Small`).
3. **Zero Visibility Correction**:
   - 526 rows contained `shelf_visibility == 0.0`. Physically, displayed inventory occupying shelf space cannot have zero visual presence. Zeros were replaced with product-specific or category-median visibility.
4. **Text Cleaning**:
   - 48 inconsistent casing variations in `product_category` were normalized into 16 canonical retail categories.

---

## 3. MODEL: Architecture & Engineering Strategy

### 3.1 Feature Engineering Pipeline
- **Price Interactions**: `log_price`, price ratios relative to category and store averages, and price-visibility interactions (`product_price * shelf_visibility`).
- **Store Profile Representations**: Combined categorical features encoding `store_format + store_size + store_location_tier`.
- **Visibility Ratios**: Normalizing visibility by category average to identify premium vs under-displayed placements.

### 3.2 Heterogeneous Multi-Model Ensembling
To capture distinct functional relationships without overfitting, three distinct gradient-boosted tree families and a regularized linear model were trained:
1. **LightGBM**: Leaf-wise tree growth with histogram-based splitting, optimal for tabular interactions.
2. **XGBoost**: Depth-wise tree growth with L1/L2 leaf weight regularization.
3. **CatBoost**: Symmetric oblivious trees with native categorical combinations.
4. **Ridge Regression**: L2-regularized linear baseline providing architectural diversity.

### 3.3 Dual-Objective Formulation: Direct Sales vs Unit Demand
Each tree family was trained under two complementary formulations:
1. **Direct Regression**: Directly predicting $y = \text{total\_sales}$ minimizing squared error.
2. **Unit Demand Regression**: Predicting $u = \frac{\text{total\_sales}}{\text{product\_price}}$, then reconstructing sales $\hat{y} = \hat{u} \times \text{product\_price}$.
This dual formulation stabilized predictions across price extremes.

---

## 4. PREDICT: Validation Framework & Optimization

### 4.1 5-Fold Cross-Validation Framework
All models were evaluated using a leak-free 5-Fold cross-validation protocol. All fold transformations were strictly out-of-fold to prevent data leakage and guarantee that local CV scores closely match the Kaggle leaderboard.

### 4.2 Convex Ensemble Optimization (SciPy SLSQP)
Rather than simple unweighted averaging, out-of-fold predictions were stacked and solved via Sequential Least Squares Programming (SLSQP):
$$\min_{\mathbf{w}} \sqrt{\frac{1}{N}\sum_{i=1}^{N} \left(y_i - \sum_{m=1}^{M} w_m \hat{y}_{i,m}\right)^2} \quad \text{s.t.} \quad w_m \ge 0, \quad \sum_{m=1}^{M} w_m = 1$$

### 4.3 Post-Processing Guardrails
- **Non-negativity bounds**: Retail sales cannot be negative; predictions clipped at 0.0.
- **Distribution integrity**: Verified that prediction mean and standard deviation match empirical retail distributions.

---

## 5. COMMUNICATE: Business Insights & Strategic Recommendations

1. **Format-Driven Store Allocation**:
   - Flagship Hypermarkets generate 10.8x the revenue per product of Corner Shops. Inventory holding policies should prioritize high-variety assortments in Hypermarkets and essential high-turnover staples in Corner Shops.
2. **Price Sensitivity vs Visibility**:
   - Shelf visibility has diminishing returns for commodities, but disproportionately lifts sales for premium tiers ($>150$ price index).
3. **Reproducibility & Code Standards**:
   - The solution is completely modularized under `src/`, with clean separation of data loading, feature engineering, model training, and submission verification.
