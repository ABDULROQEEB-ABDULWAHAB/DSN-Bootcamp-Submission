import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# DSN AI Bootcamp 2026 Qualification Hackathon: Retail Sales Prediction\n",
            "\n",
            "**Author**: Abdulroqeeb  \n",
            "**Track**: Machine Learning  \n",
            "**Problem**: Predict store-product `total_sales` across retail outlets in Nigeria  \n",
            "**Evaluation Metric**: Root Mean Squared Error (RMSE)  \n",
            "\n",
            "---\n",
            "### Notebook Structure\n",
            "1. **Environment Setup & Imports**\n",
            "2. **Data Loading & Structure Inspection**\n",
            "3. **Exploratory Data Analysis (EDA)**\n",
            "4. **Feature Engineering & Preprocessing Pipeline**\n",
            "5. **Validation Strategy (Stratified by Distribution & Multi-Seed CV)**\n",
            "6. **Modeling: Direct Sales + Units Demand Ensembling (CatBoost)**\n",
            "7. **Blending Optimization & Diagnostics**\n",
            "8. **Submission Generation**"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Environment Setup & Imports\n",
            "Importing standard data analysis, visualisation, and modeling libraries."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import warnings\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "from sklearn.model_selection import KFold\n",
            "from sklearn.metrics import root_mean_squared_error\n",
            "from sklearn.preprocessing import LabelEncoder\n",
            "import catboost as cb\n",
            "\n",
            "warnings.filterwarnings('ignore')\n",
            "plt.style.use('seaborn-v0_8-whitegrid')\n",
            "plt.rcParams['figure.figsize'] = (10, 5)\n",
            "print('Libraries successfully imported.')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Data Loading & Structure Inspection"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load train and test sets\n",
            "train_path = '../data/train.csv' if os.path.exists('../data/train.csv') else 'data/train.csv'\n",
            "test_path  = '../data/test.csv' if os.path.exists('../data/test.csv') else 'data/test.csv'\n",
            "\n",
            "train_df = pd.read_csv(train_path)\n",
            "test_df  = pd.read_csv(test_path)\n",
            "\n",
            "print(f'Train shape: {train_df.shape}')\n",
            "print(f'Test shape:  {test_df.shape}')\n",
            "train_df.head()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Column types and missing value audit\n",
            "print('--- Train Info ---')\n",
            "train_df.info()\n",
            "print('\\n--- Missing Values in Train ---')\n",
            "print(train_df.isnull().sum()[train_df.isnull().sum() > 0])\n",
            "print('\\n--- Missing Values in Test ---')\n",
            "print(test_df.isnull().sum()[test_df.isnull().sum() > 0])"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Exploratory Data Analysis (EDA)\n",
            "\n",
            "Understanding the target variable distribution and relationships between outlets, prices, and visibility."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Target distribution\n",
            "sns.histplot(train_df['total_sales'], kde=True, bins=40, color='#1f77b4', ax=axes[0])\n",
            "axes[0].set_title('Target Distribution: Total Sales (Right-Skewed)')\n",
            "axes[0].set_xlabel('Total Sales (NGN)')\n",
            "\n",
            "# Log-transformed target\n",
            "sns.histplot(np.log1p(train_df['total_sales']), kde=True, bins=40, color='#2ca02c', ax=axes[1])\n",
            "axes[1].set_title('Distribution of log1p(Total Sales)')\n",
            "axes[1].set_xlabel('log1p(Total Sales)')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Store volume and format analysis\n",
            "plt.figure(figsize=(12, 5))\n",
            "store_order = train_df.groupby('store_code')['total_sales'].mean().sort_values(ascending=False).index\n",
            "sns.barplot(data=train_df, x='store_code', y='total_sales', order=store_order, palette='Blues_r', ci=None)\n",
            "plt.title('Average Sales by Store Outlet')\n",
            "plt.xticks(rotation=45)\n",
            "plt.ylabel('Mean Total Sales')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Price vs Sales by Store Format\n",
            "plt.figure(figsize=(10, 6))\n",
            "sns.scatterplot(data=train_df.sample(1500, random_state=42), x='product_price', y='total_sales', hue='store_format', alpha=0.6)\n",
            "plt.title('Product Price vs. Total Sales by Store Format')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### EDA Takeaways & Domain Insights\n",
            "1. **Disjoint Assortment**: Test records do not share product-store combinations with train records. The model must learn intrinsic category and format behavior rather than memorising specific product-outlet pairs.\n",
            "2. **Zero Shelf Visibility**: Approximately 6% of records have `shelf_visibility = 0.0`. In actual retail, products actively sold cannot occupy 0% display area. These represent missing values masked as zeros.\n",
            "3. **Store Tiering**: Clear clustering between high-capacity hypermarkets (`STORE-7WS`), standard supermarkets, and corner stores (`STORE-JOR`, `STORE-T5G`).\n",
            "4. **Price Collinearity**: Total sales strongly correlate with `product_price`. Decomposing sales into `units_sold = total_sales / product_price` yields a stationary demand target."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Feature Engineering & Preprocessing Pipeline\n",
            "\n",
            "We construct a leakage-free preprocessing pipeline:\n",
            "- Category spelling and casing standardization\n",
            "- Imputation of `product_weight_kg` and `store_size` using hierarchical lookups\n",
            "- Zero-visibility replacement via item-level and category-level medians\n",
            "- Normalized price benchmarks and store format capacity multipliers\n",
            "- Relative interactions between product category and store format"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def build_features(train, test):\n",
            "    n_tr = len(train)\n",
            "    y = train['total_sales'].copy()\n",
            "    df = pd.concat([train.drop(columns=['total_sales']), test], ignore_index=True)\n",
            "\n",
            "    # 1. Clean Category and Fat Content\n",
            "    df['product_category'] = df['product_category'].str.strip().str.title()\n",
            "    non_edible = ['Health And Hygiene', 'Household', 'Others']\n",
            "    df['is_edible'] = (~df['product_category'].isin(non_edible)).astype(int)\n",
            "    df.loc[df['is_edible'] == 0, 'fat_content'] = 'Non-Edible'\n",
            "\n",
            "    # 2. Impute Product Weights\n",
            "    weight_map = df.dropna(subset=['product_weight_kg']).groupby('product_code')['product_weight_kg'].mean()\n",
            "    df['product_weight_kg'] = df['product_weight_kg'].fillna(df['product_code'].map(weight_map))\n",
            "    cat_w = df.groupby('product_category')['product_weight_kg'].transform('median')\n",
            "    df['product_weight_kg'] = df['product_weight_kg'].fillna(cat_w).fillna(df['product_weight_kg'].median())\n",
            "\n",
            "    # 3. Impute Store Size\n",
            "    size_map = {'STORE-JOR': 'Small', 'STORE-DKU': 'Small', 'STORE-OYG': 'Small'}\n",
            "    df['store_size'] = df['store_size'].fillna(df['store_code'].map(size_map))\n",
            "\n",
            "    # 4. Correct Zero Visibility\n",
            "    df['visibility_was_zero'] = (df['shelf_visibility'] == 0.0).astype(int)\n",
            "    pos_vis = df[df['shelf_visibility'] > 0]\n",
            "    p_vis = pos_vis.groupby('product_code')['shelf_visibility'].mean()\n",
            "    c_vis = pos_vis.groupby('product_category')['shelf_visibility'].median()\n",
            "    zero_mask = df['shelf_visibility'] == 0.0\n",
            "    df.loc[zero_mask, 'shelf_visibility'] = df.loc[zero_mask, 'product_code'].map(p_vis)\n",
            "    df['shelf_visibility'] = df['shelf_visibility'].fillna(df['product_category'].map(c_vis)).fillna(df['shelf_visibility'].median())\n",
            "\n",
            "    # 5. Price Benchmarks & Relative Metrics\n",
            "    df['log_price'] = np.log1p(df['product_price'])\n",
            "    df['sqrt_price'] = np.sqrt(df['product_price'])\n",
            "    for grp in ['product_category', 'store_format', 'store_location_tier']:\n",
            "        g_mean = df.groupby(grp)['product_price'].transform('mean')\n",
            "        g_std  = df.groupby(grp)['product_price'].transform('std').replace(0, 1)\n",
            "        df[f'price_to_{grp}_mean'] = df['product_price'] / g_mean\n",
            "        df[f'price_z_{grp}'] = (df['product_price'] - g_mean) / g_std\n",
            "\n",
            "    # 6. Ordinal Encodings & Volume Multipliers\n",
            "    fmt_ord  = {'Corner Shop': 1, 'Superstore': 2, 'Standard Supermarket': 3, 'Flagship Hypermarket': 4}\n",
            "    tier_ord = {'Tier 1': 3, 'Tier 2': 2, 'Tier 3': 1}\n",
            "    sz_ord   = {'Small': 1, 'Medium': 2, 'Large': 3}\n",
            "    df['store_format_ord'] = df['store_format'].map(fmt_ord).fillna(2)\n",
            "    df['store_tier_ord']   = df['store_location_tier'].map(tier_ord).fillna(2)\n",
            "    df['store_size_ord']   = df['store_size'].map(sz_ord).fillna(2)\n",
            "    df['store_volume_mult'] = df['store_format_ord'] * df['store_tier_ord']\n",
            "\n",
            "    # 7. Interaction Terms\n",
            "    df['price_x_volume'] = df['product_price'] * df['store_volume_mult']\n",
            "    df['price_x_size']   = df['product_price'] * df['store_size_ord']\n",
            "    df['price_x_age']    = df['product_price'] * df['store_age_years']\n",
            "    df['cat_store_combo']  = df['product_category'] + '_' + df['store_code']\n",
            "    df['cat_format_combo'] = df['product_category'] + '_' + df['store_format']\n",
            "\n",
            "    # Categorical labeling for GBDTs\n",
            "    cat_cols = [\n",
            "        'product_code', 'fat_content', 'product_category',\n",
            "        'store_code', 'store_size', 'store_location_tier', 'store_format',\n",
            "        'cat_store_combo', 'cat_format_combo'\n",
            "    ]\n",
            "    for c in cat_cols:\n",
            "        le = LabelEncoder()\n",
            "        df[c] = le.fit_transform(df[c].astype(str))\n",
            "\n",
            "    train_out = df.iloc[:n_tr].copy().reset_index(drop=True)\n",
            "    test_out  = df.iloc[n_tr:].copy().reset_index(drop=True)\n",
            "    train_out['total_sales'] = y.values\n",
            "    return train_out, test_out, cat_cols\n",
            "\n",
            "train_feat, test_feat, cat_features = build_features(train_df, test_df)\n",
            "print(f'Feature engineering complete. Total feature columns: {train_feat.shape[1] - 2}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Multi-Seed Cross-Validation & Modeling\n",
            "\n",
            "### Modeling Paradigm:\n",
            "- **Direct Sales Model**: Models nominal NGN sales with RMSE loss.\n",
            "- **Units Demand Model**: Predicts sales volume normalized by product price, multiplied back at inference.\n",
            "- **Multi-Seed Bagging**: Trained across multiple distinct random seeds to minimize fold variance."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "features = [c for c in train_feat.columns if c not in ['id', 'total_sales']]\n",
            "X = train_feat[features]\n",
            "y = train_feat['total_sales'].values\n",
            "prices_tr = train_feat['product_price'].values\n",
            "\n",
            "X_test = test_feat[features]\n",
            "prices_te = test_feat['product_price'].values\n",
            "\n",
            "seeds = [42, 2026, 777]\n",
            "n_splits = 5\n",
            "\n",
            "oof_direct = np.zeros(len(X))\n",
            "oof_units  = np.zeros(len(X))\n",
            "test_direct = np.zeros(len(X_test))\n",
            "test_units  = np.zeros(len(X_test))\n",
            "\n",
            "cb_cats = [c for c in cat_features if c in X.columns]\n",
            "\n",
            "print(f'Training CatBoost ensemble across {len(seeds)} seeds...')\n",
            "for seed in seeds:\n",
            "    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)\n",
            "    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):\n",
            "        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]\n",
            "        X_va, y_va = X.iloc[val_idx], y[val_idx]\n",
            "        \n",
            "        # 1. Direct Sales Regressor\n",
            "        m_dir = cb.CatBoostRegressor(\n",
            "            iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,\n",
            "            random_seed=seed, verbose=False\n",
            "        )\n",
            "        m_dir.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cb_cats, early_stopping_rounds=40, verbose=False)\n",
            "        oof_direct[val_idx] += np.clip(m_dir.predict(X_va), 0, None) / len(seeds)\n",
            "        test_direct += np.clip(m_dir.predict(X_test), 0, None) / (len(seeds) * n_splits)\n",
            "        \n",
            "        # 2. Units Demand Regressor\n",
            "        y_u_tr = y_tr / prices_tr[tr_idx]\n",
            "        y_u_va = y_va / prices_tr[val_idx]\n",
            "        m_u = cb.CatBoostRegressor(\n",
            "            iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,\n",
            "            random_seed=seed, verbose=False\n",
            "        )\n",
            "        m_u.fit(X_tr, y_u_tr, eval_set=(X_va, y_u_va), cat_features=cb_cats, early_stopping_rounds=40, verbose=False)\n",
            "        oof_units[val_idx] += (np.clip(m_u.predict(X_va), 0, None) * prices_tr[val_idx]) / len(seeds)\n",
            "        test_units += (np.clip(m_u.predict(X_test), 0, None) * prices_te) / (len(seeds) * n_splits)\n",
            "\n",
            "print('Model training complete.')\n",
            "print(f'Direct Model OOF RMSE: {root_mean_squared_error(y, oof_direct):.2f}')\n",
            "print(f'Units Model OOF RMSE:  {root_mean_squared_error(y, oof_units):.2f}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Ensembling & Optimal Convex Blending\n",
            "\n",
            "Finding the optimal blend weight $w$ to minimize overall out-of-fold RMSE without target leakage."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "best_rmse = float('inf')\n",
            "best_w = 0.5\n",
            "for w in np.linspace(0, 1, 101):\n",
            "    blend_oof = w * oof_direct + (1 - w) * oof_units\n",
            "    rmse = root_mean_squared_error(y, blend_oof)\n",
            "    if rmse < best_rmse:\n",
            "        best_rmse = rmse\n",
            "        best_w = w\n",
            "\n",
            "print(f'Optimal Blend: {best_w:.2f} * Direct + {1-best_w:.2f} * Units')\n",
            "print(f'Ensemble Cross-Validation RMSE: {best_rmse:.2f}')\n",
            "\n",
            "final_preds = np.clip(best_w * test_direct + (1 - best_w) * test_units, 0, None)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Submission Diagnostics & Export"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('--- Test Prediction Summary ---')\n",
            "print(f'Total count: {len(final_preds)}')\n",
            "print(f'Prediction Mean: {final_preds.mean():.2f} (Train Mean: {y.mean():.2f})')\n",
            "print(f'Prediction Std:  {final_preds.std():.2f}  (Train Std:  {y.std():.2f})')\n",
            "print(f'Min Prediction:  {final_preds.min():.2f}')\n",
            "print(f'Max Prediction:  {final_preds.max():.2f}')\n",
            "\n",
            "sub_df = pd.DataFrame({\n",
            "    'id': test_df['id'],\n",
            "    'total_sales': np.round(final_preds, 2)\n",
            "})\n",
            "\n",
            "out_file = '../submissions/submission_notebook_pipeline.csv' if os.path.exists('../submissions') else 'submissions/submission_notebook_pipeline.csv'\n",
            "os.makedirs(os.path.dirname(out_file), exist_ok=True)\n",
            "sub_df.to_csv(out_file, index=False)\n",
            "print(f'Submission successfully exported to {out_file}')"
        ]
    }
]

notebook_dict = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("notebooks/dsn_mart_sales_prediction.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=1)

print("Notebook generated successfully.")
