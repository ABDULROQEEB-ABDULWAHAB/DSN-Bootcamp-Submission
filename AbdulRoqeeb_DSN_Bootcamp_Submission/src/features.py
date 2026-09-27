"""
Feature Engineering and Data Preparation Pipeline for DSN Mart Sales Prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_raw_data(train_path='data/train.csv', test_path='data/test.csv'):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def clean_and_impute_data(train_df, test_df):
    """
    Performs deterministic domain cleaning and imputation on train and test sets.
    """
    n_train = len(train_df)
    target = train_df['total_sales'].copy()
    
    # Concatenate features only to compute global physical invariants (e.g. product weight, store size)
    combined = pd.concat([train_df.drop(columns=['total_sales'], errors='ignore'), test_df], ignore_index=True)
    
    # 1. Product category standardization (fix messy capitalization)
    combined['product_category'] = combined['product_category'].astype(str).str.strip().str.title()
    
    # 2. Product weight imputation:
    # A product has a fixed physical weight across stores. First map by product_code, then category median.
    weight_lookup = combined.dropna(subset=['product_weight_kg']).groupby('product_code')['product_weight_kg'].mean()
    combined['product_weight_kg'] = combined['product_weight_kg'].fillna(combined['product_code'].map(weight_lookup))
    cat_weight_lookup = combined.groupby('product_category')['product_weight_kg'].transform('median')
    combined['product_weight_kg'] = combined['product_weight_kg'].fillna(cat_weight_lookup)
    
    # 3. Store size imputation:
    # STORE-JOR is Corner Shop in Tier_3 -> Small (matches STORE-T5G)
    # STORE-DKU is Standard Supermarket in Tier_2 -> Small (matches STORE-9RG)
    # STORE-OYG is Standard Supermarket in Tier_2 -> Small (matches STORE-9RG)
    store_size_impute = {
        'STORE-JOR': 'Small',
        'STORE-DKU': 'Small',
        'STORE-OYG': 'Small'
    }
    combined['store_size'] = combined['store_size'].fillna(combined['store_code'].map(store_size_impute))
    
    # 4. Zero visibility correction:
    # An item displayed in store cannot have 0.0 visibility. Replace with product average or category median.
    combined['visibility_was_zero'] = (combined['shelf_visibility'] == 0.0).astype(int)
    pos_vis = combined[combined['shelf_visibility'] > 0]
    prod_vis_lookup = pos_vis.groupby('product_code')['shelf_visibility'].mean()
    cat_vis_lookup = pos_vis.groupby('product_category')['shelf_visibility'].median()
    
    zero_mask = combined['shelf_visibility'] == 0.0
    combined.loc[zero_mask, 'shelf_visibility'] = combined.loc[zero_mask, 'product_code'].map(prod_vis_lookup)
    combined['shelf_visibility'] = combined['shelf_visibility'].fillna(combined['product_category'].map(cat_vis_lookup))
    
    # 5. Fat content standardization
    # For non-edibles, fat content is not strictly applicable, but standardize to clean strings
    non_edible_cats = ['Health And Hygiene', 'Household', 'Others']
    combined['is_edible'] = (~combined['product_category'].isin(non_edible_cats)).astype(int)
    combined.loc[combined['is_edible'] == 0, 'fat_content'] = 'Non-Edible'
    
    return combined, target, n_train

def engineer_features(combined):
    """
    Engineers rich interaction, ratio, and aggregation features.
    """
    df = combined.copy()
    
    # --- Price Interactions ---
    df['log_price'] = np.log1p(df['product_price'])
    
    # Price relative to Category
    cat_price_mean = df.groupby('product_category')['product_price'].transform('mean')
    cat_price_std = df.groupby('product_category')['product_price'].transform('std').replace(0, 1)
    df['price_to_cat_mean'] = df['product_price'] / cat_price_mean
    df['price_diff_cat_mean'] = df['product_price'] - cat_price_mean
    df['price_cat_zscore'] = (df['product_price'] - cat_price_mean) / cat_price_std
    
    # Price relative to Store
    store_price_mean = df.groupby('store_code')['product_price'].transform('mean')
    df['price_to_store_mean'] = df['product_price'] / store_price_mean
    
    # Price relative to Store Format
    format_price_mean = df.groupby('store_format')['product_price'].transform('mean')
    df['price_to_format_mean'] = df['product_price'] / format_price_mean
    
    # Price Bins (retail price tiers)
    df['price_tier'] = pd.qcut(df['product_price'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    # --- Shelf Visibility Interactions ---
    cat_vis_mean = df.groupby('product_category')['shelf_visibility'].transform('mean')
    store_vis_mean = df.groupby('store_code')['shelf_visibility'].transform('mean')
    df['vis_to_cat_mean'] = df['shelf_visibility'] / cat_vis_mean
    df['vis_to_store_mean'] = df['shelf_visibility'] / store_vis_mean
    df['price_times_visibility'] = df['product_price'] * df['shelf_visibility']
    
    # --- Product Code Features ---
    # Extract high-level product family from product_code
    df['product_family'] = df['product_code'].str[:6]
    df['product_code_freq'] = df.groupby('product_code')['id'].transform('count')
    
    # --- Store Dynamics ---
    df['store_age_sq'] = df['store_age_years'] ** 2
    df['store_age_bucket'] = pd.cut(df['store_age_years'], bins=[0, 25, 35, 45, 100], labels=[1, 2, 3, 4]).astype(int)
    
    # Ordinal mappings for Store Size & Tier
    size_map = {'Small': 1, 'Medium': 2, 'Large': 3}
    tier_map = {'Tier_1': 1, 'Tier_2': 2, 'Tier_3': 3}
    df['store_size_ord'] = df['store_size'].map(size_map).astype(int)
    df['store_tier_ord'] = df['store_location_tier'].map(tier_map).astype(int)
    
    # Combined store profile identifier
    df['store_profile'] = df['store_format'] + '_' + df['store_size'] + '_' + df['store_location_tier']
    
    # --- Item-Level Pricing & Merchandising Interactions ---
    item_mean_vis = df.groupby('product_code')['shelf_visibility'].transform('mean')
    df['item_vis_ratio'] = df['shelf_visibility'] / item_mean_vis.replace(0, 0.01)
    
    item_mean_price = df.groupby('product_code')['product_price'].transform('mean')
    df['item_price_diff'] = df['product_price'] - item_mean_price
    df['item_price_ratio'] = df['product_price'] / item_mean_price
    
    # Classic MRP 4-cluster price bands (dips observed at 67, 134, 201 NGN)
    df['mrp_band'] = pd.cut(df['product_price'], bins=[0, 69, 136, 203, 500], labels=[1, 2, 3, 4]).astype(int)
    
    # Chain-wide SKU distribution depth
    df['store_presence_count'] = df.groupby('product_code')['store_code'].transform('count')
    
    # --- Category x Store & Store Format & Location Tier Interactions ---
    df['cat_store_combo'] = df['product_category'] + '_' + df['store_code']
    df['cat_store_freq'] = df.groupby('cat_store_combo')['id'].transform('count')
    df['cat_format_combo'] = df['product_category'] + '_' + df['store_format']
    df['cat_format_freq'] = df.groupby('cat_format_combo')['id'].transform('count')
    df['cat_tier_combo'] = df['product_category'] + '_' + df['store_location_tier']
    df['cat_tier_freq'] = df.groupby('cat_tier_combo')['id'].transform('count')

    # --- Store Owner Domain Features (Chief Emeka's Recommendations) ---
    # 1. Value density & Weight Brackets (Sachet <0.2kg, Standard 0.2-2kg, Bulk >2kg)
    df['price_per_kg'] = df['product_price'] / df['product_weight_kg'].replace(0, np.nan)
    df['price_per_kg'] = df['price_per_kg'].fillna(df['price_per_kg'].median())
    df['log_price_per_kg'] = np.log1p(df['price_per_kg'])
    df['weight_bracket'] = pd.cut(df['product_weight_kg'], bins=[0, 0.2, 2.0, 100], labels=[1, 2, 3]).astype(int)
    
    # 2. POS Impulse Rack Flag (Zero-visibility items under ₦250 at register)
    df['is_pos_impulse'] = ((df['visibility_was_zero'] == 1) & (df['product_price'] < 250)).astype(int)
    
    # 3. Shelf Space Allocation ROI (Visibility Share)
    df['visibility_share'] = df['shelf_visibility'] / (item_mean_vis + 1e-6)
    format_vis_mean = df.groupby('store_format')['shelf_visibility'].transform('mean')
    df['vis_to_format_mean'] = df['shelf_visibility'] / (format_vis_mean + 1e-6)
    
    # 4. Purchasing Power & Location Price Sensitivity
    df['price_x_tier'] = df['product_price'] * df['store_tier_ord']
    
    return df

def prepare_datasets(train_path='data/train.csv', test_path='data/test.csv'):
    """
    High-level function returning train/test feature matrices ready for ML.
    """
    train_df, test_df = load_raw_data(train_path, test_path)
    combined, target, n_train = clean_and_impute_data(train_df, test_df)
    features_df = engineer_features(combined)
    
    # Separate train and test
    train_feats = features_df.iloc[:n_train].copy().reset_index(drop=True)
    test_feats = features_df.iloc[n_train:].copy().reset_index(drop=True)
    
    train_feats['total_sales'] = target.values
    
    # Identify categorical columns to encode
    categorical_cols = [
        'product_code', 'product_family', 'fat_content', 'product_category',
        'store_code', 'store_size', 'store_location_tier', 'store_format',
        'store_profile', 'cat_store_combo', 'cat_format_combo', 'cat_tier_combo',
        'weight_bracket'
    ]
    
    # Encode categoricals cleanly with complete category space
    cat_mappings = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(features_df[col].astype(str))
        num_classes = len(le.classes_)
        cat_type = pd.CategoricalDtype(categories=list(range(num_classes)))
        
        train_feats[col] = pd.Series(le.transform(train_feats[col].astype(str)), dtype=cat_type)
        test_feats[col] = pd.Series(le.transform(test_feats[col].astype(str)), dtype=cat_type)
        cat_mappings[col] = le
        
    return train_feats, test_feats, categorical_cols

if __name__ == '__main__':
    print("Testing data preparation pipeline...")
    train_feats, test_feats, cat_cols = prepare_datasets()
    print(f"Train features shape: {train_feats.shape}")
    print(f"Test features shape: {test_feats.shape}")
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
    print("Null check train:", train_feats.isnull().sum().sum())
    print("Null check test:", test_feats.isnull().sum().sum())
    print("Sample feature columns:", list(train_feats.columns[:15]))
