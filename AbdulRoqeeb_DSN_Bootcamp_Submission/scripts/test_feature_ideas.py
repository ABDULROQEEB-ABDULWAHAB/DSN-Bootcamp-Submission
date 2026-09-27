import sys
import os
sys.path.append('src')
import numpy as np
import pandas as pd
from features import prepare_datasets

train_df, test_df, cat_cols = prepare_datasets()
y = train_df['total_sales']

p_per_kg = train_df['product_price'] / train_df['product_weight_kg']
p_x_size = train_df['product_price'] * train_df['store_size_ord']
p_x_tier = train_df['product_price'] * train_df['store_tier_ord']

print(f"Corr product_price: {np.corrcoef(train_df['product_price'], y)[0,1]:.4f}")
print(f"Corr price_per_kg:  {np.corrcoef(p_per_kg, y)[0,1]:.4f}")
print(f"Corr price_x_size:  {np.corrcoef(p_x_size, y)[0,1]:.4f}")
print(f"Corr price_x_tier:  {np.corrcoef(p_x_tier, y)[0,1]:.4f}")

# Check correlation with store format volume estimate
store_vol_map = {
    'Corner Shop': 0.15,
    'Standard Supermarket': 1.0,
    'Superstore': 0.9,
    'Flagship Hypermarket': 1.6
}
raw_train = pd.read_csv('data/train.csv')
p_x_vol = train_df['product_price'] * raw_train['store_format'].map(store_vol_map)
print(f"Corr price_x_store_volume: {np.corrcoef(p_x_vol, y)[0,1]:.4f}")
