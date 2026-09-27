import sys
import os
sys.path.append('src')
import numpy as np
import pandas as pd
from features import prepare_datasets

train_df, test_df, cat_cols = prepare_datasets()
raw_train = pd.read_csv('data/train.csv')
raw_test = pd.read_csv('data/test.csv')

print("--- Train Target Stats ---")
print(f"Mean: {train_df['total_sales'].mean():.2f}")
print(f"Std:  {train_df['total_sales'].std():.2f}")
print(f"Min:  {train_df['total_sales'].min():.2f}, Max: {train_df['total_sales'].max():.2f}")
print(f"Percentiles 50, 75, 90, 95, 99: {np.percentile(train_df['total_sales'], [50, 75, 90, 95, 99])}")

print("\n--- Sales Distribution by Store Code ---")
for s, grp in raw_train.groupby('store_code'):
    print(f"{s:10s} | count={len(grp):4d} | mean={grp['total_sales'].mean():7.2f} | std={grp['total_sales'].std():7.2f} | p95={grp['total_sales'].quantile(0.95):7.2f} | max={grp['total_sales'].max():8.2f}")

print("\n--- Checking Test Set Predictions Range in latest_submission.csv ---")
sub = pd.read_csv('submissions/latest_submission.csv')
raw_test['pred'] = sub['total_sales']
for s, grp in raw_test.groupby('store_code'):
    print(f"{s:10s} | count={len(grp):4d} | mean={grp['pred'].mean():7.2f} | std={grp['pred'].std():7.2f} | p95={grp['pred'].quantile(0.95):7.2f} | max={grp['pred'].max():8.2f}")
