import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sub = pd.read_csv('submissions/latest_submission.csv')
test['pred'] = sub['total_sales']

# Analyze STORE-7WS
tr_7ws = train[train['store_code'] == 'STORE-7WS']
te_7ws = test[test['store_code'] == 'STORE-7WS']

print("--- STORE-7WS Training Distribution ---")
print(f"Mean: {tr_7ws['total_sales'].mean():.2f}")
print(f"Std:  {tr_7ws['total_sales'].std():.2f}")
print(f"Percentile 90: {tr_7ws['total_sales'].quantile(0.90):.2f}")
print(f"Percentile 95: {tr_7ws['total_sales'].quantile(0.95):.2f}")
print(f"Max: {tr_7ws['total_sales'].max():.2f}")
print(f"Count > 6000: {(tr_7ws['total_sales'] > 6000).sum()} / {len(tr_7ws)} ({(tr_7ws['total_sales'] > 6000).mean()*100:.1f}%)")

print("\n--- STORE-7WS Current Test Predictions ---")
print(f"Mean: {te_7ws['pred'].mean():.2f}")
print(f"Std:  {te_7ws['pred'].std():.2f}")
print(f"Percentile 90: {te_7ws['pred'].quantile(0.90):.2f}")
print(f"Percentile 95: {te_7ws['pred'].quantile(0.95):.2f}")
print(f"Max: {te_7ws['pred'].max():.2f}")
print(f"Count > 6000: {(te_7ws['pred'] > 6000).sum()} / {len(te_7ws)} ({(te_7ws['pred'] > 6000).mean()*100:.1f}%)")
