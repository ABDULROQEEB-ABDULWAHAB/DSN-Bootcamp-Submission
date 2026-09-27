"""
CLEAN BEST SUBMISSION
Key Insight: ALL 1705 test rows are unseen product-store combinations.
Store-ratio calibration HURTS because it overfits to training product mix per store.
Strategy: NO calibration. Focus on category x format x price generalisation.
Best LB so far: 1071.18 (clean_uncalibrated, 1 seed, 2 models)
This: 5 seeds x 5 folds x 3 targets = 75 CatBoost fits, optimal blend.
"""

import os
import sys
import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder

sys.path.append('src')


def load_features():
    train = pd.read_csv('data/train.csv')
    test  = pd.read_csv('data/test.csv')

    target  = train['total_sales'].copy()
    n_train = len(train)
    combined = pd.concat([train.drop(columns=['total_sales']), test], ignore_index=True)

    # 1. Category standardisation
    combined['product_category'] = combined['product_category'].str.strip().str.title()

    # 2. Weight imputation
    wl = combined.dropna(subset=['product_weight_kg']).groupby('product_code')['product_weight_kg'].mean()
    combined['product_weight_kg'] = combined['product_weight_kg'].fillna(combined['product_code'].map(wl))
    cw = combined.groupby('product_category')['product_weight_kg'].transform('median')
    combined['product_weight_kg'] = combined['product_weight_kg'].fillna(cw).fillna(combined['product_weight_kg'].median())

    # 3. Store size imputation
    sz = {'STORE-JOR': 'Small', 'STORE-DKU': 'Small', 'STORE-OYG': 'Small'}
    combined['store_size'] = combined['store_size'].fillna(combined['store_code'].map(sz))

    # 4. Visibility correction
    combined['visibility_was_zero'] = (combined['shelf_visibility'] == 0.0).astype(int)
    pos = combined[combined['shelf_visibility'] > 0]
    pv  = pos.groupby('product_code')['shelf_visibility'].mean()
    cv  = pos.groupby('product_category')['shelf_visibility'].median()
    mask = combined['shelf_visibility'] == 0.0
    combined.loc[mask, 'shelf_visibility'] = combined.loc[mask, 'product_code'].map(pv)
    combined['shelf_visibility'] = (combined['shelf_visibility']
                                    .fillna(combined['product_category'].map(cv))
                                    .fillna(combined['shelf_visibility'].median()))

    # 5. Fat content
    non_edible = ['Health And Hygiene', 'Household', 'Others']
    combined['is_edible'] = (~combined['product_category'].isin(non_edible)).astype(int)
    combined.loc[combined['is_edible'] == 0, 'fat_content'] = 'Non-Edible'

    df = combined.copy()

    # Price features
    df['log_price']  = np.log1p(df['product_price'])
    df['sqrt_price'] = np.sqrt(df['product_price'])

    for grp in ['product_category', 'store_format', 'store_location_tier']:
        gm   = df.groupby(grp)['product_price'].transform('mean')
        gs   = df.groupby(grp)['product_price'].transform('std').replace(0, 1)
        gmed = df.groupby(grp)['product_price'].transform('median')
        df[f'price_to_{grp}_mean']   = df['product_price'] / gm
        df[f'price_diff_{grp}_mean'] = df['product_price'] - gm
        df[f'price_zscore_{grp}']    = (df['product_price'] - gm) / gs
        df[f'price_to_{grp}_median'] = df['product_price'] / gmed

    df['price_tier'] = pd.qcut(df['product_price'], q=5, labels=[1,2,3,4,5]).astype(int)
    df['mrp_band']   = pd.cut(df['product_price'], bins=[0,69,136,203,500], labels=[1,2,3,4]).astype(int)

    # Visibility features
    df['log_visibility'] = np.log1p(df['shelf_visibility'])
    for grp in ['store_code', 'product_category', 'store_format']:
        gm = df.groupby(grp)['shelf_visibility'].transform('mean')
        df[f'vis_to_{grp}_mean'] = df['shelf_visibility'] / (gm + 1e-6)

    # Store ordinal features
    store_fmt  = {'Corner Shop': 1, 'Superstore': 2, 'Standard Supermarket': 3, 'Flagship Hypermarket': 4}
    store_tier = {'Tier 1': 3, 'Tier 2': 2, 'Tier 3': 1}
    store_sz   = {'Small': 1, 'Medium': 2, 'Large': 3}
    df['store_format_ord']  = df['store_format'].map(store_fmt).fillna(2)
    df['store_tier_ord']    = df['store_location_tier'].map(store_tier).fillna(2)
    df['store_size_ord']    = df['store_size'].map(store_sz).fillna(2)
    # store_age_years is already a direct column in the dataset
    df['store_volume_mult'] = df['store_format_ord'] * df['store_tier_ord']

    # Weight features
    df['price_per_kg']     = (df['product_price'] / df['product_weight_kg'].replace(0, np.nan)).fillna(df['product_price'].median() / 5)
    df['log_price_per_kg'] = np.log1p(df['price_per_kg'])
    df['weight_bracket']   = pd.cut(df['product_weight_kg'], bins=[0,5,10,15,25], labels=[1,2,3,4]).astype(int)

    # Category x Store interaction (key generalising features)
    df['cat_store_combo']  = df['product_category'] + '_' + df['store_code']
    df['cat_format_combo'] = df['product_category'] + '_' + df['store_format']
    df['cat_tier_combo']   = df['product_category'] + '_' + df['store_location_tier']
    df['cat_store_freq']   = df.groupby('cat_store_combo')['id'].transform('count')
    df['cat_format_freq']  = df.groupby('cat_format_combo')['id'].transform('count')
    df['cat_tier_freq']    = df.groupby('cat_tier_combo')['id'].transform('count')

    # Price x Store interactions
    df['price_x_volume'] = df['product_price'] * df['store_volume_mult']
    df['price_x_size']   = df['product_price'] * df['store_size_ord']
    df['price_x_tier']   = df['product_price'] * df['store_tier_ord']
    df['price_x_age']    = df['product_price'] * df['store_age_years']  # store_age_years is a direct column

    # Visibility share
    item_vis   = df.groupby('product_code')['shelf_visibility'].transform('mean')
    format_vis = df.groupby('store_format')['shelf_visibility'].transform('mean')
    df['vis_to_format_mean'] = df['shelf_visibility'] / (format_vis + 1e-6)
    df['visibility_share']   = df['shelf_visibility'] / (item_vis + 1e-6)

    # Store presence
    df['store_presence_count'] = df.groupby('product_code')['store_code'].transform('count')

    # Premium flags
    df['is_hypermarket_top_mrp'] = ((df['store_format'] == 'Flagship Hypermarket') & (df['product_price'] > 185)).astype(int)
    df['is_corner_shop']         = (df['store_format'] == 'Corner Shop').astype(int)
    df['is_pos_impulse']         = ((df['visibility_was_zero'] == 1) & (df['product_price'] < 250)).astype(int)

    # Encode categoricals
    cat_cols = [
        'product_code', 'fat_content', 'product_category',
        'store_code', 'store_size', 'store_location_tier', 'store_format',
        'cat_store_combo', 'cat_format_combo', 'cat_tier_combo',
        'weight_bracket', 'mrp_band', 'price_tier'
    ]
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        n = len(le.classes_)
        df[col] = pd.Series(le.transform(df[col].astype(str)),
                            dtype=pd.CategoricalDtype(categories=list(range(n))))

    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    test_df  = df.iloc[n_train:].copy().reset_index(drop=True)
    train_df['total_sales'] = target.values

    return train_df, test_df, cat_cols


def run():
    print("=" * 70)
    print("  DSN MART: CLEAN BEST SUBMISSION (No Store Calibration)")
    print("  Key insight: ALL test combos are unseen => calibration HURTS")
    print("=" * 70)

    train_df, test_df, cat_cols = load_features()

    drop_cols    = ['id', 'total_sales']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X            = train_df[feature_cols]
    y            = train_df['total_sales'].values
    y_log        = np.log1p(y)
    prices_train = train_df['product_price'].values

    X_test       = test_df[feature_cols]
    prices_test  = test_df['product_price'].values
    test_ids     = test_df['id'].values

    cb_cat_cols = [c for c in cat_cols if c in X.columns]

    import catboost as cb

    seeds    = [42, 2026, 777, 101, 888]
    n_splits = 5
    n_total  = len(seeds) * n_splits * 3

    # Accumulators
    oof_dir   = np.zeros(len(X))
    oof_log   = np.zeros(len(X))
    oof_units = np.zeros(len(X))
    test_dir   = np.zeros(len(X_test))
    test_log   = np.zeros(len(X_test))
    test_units = np.zeros(len(X_test))
    run_count = 0

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*55}")
        print(f"  SEED {seed_idx+1}/{len(seeds)} (seed={seed})")
        print(f"{'='*55}")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        s_oof_dir   = np.zeros(len(X))
        s_oof_log   = np.zeros(len(X))
        s_oof_units = np.zeros(len(X))
        s_test_dir   = np.zeros(len(X_test))
        s_test_log   = np.zeros(len(X_test))
        s_test_units = np.zeros(len(X_test))

        base_params = dict(
            loss_function='RMSE', eval_metric='RMSE',
            learning_rate=0.025, depth=6, l2_leaf_reg=5.0,
            iterations=2500, random_seed=seed, verbose=False
        )

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_va, y_va = X.iloc[va_idx], y[va_idx]
            y_log_tr   = y_log[tr_idx]
            y_log_va   = y_log[va_idx]
            y_u_tr     = y_tr / prices_train[tr_idx]
            y_u_va     = y_va / prices_train[va_idx]

            # Direct
            m1 = cb.CatBoostRegressor(**base_params)
            m1.fit(X_tr, y_tr, eval_set=(X_va, y_va),
                   cat_features=cb_cat_cols, early_stopping_rounds=50, verbose=False)
            s_oof_dir[va_idx] = np.clip(m1.predict(X_va), 0, None)
            s_test_dir += np.clip(m1.predict(X_test), 0, None) / n_splits
            run_count += 1
            print(f"  [dir  f{fold+1} s{seed}] RMSE={root_mean_squared_error(y_va, s_oof_dir[va_idx]):.2f} ({run_count}/{n_total})")

            # LogSales
            m2 = cb.CatBoostRegressor(**base_params)
            m2.fit(X_tr, y_log_tr, eval_set=(X_va, y_log_va),
                   cat_features=cb_cat_cols, early_stopping_rounds=50, verbose=False)
            s_oof_log[va_idx] = np.clip(np.expm1(m2.predict(X_va)), 0, None)
            s_test_log += np.clip(np.expm1(m2.predict(X_test)), 0, None) / n_splits
            run_count += 1
            print(f"  [log  f{fold+1} s{seed}] RMSE={root_mean_squared_error(y_va, s_oof_log[va_idx]):.2f} ({run_count}/{n_total})")

            # Units
            m3 = cb.CatBoostRegressor(**base_params)
            m3.fit(X_tr, y_u_tr, eval_set=(X_va, y_u_va),
                   cat_features=cb_cat_cols, early_stopping_rounds=50, verbose=False)
            s_oof_units[va_idx] = np.clip(m3.predict(X_va), 0, None) * prices_train[va_idx]
            s_test_units += np.clip(m3.predict(X_test), 0, None) * prices_test / n_splits
            run_count += 1
            print(f"  [unit f{fold+1} s{seed}] RMSE={root_mean_squared_error(y_va, s_oof_units[va_idx]):.2f} ({run_count}/{n_total})")

        oof_dir   += s_oof_dir   / len(seeds)
        oof_log   += s_oof_log   / len(seeds)
        oof_units += s_oof_units / len(seeds)
        test_dir  += s_test_dir  / len(seeds)
        test_log  += s_test_log  / len(seeds)
        test_units += s_test_units / len(seeds)

        r_d = root_mean_squared_error(y, s_oof_dir)
        r_l = root_mean_squared_error(y, s_oof_log)
        r_u = root_mean_squared_error(y, s_oof_units)
        r_b = root_mean_squared_error(y, (s_oof_dir + s_oof_log + s_oof_units) / 3)
        print(f"\n  Seed {seed} summary: dir={r_d:.2f}  log={r_l:.2f}  unit={r_u:.2f}  blend={r_b:.2f}")

    # Final bagged scores
    r_dir   = root_mean_squared_error(y, oof_dir)
    r_log   = root_mean_squared_error(y, oof_log)
    r_units = root_mean_squared_error(y, oof_units)
    print(f"\n{'='*60}")
    print(f"  BAGGED OOF RMSE (5 seeds x 5 folds)")
    print(f"  Direct:   {r_dir:.2f}")
    print(f"  LogSales: {r_log:.2f}")
    print(f"  Units:    {r_units:.2f}")

    # Grid search blend weights
    best_rmse = float('inf')
    best_w    = (0.33, 0.34, 0.33)
    for w1 in np.arange(0.0, 1.01, 0.05):
        for w2 in np.arange(0.0, 1.01 - w1, 0.05):
            w3 = round(1.0 - w1 - w2, 6)
            if w3 < -1e-9:
                continue
            w3 = max(w3, 0.0)
            blend = w1 * oof_dir + w2 * oof_log + w3 * oof_units
            rmse  = root_mean_squared_error(y, blend)
            if rmse < best_rmse:
                best_rmse = rmse
                best_w    = (w1, w2, w3)

    w1, w2, w3 = best_w
    print(f"\n  Optimal blend: Direct={w1:.2f}  LogSales={w2:.2f}  Units={w3:.2f}")
    print(f"  Blended OOF RMSE: {best_rmse:.2f}")

    final_preds = np.clip(w1*test_dir + w2*test_log + w3*test_units, 0, None)

    # Diagnostics
    raw_train   = pd.read_csv('data/train.csv')
    raw_test    = pd.read_csv('data/test.csv')
    stores_tr   = raw_train['store_code'].values
    stores_te   = raw_test['store_code'].values

    print(f"\n{'='*60}")
    print(f"  SUBMISSION DIAGNOSTICS")
    print(f"  Rows:  {len(final_preds)}")
    print(f"  Min:   {final_preds.min():.2f}")
    print(f"  Max:   {final_preds.max():.2f}")
    print(f"  Mean:  {final_preds.mean():.2f}  (Train: {y.mean():.2f})")
    print(f"  Std:   {final_preds.std():.2f}   (Train: {y.std():.2f})")
    print(f"  Nulls: {np.isnan(final_preds).sum()}")
    print(f"\n  Store-Level Check (NO calibration applied):")
    for s in sorted(np.unique(stores_tr)):
        tr_m = y[stores_tr == s].mean()
        te_m = final_preds[stores_te == s].mean() if (stores_te == s).sum() > 0 else float('nan')
        diff = (te_m - tr_m) / tr_m * 100 if tr_m > 0 else 0
        print(f"    {s:10s}: Train={tr_m:7.2f} | Pred={te_m:7.2f} | Diff={diff:+6.1f}%")

    # Save
    ts    = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    sub   = pd.DataFrame({'id': test_ids, 'total_sales': np.round(final_preds, 2)})
    os.makedirs('submissions', exist_ok=True)
    fname = f"submissions/sub_clean_best_cv{best_rmse:.2f}_{ts}.csv"
    sub.to_csv(fname, index=False)
    sub.to_csv('submissions/latest_submission.csv', index=False)
    print(f"\n  Saved: {fname}")
    print(f"  Updated: submissions/latest_submission.csv")
    print(f"\n  DONE. Submit latest_submission.csv to Kaggle.")


if __name__ == '__main__':
    run()
