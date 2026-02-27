"""
PayJoy FPD — V9: Back to basics that scored 0.603 on Kaggle.
Reconstruct V3 exactly, then try small improvements one at a time.
NO Payment_History (it hurt Kaggle score).
"""
import os, time
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
t0 = time.time()
def elapsed():
    return f"[{time.time()-t0:.0f}s]"

# ============================================================
# 1. LOAD
# ============================================================
print(f"{elapsed()} Loading...")
orders = pd.read_csv('Orders.csv', low_memory=False)
test_ids = pd.read_csv('Test_OrderIDs.csv')

for c in ['TRANSACTIONTIME', 'LOAN_START_DATE', 'FIRST_PAYMENT_DUE_TIMESTAMP', 'MERCHANT_FIRST_SALE_DATE']:
    orders[c] = pd.to_datetime(orders[c], utc=True)

train_raw = orders[orders['FPD_15'].notna()].copy()
test_raw = orders[orders['FINANCEORDERID'].isin(test_ids['FINANCEORDERID'])].copy()
print(f"  Train: {len(train_raw):,} | Test: {len(test_raw):,}")

# ============================================================
# 2. COMBINE & SORT + CUMULATIVE ENTITY STATS
# ============================================================
print(f"\n{elapsed()} Cumulative entity stats...")
all_df = pd.concat([train_raw, test_raw], ignore_index=True)
all_df = all_df.sort_values('TRANSACTIONTIME').reset_index(drop=True)

all_df['_fpd'] = all_df['FPD_15'].fillna(0.0)
all_df['_known'] = all_df['FPD_15'].notna().astype(float)

for entity, prefix in [('MERCHANTID', 'merch'), ('CLERK_ID', 'clerk'), ('ADMINID', 'admin')]:
    g = all_df.groupby(entity)
    all_df[f'{prefix}_cum_fpd'] = g['_fpd'].transform(lambda x: x.cumsum().shift(1, fill_value=0))
    all_df[f'{prefix}_cum_known'] = g['_known'].transform(lambda x: x.cumsum().shift(1, fill_value=0))
    all_df[f'{prefix}_cum_orders'] = g.cumcount()
    all_df[f'{prefix}_cum_rate'] = all_df[f'{prefix}_cum_fpd'] / all_df[f'{prefix}_cum_known'].replace(0, np.nan)

all_df.drop(columns=['_fpd', '_known'], inplace=True)
print("  Done")

# ============================================================
# 3. TARGET ENCODING (fit on months 1-10)
# ============================================================
print(f"\n{elapsed()} Target encoding...")
oot_cutoff = pd.Timestamp('2025-11-01', tz='UTC')
fit_mask = all_df['FPD_15'].notna() & (all_df['TRANSACTIONTIME'] < oot_cutoff)
gm = all_df.loc[fit_mask, 'FPD_15'].mean()

TE_CATS = ['LOCK_NAME', 'COUNTRY', 'MANUFACTURER', 'LOCK_PRODUCT', 'CURRENCY',
           'STATE', 'MODEL', 'CITY']
SMOOTH = 200

te_maps = {}
for col in TE_CATS:
    g = all_df.loc[fit_mask].groupby(col)['FPD_15']
    s = pd.DataFrame({'m': g.mean(), 'c': g.count()})
    s['te'] = (s['c'] * s['m'] + SMOOTH * gm) / (s['c'] + SMOOTH)
    te_maps[col] = s['te'].to_dict()
    all_df[f'{col}_te'] = all_df[col].map(te_maps[col]).fillna(gm).astype(np.float32)

# ============================================================
# 4. ORDER FEATURES
# ============================================================
print(f"\n{elapsed()} Order features...")
df = all_df

df['tx_hour'] = df['TRANSACTIONTIME'].dt.hour
df['tx_dow'] = df['TRANSACTIONTIME'].dt.dayofweek
df['tx_day'] = df['TRANSACTIONTIME'].dt.day
df['tx_is_weekend'] = (df['tx_dow'] >= 5).astype(np.int8)
df['tx_is_night'] = df['tx_hour'].isin([0,1,2,3,4,21,22,23]).astype(np.int8)
df['days_to_due'] = (df['FIRST_PAYMENT_DUE_TIMESTAMP'] - df['LOAN_START_DATE']).dt.total_seconds() / 86400
df['hour_sin'] = np.sin(2 * np.pi * df['tx_hour'] / 24).astype(np.float32)
df['hour_cos'] = np.cos(2 * np.pi * df['tx_hour'] / 24).astype(np.float32)

df['merchant_age_days'] = (df['TRANSACTIONTIME'] - df['MERCHANT_FIRST_SALE_DATE']).dt.total_seconds() / 86400
df['merchant_is_new'] = (df['merchant_age_days'] < 30).astype(np.int8)
df['log_merchant_age'] = np.log1p(df['merchant_age_days'].clip(lower=0))

df['down_pmt_ratio'] = df['DOWN_PAYMENT_AMOUNT'] / (df['PURCHASE_AMOUNT'] + 1)
df['finance_pmt_ratio'] = df['FINANCE_AMOUNT'] / (df['PURCHASE_AMOUNT'] + 1)
df['total_due_ratio'] = df['TOTAL_DUE'] / (df['PURCHASE_AMOUNT'] + 1)
df['finance_down_ratio'] = df['FINANCE_AMOUNT'] / (df['DOWN_PAYMENT_AMOUNT'] + 1)
df['interest_ratio'] = (df['TOTAL_DUE'] - df['FINANCE_AMOUNT']) / (df['FINANCE_AMOUNT'] + 1)
df['log_finance'] = np.log1p(df['FINANCE_AMOUNT'].clip(lower=0))
df['log_purchase'] = np.log1p(df['PURCHASE_AMOUNT'].clip(lower=0))
df['log_down'] = np.log1p(df['DOWN_PAYMENT_AMOUNT'].clip(lower=0))

scols = ['FACE_RECOGNITION_SCORE', 'IDVALIDATION_OVERALL_SCORE', 'LIVENESS_SCORE', 'OVERALL_SCORE']
df['score_mean'] = df[scols].mean(axis=1)
df['score_min'] = df[scols].min(axis=1)
df['score_std'] = df[scols].std(axis=1)
df['overall_low'] = (df['OVERALL_SCORE'] < 50).astype(np.int8)

df['merch_cum_x_vol'] = df['merch_cum_rate'].fillna(0) * np.log1p(df['merch_cum_orders'])
df['clerk_cum_x_vol'] = df['clerk_cum_rate'].fillna(0) * np.log1p(df['clerk_cum_orders'])
df['night_x_model'] = df['tx_is_night'] * df['MODEL_te']
df['model_x_country'] = df['MODEL_te'] * df['COUNTRY_te']

# ============================================================
# 5. FEATURE MATRIX
# ============================================================
print(f"\n{elapsed()} Feature matrix...")
EXCLUDE = {
    'FPD_15', 'FINANCEORDERID',
    'TRANSACTIONTIME', 'LOAN_START_DATE', 'FIRST_PAYMENT_DUE_TIMESTAMP',
    'MERCHANT_FIRST_SALE_DATE', 'MERCHANT_LAST_SALE_DATE',
    'FIRST_PAYMENT_TIMESTAMP', 'PAID_AMOUNT', 'BALANCE', 'NUMBER_OF_PAYMENTS',
    'STATE_NAME', 'COUNTRY', 'CURRENCY', 'LOCK_PRODUCT', 'LOCK_NAME', 'STATE',
    'MANUFACTURER', 'MODEL', 'MERCHANTID', 'CLERK_ID', 'ADMINID',
    'CITY', 'USER_STATE', 'MERCHANT_STATE',
}
feat_cols = [c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
feat_cols = list(dict.fromkeys(feat_cols))

train_mask = df['FPD_15'].notna()
train_df = df[train_mask].copy()
test_df = df[~train_mask].copy()

medians = train_df[feat_cols].median()
X_all = train_df[feat_cols].fillna(medians).astype(np.float32)
X_test = test_df[feat_cols].fillna(medians).astype(np.float32)
y_all = train_df['FPD_15'].astype(int).values

print(f"  {len(feat_cols)} features")

# ============================================================
# 6. OOT SPLIT
# ============================================================
is_val = train_df['TRANSACTIONTIME'].values >= np.datetime64(oot_cutoff)
X_tr, y_tr = X_all[~is_val], y_all[~is_val]
X_va, y_va = X_all[is_val], y_all[is_val]
spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
print(f"  Train: {len(X_tr):,} | Val: {len(X_va):,} | spw: {spw:.2f}")

# ============================================================
# 7. TRAIN XGBOOST (same config as V3 which scored 0.603)
# ============================================================
print(f"\n{elapsed()} XGBoost...")
xgb_m = xgb.XGBClassifier(
    n_estimators=3000, learning_rate=0.03, max_depth=6,
    min_child_weight=15, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=3.0, gamma=0.1,
    scale_pos_weight=spw, tree_method='hist',
    random_state=SEED, n_jobs=1, eval_metric='auc',
    early_stopping_rounds=100,
)
xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=100)
xgb_val = xgb_m.predict_proba(X_va)[:, 1]
xgb_auc = roc_auc_score(y_va, xgb_val)
print(f"  XGBoost AUC: {xgb_auc:.5f} (iter={xgb_m.best_iteration})")

fi = pd.Series(xgb_m.feature_importances_, index=feat_cols).sort_values(ascending=False)
print("  Top 15:")
for name, val in fi.head(15).items():
    print(f"    {name:35s} {val:.4f}")

# ============================================================
# 8. LOGISTIC REGRESSION (ensemble component)
# ============================================================
print(f"\n{elapsed()} Logistic Regression...")
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr.fillna(0))
X_va_sc = scaler.transform(X_va.fillna(0))
lr_m = LogisticRegression(max_iter=1000, C=0.1, random_state=SEED)
lr_m.fit(X_tr_sc, y_tr)
lr_val = lr_m.predict_proba(X_va_sc)[:, 1]
lr_auc = roc_auc_score(y_va, lr_val)
print(f"  LR AUC: {lr_auc:.5f}")

# ============================================================
# 9. ENSEMBLE (try blending)
# ============================================================
print(f"\n{elapsed()} Ensemble...")
for alpha in [0.0, 0.1, 0.2, 0.3, 0.5]:
    blend = (1 - alpha) * xgb_val + alpha * lr_val
    blend_auc = roc_auc_score(y_va, blend)
    print(f"  alpha={alpha:.1f} (XGB={1-alpha:.1f}, LR={alpha:.1f}): AUC={blend_auc:.5f}")

# ============================================================
# 10. FINAL: RETRAIN ON ALL + PREDICT
#     Use only XGBoost (it scored 0.603 alone on Kaggle).
#     The blend might help but let's submit the simpler model.
# ============================================================
print(f"\n{elapsed()} Final retrain (XGBoost only, all data)...")
spw_f = (y_all == 0).sum() / max((y_all == 1).sum(), 1)

xgb_f = xgb.XGBClassifier(
    n_estimators=xgb_m.best_iteration + 100,
    learning_rate=0.03, max_depth=6, min_child_weight=15,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=3.0, gamma=0.1,
    scale_pos_weight=spw_f, tree_method='hist',
    random_state=SEED, n_jobs=1,
)
xgb_f.fit(X_all, y_all, verbose=False)
test_pred = xgb_f.predict_proba(X_test)[:, 1]

sub = pd.DataFrame({
    'FINANCEORDERID': test_df['FINANCEORDERID'].values,
    'FPD_15_pred': test_pred,
}).sort_values('FINANCEORDERID').reset_index(drop=True)
sub.to_csv('final_submission.csv', index=False)

print(f"\n  Rows: {len(sub):,} | Missing: {len(set(test_ids['FINANCEORDERID'])-set(sub['FINANCEORDERID']))}")
print(f"  Pred: [{test_pred.min():.4f}, {test_pred.max():.4f}] mean={test_pred.mean():.4f}")

# Also save blend version
X_all_sc = scaler.fit_transform(X_all.fillna(0))
X_test_sc = scaler.transform(X_test.fillna(0))
lr_f = LogisticRegression(max_iter=1000, C=0.1, random_state=SEED)
lr_f.fit(X_all_sc, y_all)
tp_lr = lr_f.predict_proba(X_test_sc)[:, 1]

blend_pred = 0.9 * test_pred + 0.1 * tp_lr
sub_blend = pd.DataFrame({
    'FINANCEORDERID': test_df['FINANCEORDERID'].values,
    'FPD_15_pred': blend_pred,
}).sort_values('FINANCEORDERID').reset_index(drop=True)
sub_blend.to_csv('final_submission_blend.csv', index=False)

print(f"  Blend pred: [{blend_pred.min():.4f}, {blend_pred.max():.4f}] mean={blend_pred.mean():.4f}")

print(f"\n{'='*60}")
print(f"  XGB OOT AUC:  {xgb_auc:.5f}")
print(f"  LR OOT AUC:   {lr_auc:.5f}")
print(f"  Saved: final_submission.csv (XGB only)")
print(f"  Saved: final_submission_blend.csv (90% XGB + 10% LR)")
print(f"{'='*60}")
