"""
PayJoy FPD_15 Prediction — Final Model v3 (no PH, proven LB-safe)

Design principles (anti-overfitting):
  1. Train only on months 6-11 (distribution-matched to Dec test, JS < 0.15)
  2. Remove SideLoad/KnoxHack rows (30-49% FPD in train, <0.02% of test)
  3. Drop LOCK_NAME_te (37% importance but 0.51 univariate AUC = noise)
  4. scale_pos_weight=1.0 (proven better on LB than auto SPW)
  5. Conservative regularization (depth=5, higher lambda)
  6. Multi-seed averaging to reduce variance
  7. NO Payment_History features (hurt LB every time: +0.006 local, -0.017 LB)
  8. OOT validation on month 11 (most representative of Dec test)

LB history:
  - V9 (all data, auto SPW):                  LB 0.603
  - v1_data_selection (m6-11, auto SPW):       LB 0.613  <-- best
  - v2 with PH features (m6-11, SPW=1):        LB 0.596  <-- PH hurt
  - This model: v1 data selection + SPW=1 + 3-seed avg, NO PH
"""
import os, time
os.environ['OMP_NUM_THREADS'] = '4'
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
# 1. LOAD DATA
# ============================================================
print(f"{elapsed()} Loading...", flush=True)
orders = pd.read_csv('Orders.csv', low_memory=False)
test_ids = pd.read_csv('Test_OrderIDs.csv')

date_cols = ['TRANSACTIONTIME', 'LOAN_START_DATE',
             'FIRST_PAYMENT_DUE_TIMESTAMP', 'MERCHANT_FIRST_SALE_DATE']
for c in date_cols:
    orders[c] = pd.to_datetime(orders[c], utc=True)

train_raw = orders[orders['FPD_15'].notna()].copy()
test_raw  = orders[orders['FINANCEORDERID'].isin(test_ids['FINANCEORDERID'])].copy()
print(f"  Full train: {len(train_raw):,} | Test: {len(test_raw):,}", flush=True)

# ============================================================
# 2. DATA SELECTION — proven to improve LB generalization
# ============================================================
print(f"\n{elapsed()} Data selection...", flush=True)

# 2a. Maturity filter: orders need 15+ days past due date to have a valid FPD_15 label
latest_date = train_raw['TRANSACTIONTIME'].max()
days_since_due = (latest_date - train_raw['FIRST_PAYMENT_DUE_TIMESTAMP']).dt.total_seconds() / 86400
maturity_mask = days_since_due >= 15
train_sel = train_raw[maturity_mask].copy()
print(f"  Maturity filter: {len(train_sel):,} kept (dropped {(~maturity_mask).sum():,})", flush=True)

# 2b. Remove SideLoad/KnoxHack: 30-49% FPD but <0.02% of test set
sideload = train_sel['LOCK_NAME'].str.contains(
    'SideLoad|KnoxHack|prodPhoneFinance', case=True, na=False)
print(f"  SideLoad/KnoxHack: removing {sideload.sum():,} rows "
      f"(FPD={train_sel.loc[sideload, 'FPD_15'].mean():.3f})", flush=True)
train_sel = train_sel[~sideload].copy()

# 2c. Month filter: months 1-5 have very different distribution (JS > 0.2)
train_sel['_month'] = train_sel['TRANSACTIONTIME'].dt.month
month_mask = train_sel['_month'].isin(range(6, 12))
print(f"  Month filter 6-11: {month_mask.sum():,} kept (dropped {(~month_mask).sum():,})", flush=True)
train_sel = train_sel[month_mask].copy()
train_sel.drop(columns=['_month'], inplace=True)

print(f"  Final selected: {len(train_sel):,} | FPD rate: {train_sel['FPD_15'].mean():.4f}", flush=True)

# ============================================================
# 3. CUMULATIVE ENTITY STATS (point-in-time safe)
#    Use ALL raw data for complete entity history
# ============================================================
print(f"\n{elapsed()} Cumulative entity stats...", flush=True)
all_df = pd.concat([train_raw, test_raw], ignore_index=True)
all_df = all_df.sort_values('TRANSACTIONTIME').reset_index(drop=True)

all_df['_fpd']   = all_df['FPD_15'].fillna(0.0)
all_df['_known'] = all_df['FPD_15'].notna().astype(float)

for entity, pfx in [('MERCHANTID', 'merch'), ('CLERK_ID', 'clerk'), ('ADMINID', 'admin')]:
    g = all_df.groupby(entity)
    cum_fpd   = g['_fpd'].transform(lambda x: x.cumsum().shift(1, fill_value=0))
    cum_known = g['_known'].transform(lambda x: x.cumsum().shift(1, fill_value=0))
    cum_ord   = g.cumcount()

    all_df[f'{pfx}_cum_fpd']    = cum_fpd
    all_df[f'{pfx}_cum_known']  = cum_known
    all_df[f'{pfx}_cum_orders'] = cum_ord
    all_df[f'{pfx}_cum_rate']   = cum_fpd / cum_known.replace(0, np.nan)

all_df.drop(columns=['_fpd', '_known'], inplace=True)
print("  Done", flush=True)

# ============================================================
# 4. TARGET ENCODING
#    Fit on SELECTED train months < 11 (no val leakage)
#    Drop LOCK_NAME (0.51 AUC = noise, was consuming 37% importance)
# ============================================================
print(f"\n{elapsed()} Target encoding...", flush=True)
oot_cutoff = pd.Timestamp('2025-11-01', tz='UTC')
selected_ids = set(train_sel['FINANCEORDERID'])

fit_mask = (all_df['FINANCEORDERID'].isin(selected_ids)) & \
           (all_df['TRANSACTIONTIME'] < oot_cutoff)
gm = all_df.loc[fit_mask, 'FPD_15'].mean()

TE_CATS = ['COUNTRY', 'MANUFACTURER', 'LOCK_PRODUCT', 'CURRENCY',
           'STATE', 'MODEL', 'CITY']
SMOOTH = 200

te_maps = {}
for col in TE_CATS:
    g = all_df.loc[fit_mask].groupby(col)['FPD_15']
    s = pd.DataFrame({'m': g.mean(), 'c': g.count()})
    s['te'] = (s['c'] * s['m'] + SMOOTH * gm) / (s['c'] + SMOOTH)
    te_maps[col] = s['te'].to_dict()
    all_df[f'{col}_te'] = all_df[col].map(te_maps[col]).fillna(gm).astype(np.float32)

print(f"  {len(TE_CATS)} categories encoded (LOCK_NAME excluded)", flush=True)

# ============================================================
# 5. ORDER-LEVEL FEATURES
# ============================================================
print(f"\n{elapsed()} Order features...", flush=True)
df = all_df

df['tx_hour'] = df['TRANSACTIONTIME'].dt.hour
df['tx_dow']  = df['TRANSACTIONTIME'].dt.dayofweek
df['tx_day']  = df['TRANSACTIONTIME'].dt.day
df['tx_is_weekend'] = (df['tx_dow'] >= 5).astype(np.int8)
df['tx_is_night']   = df['tx_hour'].isin([0,1,2,3,4,21,22,23]).astype(np.int8)

df['days_to_due'] = (df['FIRST_PAYMENT_DUE_TIMESTAMP'] - df['LOAN_START_DATE']).dt.total_seconds() / 86400
df['hour_sin'] = np.sin(2 * np.pi * df['tx_hour'] / 24).astype(np.float32)
df['hour_cos'] = np.cos(2 * np.pi * df['tx_hour'] / 24).astype(np.float32)

df['merchant_age_days'] = (df['TRANSACTIONTIME'] - df['MERCHANT_FIRST_SALE_DATE']).dt.total_seconds() / 86400
df['merchant_is_new']   = (df['merchant_age_days'] < 30).astype(np.int8)
df['log_merchant_age']  = np.log1p(df['merchant_age_days'].clip(lower=0))

df['down_pmt_ratio']    = df['DOWN_PAYMENT_AMOUNT'] / (df['PURCHASE_AMOUNT'] + 1)
df['finance_pmt_ratio'] = df['FINANCE_AMOUNT'] / (df['PURCHASE_AMOUNT'] + 1)
df['total_due_ratio']   = df['TOTAL_DUE'] / (df['PURCHASE_AMOUNT'] + 1)
df['finance_down_ratio']= df['FINANCE_AMOUNT'] / (df['DOWN_PAYMENT_AMOUNT'] + 1)
df['interest_ratio']    = (df['TOTAL_DUE'] - df['FINANCE_AMOUNT']) / (df['FINANCE_AMOUNT'] + 1)
df['log_finance']  = np.log1p(df['FINANCE_AMOUNT'].clip(lower=0))
df['log_purchase'] = np.log1p(df['PURCHASE_AMOUNT'].clip(lower=0))
df['log_down']     = np.log1p(df['DOWN_PAYMENT_AMOUNT'].clip(lower=0))

scols = ['FACE_RECOGNITION_SCORE', 'IDVALIDATION_OVERALL_SCORE',
         'LIVENESS_SCORE', 'OVERALL_SCORE']
df['score_mean'] = df[scols].mean(axis=1)
df['score_min']  = df[scols].min(axis=1)
df['score_std']  = df[scols].std(axis=1)
df['overall_low'] = (df['OVERALL_SCORE'] < 50).astype(np.int8)

df['merch_cum_x_vol'] = df['merch_cum_rate'].fillna(0) * np.log1p(df['merch_cum_orders'])
df['clerk_cum_x_vol'] = df['clerk_cum_rate'].fillna(0) * np.log1p(df['clerk_cum_orders'])
df['night_x_model']   = df['tx_is_night'] * df['MODEL_te']
df['model_x_country'] = df['MODEL_te'] * df['COUNTRY_te']

print("  Done", flush=True)

# ============================================================
# 6. FEATURE MATRIX
# ============================================================
print(f"\n{elapsed()} Feature matrix...", flush=True)
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

train_df = df[df['FINANCEORDERID'].isin(selected_ids)].copy()
test_df  = df[df['FINANCEORDERID'].isin(test_ids['FINANCEORDERID'])].copy()

medians = train_df[feat_cols].median()
X_all  = train_df[feat_cols].fillna(medians).astype(np.float32)
X_test = test_df[feat_cols].fillna(medians).astype(np.float32)
y_all  = train_df['FPD_15'].astype(int).values

print(f"  {len(feat_cols)} features | Train: {len(X_all):,} | Test: {len(X_test):,}", flush=True)

# ============================================================
# 7. OOT SPLIT — month 11 validation
# ============================================================
is_val = train_df['TRANSACTIONTIME'].values >= np.datetime64(oot_cutoff)
X_tr, y_tr = X_all[~is_val], y_all[~is_val]
X_va, y_va = X_all[is_val],  y_all[is_val]
print(f"  OOT split: Train {len(X_tr):,} | Val {len(X_va):,}", flush=True)

# ============================================================
# 8. TRAIN XGBoost — anti-overfit configuration
#    Key choices vs previous models:
#    - scale_pos_weight=1.0 (experiments: LB better than auto SPW)
#    - depth=5 with lower LR (more conservative)
#    - higher reg_lambda for L2 regularization
#    - multi-seed averaging reduces prediction variance
# ============================================================
print(f"\n{elapsed()} XGBoost training (3 seeds)...", flush=True)

XGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.02, max_depth=5,
    min_child_weight=20, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=5.0, gamma=0.2,
    scale_pos_weight=1.0, tree_method='hist',
    n_jobs=4, eval_metric='auc', early_stopping_rounds=150,
)

SEEDS = [42, 123, 2026]
val_preds_list = []
best_iters = []

for seed in SEEDS:
    m = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=0)
    vp = m.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, vp)
    print(f"  Seed {seed}: AUC={auc:.5f} (iter={m.best_iteration})", flush=True)
    val_preds_list.append(vp)
    best_iters.append(m.best_iteration)

xgb_val = np.mean(val_preds_list, axis=0)
xgb_auc = roc_auc_score(y_va, xgb_val)
avg_iter = int(np.mean(best_iters))
print(f"  Averaged AUC: {xgb_auc:.5f} (avg iter={avg_iter})", flush=True)

fi = pd.Series(m.feature_importances_, index=feat_cols).sort_values(ascending=False)
print("  Top 15 features:")
for name, val in fi.head(15).items():
    print(f"    {name:35s} {val:.4f}")

# ============================================================
# 9. LOGISTIC REGRESSION (diversifies the ensemble)
# ============================================================
print(f"\n{elapsed()} Logistic Regression...", flush=True)
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr.fillna(0))
X_va_sc = scaler.transform(X_va.fillna(0))

lr_m = LogisticRegression(max_iter=1000, C=0.1, random_state=SEED)
lr_m.fit(X_tr_sc, y_tr)
lr_val = lr_m.predict_proba(X_va_sc)[:, 1]
lr_auc = roc_auc_score(y_va, lr_val)
print(f"  LR AUC: {lr_auc:.5f}", flush=True)

# ============================================================
# 10. BLEND OPTIMIZATION on validation
# ============================================================
print(f"\n{elapsed()} Blend optimization...", flush=True)
best_blend_auc = 0
best_alpha = 0
for alpha in np.arange(0, 1.01, 0.05):
    blend = (1 - alpha) * xgb_val + alpha * lr_val
    ba = roc_auc_score(y_va, blend)
    if ba > best_blend_auc:
        best_blend_auc = ba
        best_alpha = alpha
    if alpha in [0, 0.1, 0.2, 0.3, 0.5, 1.0]:
        print(f"  alpha={alpha:.2f}: AUC={ba:.5f}", flush=True)

print(f"  Best blend: alpha={best_alpha:.2f} -> AUC={best_blend_auc:.5f}", flush=True)

# ============================================================
# 11. FINAL RETRAIN + PREDICT
#     Multi-seed XGBoost averaged, then blended with LR
# ============================================================
print(f"\n{elapsed()} Final retrain...", flush=True)
n_final = avg_iter + 100

test_xgb_list = []
for seed in SEEDS:
    params = {k: v for k, v in XGB_PARAMS.items()
              if k not in ('early_stopping_rounds', 'eval_metric')}
    params['n_estimators'] = n_final
    params['random_state'] = seed
    mf = xgb.XGBClassifier(**params)
    mf.fit(X_all, y_all, verbose=False)
    test_xgb_list.append(mf.predict_proba(X_test)[:, 1])

test_xgb = np.mean(test_xgb_list, axis=0)

X_all_sc  = scaler.fit_transform(X_all.fillna(0))
X_test_sc = scaler.transform(X_test.fillna(0))
lr_f = LogisticRegression(max_iter=1000, C=0.1, random_state=SEED)
lr_f.fit(X_all_sc, y_all)
test_lr = lr_f.predict_proba(X_test_sc)[:, 1]

test_blend = (1 - best_alpha) * test_xgb + best_alpha * test_lr

# ============================================================
# 12. SAVE SUBMISSIONS
# ============================================================
print(f"\n{elapsed()} Saving submissions...", flush=True)
for name, pred in [('final_submission_final.csv', test_blend),
                    ('final_submission_final_xgb.csv', test_xgb)]:
    sub = pd.DataFrame({
        'FINANCEORDERID': test_df['FINANCEORDERID'].values,
        'FPD_15_pred': pred,
    }).sort_values('FINANCEORDERID').reset_index(drop=True)
    sub.to_csv(name, index=False)

missing = len(set(test_ids['FINANCEORDERID']) - set(sub['FINANCEORDERID']))
print(f"  Rows: {len(sub):,} | Missing: {missing}", flush=True)
print(f"  XGB pred:   [{test_xgb.min():.4f}, {test_xgb.max():.4f}] mean={test_xgb.mean():.4f}", flush=True)
print(f"  Blend pred: [{test_blend.min():.4f}, {test_blend.max():.4f}] mean={test_blend.mean():.4f}", flush=True)

print(f"\n{'='*60}")
print(f"  XGB OOT AUC (3-seed avg): {xgb_auc:.5f}")
print(f"  LR  OOT AUC:             {lr_auc:.5f}")
print(f"  Best blend AUC:           {best_blend_auc:.5f} (alpha={best_alpha:.2f})")
print(f"  Final n_estimators:       {n_final}")
print(f"  Saved: final_submission_final.csv (blend)")
print(f"  Saved: final_submission_final_xgb.csv (XGB only)")
print(f"{'='*60}")
