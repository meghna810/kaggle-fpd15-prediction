"""
PayJoy FPD - GM-style time-aware pipeline.

Goal:
- Improve over build_model.py by tightening point-in-time features and validation.

Key design:
1) Strict time-aware rolling validation (fast: months 10-11)
2) Remove SideLoad/KnoxHack from train
3) Train window focused on months 6-11
4) Leakage-safe month-wise target encoding (only prior months)
5) Point-in-time SAFE entity history (counts/tenure only; no label history, no payment-history proxies)
6) Fast XGBoost config search + single final fit
"""

import os
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

SEED = 42
np.random.seed(SEED)
t0 = time.time()

# ------------------------------------------------------------
# Speed/quality controls
# ------------------------------------------------------------
# Fast mode: minimal folds + small config sweep, then single final fit.
FAST_MODE = True
FOLD_MONTHS = [10, 11] if FAST_MODE else [8, 9, 10, 11]
USE_CATBOOST = False  # turn on only if you can afford extra runtime

XGB_CANDIDATES = [
    {
        "name": "xgb_fast_1",
        "n_estimators": 1400,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.5,
        "reg_lambda": 4.0,
        "gamma": 0.15,
        "early_stopping_rounds": 80,
    },
    {
        "name": "xgb_fast_2",
        "n_estimators": 1800,
        "learning_rate": 0.025,
        "max_depth": 7,
        "min_child_weight": 25,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.7,
        "reg_lambda": 5.0,
        "gamma": 0.2,
        "early_stopping_rounds": 100,
    },
    {
        "name": "xgb_fast_3",
        "n_estimators": 1200,
        "learning_rate": 0.04,
        "max_depth": 5,
        "min_child_weight": 18,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.4,
        "reg_lambda": 3.5,
        "gamma": 0.1,
        "early_stopping_rounds": 70,
    },
]


def elapsed() -> str:
    return f"[{time.time()-t0:.0f}s]"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"{elapsed()} Loading files...")
    orders = pd.read_csv("Orders.csv", low_memory=False)
    test_ids = pd.read_csv("Test_OrderIDs.csv")

    for c in [
        "TRANSACTIONTIME",
        "LOAN_START_DATE",
        "FIRST_PAYMENT_DUE_TIMESTAMP",
        "MERCHANT_FIRST_SALE_DATE",
    ]:
        orders[c] = pd.to_datetime(orders[c], utc=True)
    orders["tx_month"] = orders["TRANSACTIONTIME"].dt.month.astype(np.int8)

    print(f"  Orders: {len(orders):,} | Test IDs: {len(test_ids):,}")
    return orders, test_ids


def add_safe_entity_history(df: pd.DataFrame) -> pd.DataFrame:
    print(f"{elapsed()} Building safe point-in-time entity history...")
    df = df.sort_values("TRANSACTIONTIME").reset_index(drop=True).copy()
    entity_defs = [("MERCHANTID", "merch"), ("CLERK_ID", "clerk"), ("ADMINID", "admin")]
    for entity, prefix in entity_defs:
        g = df.groupby(entity, sort=False)
        # Strictly prior observed volume (safe at order-time)
        df[f"{prefix}_cum_orders"] = g.cumcount().astype(np.int32)
        # Entity tenure proxy from first seen order timestamp
        df[f"{prefix}_first_tx"] = g["TRANSACTIONTIME"].transform("min")
        df[f"{prefix}_age_days"] = (
            (df["TRANSACTIONTIME"] - df[f"{prefix}_first_tx"]).dt.total_seconds() / 86400.0
        )
        df[f"{prefix}_order_velocity"] = df[f"{prefix}_cum_orders"] / (df[f"{prefix}_age_days"] + 1.0)
        df.drop(columns=[f"{prefix}_first_tx"], inplace=True)

    return df


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"{elapsed()} Basic engineered features...")
    out = df.copy()

    out["tx_hour"] = out["TRANSACTIONTIME"].dt.hour.astype(np.int8)
    out["tx_dow"] = out["TRANSACTIONTIME"].dt.dayofweek.astype(np.int8)
    out["tx_day"] = out["TRANSACTIONTIME"].dt.day.astype(np.int8)
    out["tx_is_weekend"] = (out["tx_dow"] >= 5).astype(np.int8)
    out["tx_is_night"] = out["tx_hour"].isin([0, 1, 2, 3, 4, 21, 22, 23]).astype(np.int8)
    out["hour_sin"] = np.sin(2 * np.pi * out["tx_hour"] / 24).astype(np.float32)
    out["hour_cos"] = np.cos(2 * np.pi * out["tx_hour"] / 24).astype(np.float32)

    out["days_to_due"] = (
        (out["FIRST_PAYMENT_DUE_TIMESTAMP"] - out["LOAN_START_DATE"]).dt.total_seconds() / 86400.0
    )
    out["merchant_age_days"] = (
        (out["TRANSACTIONTIME"] - out["MERCHANT_FIRST_SALE_DATE"]).dt.total_seconds() / 86400.0
    )
    out["merchant_is_new"] = (out["merchant_age_days"] < 30).astype(np.int8)

    # Currency-robust ratios to mitigate mixed nominal amounts.
    out["down_pmt_ratio"] = out["DOWN_PAYMENT_AMOUNT"] / (out["PURCHASE_AMOUNT"] + 1.0)
    out["finance_pmt_ratio"] = out["FINANCE_AMOUNT"] / (out["PURCHASE_AMOUNT"] + 1.0)
    out["total_due_ratio"] = out["TOTAL_DUE"] / (out["PURCHASE_AMOUNT"] + 1.0)
    out["finance_down_ratio"] = out["FINANCE_AMOUNT"] / (out["DOWN_PAYMENT_AMOUNT"] + 1.0)
    out["interest_ratio"] = (out["TOTAL_DUE"] - out["FINANCE_AMOUNT"]) / (out["FINANCE_AMOUNT"] + 1.0)

    out["log_finance"] = np.log1p(out["FINANCE_AMOUNT"].clip(lower=0))
    out["log_purchase"] = np.log1p(out["PURCHASE_AMOUNT"].clip(lower=0))
    out["log_down"] = np.log1p(out["DOWN_PAYMENT_AMOUNT"].clip(lower=0))
    out["log_merchant_age"] = np.log1p(out["merchant_age_days"].clip(lower=0))

    score_cols = ["FACE_RECOGNITION_SCORE", "IDVALIDATION_OVERALL_SCORE", "LIVENESS_SCORE", "OVERALL_SCORE"]
    out["score_mean"] = out[score_cols].mean(axis=1)
    out["score_min"] = out[score_cols].min(axis=1)
    out["score_std"] = out[score_cols].std(axis=1)
    out["overall_low"] = (out["OVERALL_SCORE"] < 50).astype(np.int8)

    return out


def apply_monthwise_oof_te(
    df: pd.DataFrame,
    cats: List[str],
    month_col: str = "tx_month",
    target_col: str = "FPD_15",
    smooth: float = 200.0,
) -> pd.DataFrame:
    """
    Time-safe target encoding:
    - For train month m: use only rows from earlier months (< m)
    - For test rows: use all train rows
    """
    print(f"{elapsed()} Time-safe month-wise target encoding...")
    out = df.copy()
    train_mask = out[target_col].notna()
    gm = out.loc[train_mask, target_col].mean()

    train_months = sorted(out.loc[train_mask, month_col].dropna().unique().tolist())
    for c in cats:
        te_col = f"{c}_te"
        out[te_col] = np.nan

        for m in train_months:
            val_idx = out.index[train_mask & (out[month_col] == m)]
            fit_idx = out.index[train_mask & (out[month_col] < m)]
            if len(val_idx) == 0:
                continue
            if len(fit_idx) == 0:
                out.loc[val_idx, te_col] = gm
                continue

            s = out.loc[fit_idx].groupby(c)[target_col].agg(["mean", "count"])
            te_map = (s["count"] * s["mean"] + smooth * gm) / (s["count"] + smooth)
            out.loc[val_idx, te_col] = out.loc[val_idx, c].map(te_map)

        # Test rows use all train rows
        s_all = out.loc[train_mask].groupby(c)[target_col].agg(["mean", "count"])
        te_all = (s_all["count"] * s_all["mean"] + smooth * gm) / (s_all["count"] + smooth)
        out.loc[~train_mask, te_col] = out.loc[~train_mask, c].map(te_all)

        out[te_col] = out[te_col].fillna(gm).astype(np.float32)

    return out


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "FPD_15",
        "FINANCEORDERID",
        "TRANSACTIONTIME",
        "LOAN_START_DATE",
        "FIRST_PAYMENT_DUE_TIMESTAMP",
        "MERCHANT_FIRST_SALE_DATE",
        "MERCHANT_LAST_SALE_DATE",
        "FIRST_PAYMENT_TIMESTAMP",
        "PAID_AMOUNT",
        "BALANCE",
        "NUMBER_OF_PAYMENTS",
        "STATE_NAME",
        "COUNTRY",
        "CURRENCY",
        "LOCK_PRODUCT",
        "LOCK_NAME",
        "STATE",
        "MANUFACTURER",
        "MODEL",
        "CITY",
        "MERCHANTID",
        "CLERK_ID",
        "ADMINID",
        "USER_STATE",
        "MERCHANT_STATE",
    }
    feats = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return list(dict.fromkeys(feats))


def rolling_folds(df_train: pd.DataFrame, fold_months: List[int]) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Train on months < m, validate on month m for selected months."""
    folds = []
    for m in fold_months:
        tr = df_train["tx_month"] < m
        va = df_train["tx_month"] == m
        if tr.sum() > 0 and va.sum() > 0:
            folds.append((np.where(tr.values)[0], np.where(va.values)[0], m))
    return folds


def fit_xgb(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    cfg: Dict[str, float],
) -> Tuple[xgb.XGBClassifier, np.ndarray, float, int]:
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=int(cfg["n_estimators"]),
        learning_rate=float(cfg["learning_rate"]),
        max_depth=int(cfg["max_depth"]),
        min_child_weight=float(cfg["min_child_weight"]),
        subsample=float(cfg["subsample"]),
        colsample_bytree=float(cfg["colsample_bytree"]),
        reg_alpha=float(cfg["reg_alpha"]),
        reg_lambda=float(cfg["reg_lambda"]),
        gamma=float(cfg["gamma"]),
        scale_pos_weight=spw,
        tree_method="hist",
        random_state=SEED,
        n_jobs=1,
        eval_metric="auc",
        early_stopping_rounds=int(cfg["early_stopping_rounds"]),
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    pred = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, pred)
    best_iter = int(model.best_iteration if model.best_iteration is not None else cfg["n_estimators"] - 1)
    return model, pred, auc, best_iter


def run_overfitting_audit(
    y_oof: np.ndarray,
    pred_oof: np.ndarray,
    fold_stats: List[Dict[str, float]],
    month_oof_stats: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Diagnostics to determine whether uplift is likely real or overfit.
    Returns summary metrics and prints a verdict.
    """
    print(f"{elapsed()} Running overfitting audit...")

    overall_auc = float(roc_auc_score(y_oof, pred_oof))
    shuffled_auc = float(roc_auc_score(np.random.permutation(y_oof), pred_oof))

    val_aucs = np.array([r["xgb_val_auc"] for r in fold_stats], dtype=float)
    train_aucs = np.array([r["xgb_train_auc"] for r in fold_stats], dtype=float)
    auc_gap = train_aucs - val_aucs

    mean_val_auc = float(np.mean(val_aucs))
    std_val_auc = float(np.std(val_aucs))
    mean_gap = float(np.mean(auc_gap))
    max_gap = float(np.max(auc_gap))

    month_aucs = np.array([r["auc"] for r in month_oof_stats], dtype=float)
    month_std = float(np.std(month_aucs))
    month_min = float(np.min(month_aucs))
    month_max = float(np.max(month_aucs))

    # Simple heuristic score
    risk_points = 0
    if mean_gap > 0.08:
        risk_points += 2
    elif mean_gap > 0.05:
        risk_points += 1
    if max_gap > 0.10:
        risk_points += 1
    if std_val_auc > 0.02:
        risk_points += 1
    if month_std > 0.025:
        risk_points += 1
    if abs(shuffled_auc - 0.5) > 0.02:
        risk_points += 2

    if risk_points >= 4:
        verdict = "likely_overfit"
    elif risk_points >= 2:
        verdict = "partially_overfit"
    else:
        verdict = "likely_real"

    print("-" * 64)
    print("OVERFITTING AUDIT")
    print(f"OOF AUC:                    {overall_auc:.5f}")
    print(f"Shuffled-label AUC:         {shuffled_auc:.5f} (should be ~0.50)")
    print(f"Fold val AUC mean/std:      {mean_val_auc:.5f} / {std_val_auc:.5f}")
    print(f"Fold train-val gap mean:    {mean_gap:.5f}")
    print(f"Fold train-val gap max:     {max_gap:.5f}")
    print(f"Month AUC min/max/std:      {month_min:.5f} / {month_max:.5f} / {month_std:.5f}")
    print(f"Leakage risk points:        {risk_points}")
    print(f"Verdict:                    {verdict}")
    print("-" * 64)

    return {
        "oof_auc": overall_auc,
        "shuffled_auc": shuffled_auc,
        "mean_val_auc": mean_val_auc,
        "std_val_auc": std_val_auc,
        "mean_gap": mean_gap,
        "max_gap": max_gap,
        "month_std": month_std,
        "risk_points": float(risk_points),
        "verdict": verdict,
    }


def fit_catboost_if_available(
    X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray
):
    try:
        from catboost import CatBoostClassifier
    except Exception:
        return None, None, None

    model = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=8.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=False,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, verbose=False)
    pred = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, pred)
    return model, pred, auc


def main():
    orders, test_ids = load_data()
    orders = add_safe_entity_history(orders)
    orders = add_basic_features(orders)

    # Filter policy from experiments.
    is_train = orders["FPD_15"].notna()
    is_test = orders["FINANCEORDERID"].isin(set(test_ids["FINANCEORDERID"]))
    bad_lock = orders["LOCK_NAME"].str.contains("SideLoad|KnoxHack", case=False, na=False)

    train_keep = is_train & (~bad_lock) & (orders["tx_month"].between(6, 11))
    test_keep = is_test
    df = orders.loc[train_keep | test_keep].copy()

    print(f"{elapsed()} Filtered train rows: {train_keep.sum():,}")
    print(f"{elapsed()} Filtered test rows: {test_keep.sum():,}")

    # Exclude LOCK_NAME_te per prior evidence.
    te_cols = ["COUNTRY", "MANUFACTURER", "LOCK_PRODUCT", "CURRENCY", "STATE", "MODEL", "CITY"]
    df = apply_monthwise_oof_te(df, te_cols, smooth=200.0)

    train_df = df[df["FPD_15"].notna()].copy()
    test_df = df[df["FPD_15"].isna()].copy()

    feat_cols = get_feature_columns(df)
    med = train_df[feat_cols].median()

    X_all = train_df[feat_cols].fillna(med).astype(np.float32)
    y_all = train_df["FPD_15"].astype(int).values
    X_test = test_df[feat_cols].fillna(med).astype(np.float32)

    folds = rolling_folds(train_df, FOLD_MONTHS)
    print(f"{elapsed()} Rolling folds: {[m for _, _, m in folds]}")
    if len(folds) == 0:
        raise ValueError(f"No valid folds found for months={FOLD_MONTHS}.")

    # --------------------------------------------------------
    # Fast config selection
    # --------------------------------------------------------
    print(f"{elapsed()} Fast XGBoost config selection...")
    cfg_scores = []
    for cfg in XGB_CANDIDATES:
        aucs = []
        iters = []
        for tr_idx, va_idx, _ in folds:
            X_tr, y_tr = X_all.iloc[tr_idx], y_all[tr_idx]
            X_va, y_va = X_all.iloc[va_idx], y_all[va_idx]
            _, _, auc_xgb, best_iter = fit_xgb(X_tr, y_tr, X_va, y_va, cfg)
            aucs.append(float(auc_xgb))
            iters.append(int(best_iter))
        mean_auc = float(np.mean(aucs))
        cfg_scores.append({"cfg": cfg, "mean_auc": mean_auc, "mean_best_iter": int(np.mean(iters))})
        print(f"{elapsed()} {cfg['name']}: mean AUC={mean_auc:.5f}, mean best_iter={int(np.mean(iters))}")

    cfg_scores.sort(key=lambda x: x["mean_auc"], reverse=True)
    best_cfg = dict(cfg_scores[0]["cfg"])
    best_cfg["n_estimators"] = int(cfg_scores[0]["mean_best_iter"] + 120)
    print(f"{elapsed()} Selected config: {best_cfg['name']} with n_estimators={best_cfg['n_estimators']}")

    oof_xgb = np.zeros(len(train_df), dtype=np.float32)
    oof_cb = np.zeros(len(train_df), dtype=np.float32)
    use_cb = USE_CATBOOST

    fold_rows = []
    for tr_idx, va_idx, m in folds:
        X_tr, y_tr = X_all.iloc[tr_idx], y_all[tr_idx]
        X_va, y_va = X_all.iloc[va_idx], y_all[va_idx]

        xgb_model, pred_xgb, auc_xgb, _ = fit_xgb(X_tr, y_tr, X_va, y_va, best_cfg)
        pred_xgb_train = xgb_model.predict_proba(X_tr)[:, 1]
        auc_xgb_train = float(roc_auc_score(y_tr, pred_xgb_train))
        oof_xgb[va_idx] = pred_xgb

        if use_cb:
            cb_model, pred_cb, auc_cb = fit_catboost_if_available(X_tr, y_tr, X_va, y_va)
            if cb_model is None:
                use_cb = False
                auc_cb = np.nan
            else:
                oof_cb[va_idx] = pred_cb
        else:
            auc_cb = np.nan

        fold_rows.append(
            {
                "month": int(m),
                "xgb_train_auc": auc_xgb_train,
                "xgb_val_auc": float(auc_xgb),
                "cb_val_auc": float(auc_cb) if not np.isnan(auc_cb) else np.nan,
            }
        )
        print(
            f"{elapsed()} Fold m={m}: "
            f"XGB train AUC={auc_xgb_train:.5f} | "
            f"XGB val AUC={auc_xgb:.5f} | "
            f"CB val AUC={auc_cb if np.isnan(auc_cb) else round(float(auc_cb), 5)}"
        )

    # OOF blend tuning
    valid_mask = np.zeros(len(train_df), dtype=bool)
    for _, va_idx, _ in folds:
        valid_mask[va_idx] = True

    y_oof = y_all[valid_mask]
    xgb_oof = oof_xgb[valid_mask]

    if use_cb:
        cb_oof = oof_cb[valid_mask]
        best_w, best_auc = 1.0, roc_auc_score(y_oof, xgb_oof)
        for w in np.arange(0.0, 1.01, 0.05):
            pred = w * xgb_oof + (1.0 - w) * cb_oof
            auc = roc_auc_score(y_oof, pred)
            if auc > best_auc:
                best_auc = auc
                best_w = float(w)
        print(f"{elapsed()} Best OOF blend: w_xgb={best_w:.2f}, w_cb={1-best_w:.2f}, AUC={best_auc:.5f}")
    else:
        best_w = 1.0
        best_auc = roc_auc_score(y_oof, xgb_oof)
        print(f"{elapsed()} CatBoost unavailable, using XGB only. OOF AUC={best_auc:.5f}")

    # Month-level OOF stability
    month_oof_stats = []
    valid_months = sorted(train_df.loc[valid_mask, "tx_month"].unique().tolist())
    for m in valid_months:
        m_mask = valid_mask & (train_df["tx_month"].values == m)
        if m_mask.sum() == 0:
            continue
        if use_cb:
            m_pred = best_w * oof_xgb[m_mask] + (1.0 - best_w) * oof_cb[m_mask]
        else:
            m_pred = oof_xgb[m_mask]
        m_auc = float(roc_auc_score(y_all[m_mask], m_pred))
        month_oof_stats.append({"month": int(m), "auc": m_auc, "n": int(m_mask.sum())})
        print(f"{elapsed()} Month {m} OOF AUC={m_auc:.5f} (n={int(m_mask.sum()):,})")

    # Overfitting audit verdict
    if use_cb:
        blend_oof = best_w * xgb_oof + (1.0 - best_w) * oof_cb[valid_mask]
    else:
        blend_oof = xgb_oof
    audit = run_overfitting_audit(y_oof, blend_oof, fold_rows, month_oof_stats)

    # Final train on all filtered training rows with selected config
    print(f"{elapsed()} Final training on all filtered train rows...")
    final_xgb = xgb.XGBClassifier(
        n_estimators=int(best_cfg["n_estimators"]),
        learning_rate=float(best_cfg["learning_rate"]),
        max_depth=int(best_cfg["max_depth"]),
        min_child_weight=float(best_cfg["min_child_weight"]),
        subsample=float(best_cfg["subsample"]),
        colsample_bytree=float(best_cfg["colsample_bytree"]),
        reg_alpha=float(best_cfg["reg_alpha"]),
        reg_lambda=float(best_cfg["reg_lambda"]),
        gamma=float(best_cfg["gamma"]),
        scale_pos_weight=(y_all == 0).sum() / max((y_all == 1).sum(), 1),
        tree_method="hist",
        random_state=SEED,
        n_jobs=1,
        eval_metric="auc",
    )
    final_xgb.fit(X_all, y_all, verbose=False)
    pred_test_xgb = final_xgb.predict_proba(X_test)[:, 1]

    if use_cb and best_w < 1.0:
        from catboost import CatBoostClassifier

        final_cb = CatBoostClassifier(
            iterations=5000,
            learning_rate=0.03,
            depth=7,
            l2_leaf_reg=8.0,
            loss_function="Logloss",
            random_seed=SEED,
            verbose=False,
        )
        final_cb.fit(X_all, y_all, verbose=False)
        pred_test_cb = final_cb.predict_proba(X_test)[:, 1]
        pred_test = best_w * pred_test_xgb + (1.0 - best_w) * pred_test_cb
    else:
        pred_test = pred_test_xgb

    sub = pd.DataFrame(
        {"FINANCEORDERID": test_df["FINANCEORDERID"].values, "FPD_15_pred": pred_test}
    ).sort_values("FINANCEORDERID").reset_index(drop=True)
    sub.to_csv("final_submission_gm.csv", index=False)

    # Also keep a pure XGB file for robustness.
    sub_xgb = pd.DataFrame(
        {"FINANCEORDERID": test_df["FINANCEORDERID"].values, "FPD_15_pred": pred_test_xgb}
    ).sort_values("FINANCEORDERID").reset_index(drop=True)
    sub_xgb.to_csv("final_submission_gm_xgb.csv", index=False)

    missing = len(set(test_ids["FINANCEORDERID"]) - set(sub["FINANCEORDERID"]))
    print("=" * 64)
    print(f"Features: {len(feat_cols)}")
    print(f"OOF AUC (rolling months): {best_auc:.5f}")
    print(f"Selected config: {best_cfg['name']}")
    print(f"Overfitting verdict: {audit['verdict']}")
    print(f"Saved: final_submission_gm.csv")
    print(f"Saved: final_submission_gm_xgb.csv")
    print(f"Missing test IDs: {missing}")
    print(f"Prediction range: [{pred_test.min():.5f}, {pred_test.max():.5f}]")
    print("=" * 64)


if __name__ == "__main__":
    main()
