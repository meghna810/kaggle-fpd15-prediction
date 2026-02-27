"""
PayJoy FPD_15 — Best-of-all pipeline (LB + generalization focused).

What this combines:
- Maturity-clean labels (sota)
- Strict rolling OOT validation (gm/sota)
- Point-in-time safe entity history (gm/sota)
- Month-wise target encoding (gm) for XGB numeric branch
- CatBoost on raw categoricals (sota)
- City name normalization to reduce spelling/format variance
- OOF blend weight search + multi-seed final averaging (final)

Outputs:
- final_submission_best_conservative.csv
- final_submission_best_aggressive.csv
- final_submission_best_pure_cat.csv
- final_submission_best_cat_ensemble.csv
- final_submission_best_lb_leaning.csv
- best_summary.json
"""

import os
import json
import time
import argparse
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

import multiprocessing
N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)

SEED = 42
np.random.seed(SEED)
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


@dataclass
class RunConfig:
    min_train_month: int = 6
    max_train_month: int = 11
    fold_months: Tuple[int, ...] = (9, 10, 11)
    snapshot_date: str = "2025-12-01T00:00:00Z"
    remove_sideload: bool = True
    use_catboost: bool = True
    xgb_seeds: Tuple[int, ...] = (42, 123, 2026)
    cat_seeds: Tuple[int, ...] = (42, 123, 2026)
    time_weight_alpha: float = 0.8
    ab: bool = False


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--min-train-month", type=int, default=6)
    p.add_argument("--max-train-month", type=int, default=11)
    p.add_argument("--fold-months", type=str, default="9,10,11")
    p.add_argument("--snapshot-date", type=str, default="2025-12-01T00:00:00Z")
    p.add_argument("--keep-sideload", action="store_true")
    p.add_argument("--no-catboost", action="store_true")
    p.add_argument("--time-weight-alpha", type=float, default=0.8)
    p.add_argument("--ab", action="store_true", help="Run compact A/B OOF comparison and exit")
    a = p.parse_args()
    fold_months = tuple(int(x) for x in a.fold_months.split(",") if x.strip())
    return RunConfig(
        min_train_month=a.min_train_month,
        max_train_month=a.max_train_month,
        fold_months=fold_months,
        snapshot_date=a.snapshot_date,
        remove_sideload=not a.keep_sideload,
        use_catboost=not a.no_catboost,
        time_weight_alpha=a.time_weight_alpha,
        ab=a.ab,
    )


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    log("Loading Orders and Test IDs...")
    orders = pd.read_csv("Orders.csv", low_memory=False)
    test_ids = pd.read_csv("Test_OrderIDs.csv")
    for c in [
        "TRANSACTIONTIME",
        "LOAN_START_DATE",
        "FIRST_PAYMENT_DUE_TIMESTAMP",
        "MERCHANT_FIRST_SALE_DATE",
        "MERCHANT_LAST_SALE_DATE",
    ]:
        orders[c] = pd.to_datetime(orders[c], utc=True, errors="coerce")
    orders["tx_month"] = orders["TRANSACTIONTIME"].dt.month.astype(np.int8)
    log(f"Rows: Orders={len(orders):,} | TestIDs={len(test_ids):,}")
    return orders, test_ids


def prepare_labels(orders: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """Keep only maturity-clean labels for training/validation."""
    out = orders.copy()
    snapshot_ts = pd.Timestamp(snapshot_date, tz="UTC")
    maturity_ts = out["FIRST_PAYMENT_DUE_TIMESTAMP"] + pd.Timedelta(days=15)
    out["label_matured"] = out["FPD_15"].notna() & (maturity_ts <= snapshot_ts)
    return out


def add_safe_entity_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("TRANSACTIONTIME").reset_index(drop=True).copy()
    # Global FPD mean from labeled rows — used as smoothing prior for entity FPD rate
    gm_fpd = float(out["FPD_15"].dropna().mean()) if out["FPD_15"].notna().any() else 0.05
    # Fill NaN FPD (test / unmatured rows) with 0 for cumulative counting purposes.
    # Since test rows are month 12 and come last in time order, their FPD fill
    # does not contaminate earlier rows' cumulative sums.
    out["_fpd_fill"] = out["FPD_15"].fillna(0.0).astype(np.float32)
    FPD_SMOOTH = 5.0  # small smoothing to preserve entity-level signal

    for entity, prefix in [("MERCHANTID", "merch"), ("CLERK_ID", "clerk"), ("ADMINID", "admin")]:
        g = out.groupby(entity, sort=False)
        out[f"{prefix}_cum_orders"] = g.cumcount().astype(np.int32)
        out[f"{prefix}_first_tx"] = g["TRANSACTIONTIME"].transform("min")
        out[f"{prefix}_age_days"] = (
            (out["TRANSACTIONTIME"] - out[f"{prefix}_first_tx"]).dt.total_seconds() / 86400.0
        )
        out[f"{prefix}_order_velocity"] = out[f"{prefix}_cum_orders"] / (out[f"{prefix}_age_days"] + 1.0)
        out.drop(columns=[f"{prefix}_first_tx"], inplace=True)

        # Time-safe prior FPD rate: cumsum up to (but not including) current row
        cum_fpd_incl = out.groupby(entity)["_fpd_fill"].cumsum()
        cum_fpd_prior = (cum_fpd_incl - out["_fpd_fill"]).astype(np.float32)
        cum_orders_prior = out[f"{prefix}_cum_orders"].astype(np.float32)
        # Smoothed rate shrinks to global mean when the entity has few prior orders
        out[f"{prefix}_fpd_rate"] = (
            (cum_fpd_prior + FPD_SMOOTH * gm_fpd) / (cum_orders_prior + FPD_SMOOTH)
        ).astype(np.float32)

    out.drop(columns=["_fpd_fill"], inplace=True)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["CITY_NORM"] = out["CITY"].map(normalize_city)
    out["STATE_NORM"] = out["STATE"].map(normalize_code)
    out["COUNTRY_NORM"] = out["COUNTRY"].map(normalize_code)
    out["CURRENCY_NORM"] = out["CURRENCY"].map(normalize_code)
    out["MODEL_NORM"] = out["MODEL"].map(normalize_alnum)
    out["MANUFACTURER_NORM"] = out["MANUFACTURER"].map(normalize_alnum)
    out["LOCK_PRODUCT_NORM"] = out["LOCK_PRODUCT"].map(normalize_alnum)
    out["USER_STATE_NORM"] = out["USER_STATE"].map(normalize_code)
    out["MERCHANT_STATE_NORM"] = out["MERCHANT_STATE"].map(normalize_code)
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
    out["log_merchant_age"] = np.log1p(out["merchant_age_days"].clip(lower=0))

    # Merchant recency: days since last sale before this transaction
    out["merchant_days_since_last_sale"] = (
        (out["TRANSACTIONTIME"] - out["MERCHANT_LAST_SALE_DATE"]).dt.total_seconds() / 86400.0
    )
    out["log_merchant_days_since_last"] = np.log1p(
        out["merchant_days_since_last_sale"].clip(lower=0)
    )
    out["merchant_dormant"] = (out["merchant_days_since_last_sale"] > 60).astype(np.int8)

    out["down_pmt_ratio"] = out["DOWN_PAYMENT_AMOUNT"] / (out["PURCHASE_AMOUNT"] + 1.0)
    out["finance_pmt_ratio"] = out["FINANCE_AMOUNT"] / (out["PURCHASE_AMOUNT"] + 1.0)
    out["total_due_ratio"] = out["TOTAL_DUE"] / (out["PURCHASE_AMOUNT"] + 1.0)
    out["finance_down_ratio"] = out["FINANCE_AMOUNT"] / (out["DOWN_PAYMENT_AMOUNT"] + 1.0)
    out["interest_ratio"] = (out["TOTAL_DUE"] - out["FINANCE_AMOUNT"]) / (out["FINANCE_AMOUNT"] + 1.0)

    for c in ["FINANCE_AMOUNT", "PURCHASE_AMOUNT", "TOTAL_DUE", "DOWN_PAYMENT_AMOUNT"]:
        out[f"log_{c.lower()}"] = np.log1p(out[c].clip(lower=0))

    sc = ["FACE_RECOGNITION_SCORE", "IDVALIDATION_OVERALL_SCORE", "LIVENESS_SCORE", "OVERALL_SCORE"]
    out["score_mean"] = out[sc].mean(axis=1)
    out["score_min"] = out[sc].min(axis=1)
    out["score_std"] = out[sc].std(axis=1)
    out["overall_low"] = (out["OVERALL_SCORE"] < 50).astype(np.int8)

    # Biometric score interactions — products and gaps are more discriminative than means
    out["kyc_product"] = (
        out["FACE_RECOGNITION_SCORE"] * out["LIVENESS_SCORE"]
    ).astype(np.float32) / 10000.0
    out["face_vs_liveness_gap"] = (
        out["FACE_RECOGNITION_SCORE"] - out["LIVENESS_SCORE"]
    ).astype(np.float32)
    out["kyc_all_low"] = (
        (out["FACE_RECOGNITION_SCORE"] < 50)
        & (out["LIVENESS_SCORE"] < 50)
        & (out["OVERALL_SCORE"] < 50)
    ).astype(np.int8)
    return out


def _normalize_text(val, keep_digits: bool, uppercase: bool) -> str:
    if pd.isna(val):
        return "na"
    s = str(val).strip().lower()
    if s in {"", "na", "n/a", "none", "null", "nan", "unknown", "unk"}:
        return "na"
    s = s.upper() if uppercase else s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    if keep_digits:
        s = re.sub(r"[^a-zA-Z0-9\\s]", " ", s)
    else:
        s = re.sub(r"[^a-zA-Z\\s]", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s if s else "na"


def normalize_city(val) -> str:
    return _normalize_text(val, keep_digits=False, uppercase=False)


def normalize_alnum(val) -> str:
    return _normalize_text(val, keep_digits=True, uppercase=False)


def normalize_code(val) -> str:
    s = _normalize_text(val, keep_digits=True, uppercase=True)
    return s.replace(" ", "") if s != "na" else "na"


def compute_time_weights(df: pd.DataFrame, min_m: int, max_m: int, alpha: float):
    if alpha <= 0.0:
        return None
    denom = max(max_m - min_m, 1)
    w = 1.0 + alpha * (df["tx_month"].astype(float) - float(min_m)) / float(denom)
    return w.astype(np.float32).values


def apply_currency_norm(train_df: pd.DataFrame, other_df: pd.DataFrame):
    out_tr = train_df.copy()
    out_ot = other_df.copy()
    for c in ["FINANCE_AMOUNT", "PURCHASE_AMOUNT", "TOTAL_DUE", "DOWN_PAYMENT_AMOUNT"]:
        med_by_ccy = out_tr.groupby("CURRENCY")[c].median()
        overall_med = float(out_tr[c].median())
        denom_tr = out_tr["CURRENCY"].map(med_by_ccy).fillna(overall_med) + 1.0
        denom_ot = out_ot["CURRENCY"].map(med_by_ccy).fillna(overall_med) + 1.0
        out_tr[f"{c.lower()}_ccy_norm"] = (out_tr[c] / denom_tr).astype(np.float32)
        out_ot[f"{c.lower()}_ccy_norm"] = (out_ot[c] / denom_ot).astype(np.float32)
    return out_tr, out_ot


def make_train_test(
    df: pd.DataFrame, test_ids: pd.DataFrame, cfg: RunConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    is_test = df["FINANCEORDERID"].isin(set(test_ids["FINANCEORDERID"]))
    train_mask = (
        df["label_matured"]
        & df["tx_month"].between(cfg.min_train_month, cfg.max_train_month)
    )
    if cfg.remove_sideload:
        bad = df["LOCK_NAME"].str.contains("SideLoad|KnoxHack|prodPhoneFinance", case=False, na=False)
        train_mask = train_mask & (~bad)
    out = df[train_mask | is_test].copy()
    train_df = out[out["FPD_15"].notna() & out["label_matured"]].copy()
    test_df = out[out["FINANCEORDERID"].isin(set(test_ids["FINANCEORDERID"]))].copy()
    return train_df, test_df


def fit_te_loo(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    te_cols: List[str],
    smooth: float = 200.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leakage-safe target encoding:
    - Train rows use leave-one-out encoding (no self-target leakage)
    - Val/Test rows use full-train encoding
    """
    tr = train_df.copy()
    va = val_df.copy()
    te = test_df.copy()
    gm = tr["FPD_15"].mean()
    y = tr["FPD_15"].astype(float)
    for c in te_cols:
        stats = tr.groupby(c)["FPD_15"].agg(["sum", "count"])
        sum_map = tr[c].map(stats["sum"])
        cnt_map = tr[c].map(stats["count"])
        numer = (sum_map - y + smooth * gm)
        denom = (cnt_map - 1 + smooth)
        loo = numer / denom
        loo = loo.where(cnt_map > 1, gm)
        tr[f"{c}_te"] = loo.astype(np.float32)

        m = (stats["sum"] + smooth * gm) / (stats["count"] + smooth)
        col = f"{c}_te"
        va[col] = va[c].map(m).fillna(gm).astype(np.float32)
        te[col] = te[c].map(m).fillna(gm).astype(np.float32)
    return tr, va, te


def fit_te_full(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    te_cols: List[str],
    smooth: float = 200.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full-train TE for final training/prediction (no leakage risk)."""
    tr = train_df.copy()
    te = test_df.copy()
    gm = tr["FPD_15"].mean()
    for c in te_cols:
        stats = tr.groupby(c)["FPD_15"].agg(["sum", "count"])
        m = (stats["sum"] + smooth * gm) / (stats["count"] + smooth)
        col = f"{c}_te"
        tr[col] = tr[c].map(m).fillna(gm).astype(np.float32)
        te[col] = te[c].map(m).fillna(gm).astype(np.float32)
    return tr, te


def get_xgb_features(df: pd.DataFrame) -> List[str]:
    exclude = {
        "FPD_15",
        "label_matured",
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
        "CITY_NORM",
        "MERCHANTID",
        "CLERK_ID",
        "ADMINID",
        "USER_STATE",
        "MERCHANT_STATE",
    }
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def get_catboost_specs(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = [
        "COUNTRY_NORM",
        "STATE_NORM",
        "CITY_NORM",
        "MODEL_NORM",
        "MANUFACTURER_NORM",
        "LOCK_PRODUCT_NORM",
        "CURRENCY_NORM",
        "MERCHANTID",
        "CLERK_ID",
        "ADMINID",
        "USER_STATE_NORM",
        "MERCHANT_STATE_NORM",
    ]
    num_cols = [c for c in get_xgb_features(df) if not c.endswith("_te")]
    feature_cols = num_cols + cat_cols
    return feature_cols, cat_cols


def train_xgb(
    X_tr,
    y_tr,
    X_va,
    y_va,
    params: Dict,
    sample_weight=None,
    eval_sample_weight=None,
) -> Tuple[object, np.ndarray, float, int]:
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    m = xgb.XGBClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        gamma=params["gamma"],
        scale_pos_weight=spw,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        eval_metric="auc",
        early_stopping_rounds=params["early_stopping_rounds"],
    )
    fit_kwargs = {}
    if eval_sample_weight is not None:
        fit_kwargs["sample_weight_eval_set"] = [eval_sample_weight]
    m.fit(
        X_tr,
        y_tr,
        sample_weight=sample_weight,
        eval_set=[(X_va, y_va)],
        verbose=False,
        **fit_kwargs,
    )
    p = m.predict_proba(X_va)[:, 1]
    auc = float(roc_auc_score(y_va, p))
    best_iter = int(m.best_iteration if m.best_iteration is not None else params["n_estimators"] - 1)
    return m, p, auc, best_iter


def train_catboost(
    X_tr,
    y_tr,
    X_va,
    y_va,
    cat_cols: List[str],
    sample_weight=None,
    eval_weight=None,
):
    try:
        from catboost import CatBoostClassifier, Pool
    except Exception:
        return None, None, np.nan, None

    m = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=10.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        od_type="Iter",
        od_wait=150,
        thread_count=-1,
        verbose=False,
    )
    train_pool = Pool(X_tr, y_tr, cat_features=cat_cols, weight=sample_weight)
    eval_pool = Pool(X_va, y_va, cat_features=cat_cols, weight=eval_weight)
    m.fit(
        train_pool,
        eval_set=eval_pool,
        use_best_model=True,
        verbose=False,
    )
    p = m.predict_proba(X_va)[:, 1]
    auc = float(roc_auc_score(y_va, p))
    best_iter = int(m.get_best_iteration() or m.tree_count_)
    return m, p, auc, best_iter


def rolling_oof(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: RunConfig):
    log(f"Rolling OOF on months: {cfg.fold_months}")
    te_cols = [
        "COUNTRY_NORM",
        "STATE_NORM",
        "CITY_NORM",
        "MODEL_NORM",
        "MANUFACTURER_NORM",
        "LOCK_PRODUCT_NORM",
        "CURRENCY_NORM",
        "USER_STATE_NORM",
        "MERCHANT_STATE_NORM",
    ]

    oof_xgb = np.zeros(len(train_df), dtype=np.float32)
    oof_cat = np.zeros(len(train_df), dtype=np.float32)
    valid_mask = np.zeros(len(train_df), dtype=bool)

    xgb_models = []
    cat_models = []
    fold_report = []
    use_cat = cfg.use_catboost

    xgb_params = {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "max_depth": 6,
        "min_child_weight": 8,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "gamma": 0.1,
        "early_stopping_rounds": 150,
    }

    best_iters_xgb = []
    best_iters_cat = []

    for m in cfg.fold_months:
        tr_idx = np.where(train_df["tx_month"].values < m)[0]
        va_idx = np.where(train_df["tx_month"].values == m)[0]
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue

        tr = train_df.iloc[tr_idx].copy()
        va = train_df.iloc[va_idx].copy()
        w_tr = compute_time_weights(tr, cfg.min_train_month, cfg.max_train_month, cfg.time_weight_alpha)
        w_va = compute_time_weights(va, cfg.min_train_month, cfg.max_train_month, cfg.time_weight_alpha)
        tr, va = apply_currency_norm(tr, va)
        tr_te, te_va, _ = fit_te_loo(tr, va, test_df, te_cols=te_cols)

        xgb_cols = get_xgb_features(pd.concat([tr_te, te_va], axis=0))
        med = tr_te[xgb_cols].median()
        Xtr_x = tr_te[xgb_cols].fillna(med).astype(np.float32)
        Xva_x = te_va[xgb_cols].fillna(med).astype(np.float32)
        y_tr = tr["FPD_15"].astype(int).values
        y_va = va["FPD_15"].astype(int).values

        xgb_m, p_xgb, auc_xgb, best_iter = train_xgb(
            Xtr_x,
            y_tr,
            Xva_x,
            y_va,
            xgb_params,
            sample_weight=w_tr,
            eval_sample_weight=w_va,
        )
        oof_xgb[va_idx] = p_xgb
        xgb_models.append((xgb_m, xgb_cols, med))
        best_iters_xgb.append(best_iter)

        auc_cat = np.nan
        if use_cat:
            cat_cols_all, cat_cols = get_catboost_specs(pd.concat([tr, va], axis=0))
            tr_cat = tr[cat_cols_all].copy()
            va_cat = va[cat_cols_all].copy()
            for c in cat_cols:
                tr_cat[c] = tr_cat[c].astype(str).fillna("NA")
                va_cat[c] = va_cat[c].astype(str).fillna("NA")
            cat_m, p_cat, auc_cat, best_iter_cat = train_catboost(
                tr_cat,
                y_tr,
                va_cat,
                y_va,
                cat_cols=cat_cols,
                sample_weight=w_tr,
                eval_weight=w_va,
            )
            if cat_m is None:
                use_cat = False
            else:
                oof_cat[va_idx] = p_cat
                cat_models.append((cat_m, cat_cols_all, cat_cols))
                best_iters_cat.append(best_iter_cat)

        valid_mask[va_idx] = True
        fold_report.append({"month": int(m), "xgb_auc": auc_xgb, "cat_auc": auc_cat})
        log(f"Fold m={m}: XGB={auc_xgb:.5f} | CAT={auc_cat if np.isnan(auc_cat) else round(auc_cat,5)}")

    y_oof = train_df.loc[valid_mask, "FPD_15"].astype(int).values
    p_xgb = oof_xgb[valid_mask]
    auc_xgb = float(roc_auc_score(y_oof, p_xgb))
    if use_cat and len(cat_models) > 0:
        p_cat = oof_cat[valid_mask]
        auc_cat = float(roc_auc_score(y_oof, p_cat))
        best_w, best_auc = 1.0, auc_xgb
        for w in np.arange(0.0, 1.01, 0.05):
            p = w * p_xgb + (1.0 - w) * p_cat
            a = float(roc_auc_score(y_oof, p))
            if a > best_auc:
                best_auc, best_w = a, float(w)
    else:
        auc_cat = np.nan
        best_w, best_auc = 1.0, auc_xgb
        use_cat = False

    return {
        "fold_report": fold_report,
        "valid_mask": valid_mask,
        "oof_xgb": oof_xgb,
        "oof_cat": oof_cat,
        "use_cat": use_cat,
        "xgb_models": xgb_models,
        "cat_models": cat_models,
        "blend_w_xgb": best_w,
        "oof_auc_xgb": auc_xgb,
        "oof_auc_cat": auc_cat,
        "oof_auc_blend": best_auc,
        "best_iters_xgb": best_iters_xgb,
        "best_iters_cat": best_iters_cat,
    }


def final_train_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_cat: bool,
    blend_w_xgb: float,
    xgb_seeds: Tuple[int, ...],
    cat_seeds: Tuple[int, ...],
    best_iters_xgb: List[int],
    best_iters_cat: List[int],
    time_weight_alpha: float,
    min_train_month: int,
    max_train_month: int,
):
    te_cols = [
        "COUNTRY_NORM",
        "STATE_NORM",
        "CITY_NORM",
        "MODEL_NORM",
        "MANUFACTURER_NORM",
        "LOCK_PRODUCT_NORM",
        "CURRENCY_NORM",
        "USER_STATE_NORM",
        "MERCHANT_STATE_NORM",
    ]
    train_for_xgb, test_for_xgb = fit_te_full(train_df, test_df, te_cols=te_cols)

    xgb_cols = get_xgb_features(pd.concat([train_for_xgb, test_for_xgb], axis=0))
    med = train_for_xgb[xgb_cols].median()
    Xtr_x = train_for_xgb[xgb_cols].fillna(med).astype(np.float32)
    Xte_x = test_for_xgb[xgb_cols].fillna(med).astype(np.float32)
    y_tr = train_df["FPD_15"].astype(int).values

    n_estimators = int(np.mean(best_iters_xgb) + 150) if best_iters_xgb else 2400
    w_full = compute_time_weights(train_df, min_train_month, max_train_month, time_weight_alpha)
    xgb_preds = []
    for seed in xgb_seeds:
        xgb_m = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=0.02,
            max_depth=6,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=2.0,
            gamma=0.1,
            scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        xgb_m.fit(Xtr_x, y_tr, sample_weight=w_full, verbose=False)
        xgb_preds.append(xgb_m.predict_proba(Xte_x)[:, 1])
    p_xgb = np.mean(xgb_preds, axis=0)

    if use_cat:
        cat_cols_all, cat_cols = get_catboost_specs(pd.concat([train_df, test_df], axis=0))
        tr_cat = train_df[cat_cols_all].copy()
        te_cat = test_df[cat_cols_all].copy()
        for c in cat_cols:
            tr_cat[c] = tr_cat[c].astype(str).fillna("NA")
            te_cat[c] = te_cat[c].astype(str).fillna("NA")
        try:
            from catboost import CatBoostClassifier, Pool
        except Exception:
            use_cat = False
            p_cons = p_xgb
            p_aggr = p_xgb
            p_pure_cat = p_xgb
            p_cat_ens = p_xgb
            p_lb = p_xgb
        else:
            n_iter_base = int(np.mean(best_iters_cat) + 200) if best_iters_cat else 3000
            n_iter_tuned = int(np.mean(best_iters_cat) + 300) if best_iters_cat else 3000

            # Optional month-11 holdout for early-stopping calibration
            holdout_mask = train_df["tx_month"].values == max_train_month
            use_holdout = holdout_mask.sum() > 0

            def prep_cat(df_in: pd.DataFrame) -> pd.DataFrame:
                out = df_in[cat_cols_all].copy()
                for c in cat_cols:
                    out[c] = out[c].astype(str).fillna("NA")
                return out

            def estimate_best_iter(params: Dict):
                if not use_holdout:
                    return params["iterations"]
                tr_sub = train_df.loc[~holdout_mask].copy()
                va_sub = train_df.loc[holdout_mask].copy()
                tr_sub, va_sub = apply_currency_norm(tr_sub, va_sub)
                w_tr = compute_time_weights(tr_sub, min_train_month, max_train_month, time_weight_alpha)
                w_va = compute_time_weights(va_sub, min_train_month, max_train_month, time_weight_alpha)
                X_tr = prep_cat(tr_sub)
                X_va = prep_cat(va_sub)
                y_tr = tr_sub["FPD_15"].astype(int).values
                y_va = va_sub["FPD_15"].astype(int).values
                m = CatBoostClassifier(
                    iterations=params["iterations"],
                    learning_rate=params["learning_rate"],
                    depth=params["depth"],
                    l2_leaf_reg=params["l2_leaf_reg"],
                    loss_function="Logloss",
                    eval_metric="AUC",
                    random_seed=SEED,
                    od_type="Iter",
                    od_wait=150,
                    thread_count=-1,
                    verbose=False,
                    **params.get("extra", {}),
                )
                train_pool = Pool(X_tr, y_tr, cat_features=cat_cols, weight=w_tr)
                eval_pool = Pool(X_va, y_va, cat_features=cat_cols, weight=w_va)
                m.fit(
                    train_pool,
                    eval_set=eval_pool,
                    use_best_model=True,
                    verbose=False,
                )
                best_iter = int(m.get_best_iteration() or m.tree_count_)
                return max(best_iter + 50, 200)

            def fit_cat_preds(params: Dict, seeds: Tuple[int, ...], iterations: int) -> np.ndarray:
                preds = []
                for seed in seeds:
                    log(f"CatBoost fit start | seed={seed} | iters={iterations} | depth={params['depth']}")
                    m = CatBoostClassifier(
                        iterations=iterations,
                        learning_rate=params["learning_rate"],
                        depth=params["depth"],
                        l2_leaf_reg=params["l2_leaf_reg"],
                        loss_function="Logloss",
                        eval_metric="AUC",
                        random_seed=seed,
                        thread_count=-1,
                        verbose=200,
                        **params.get("extra", {}),
                    )
                    m.fit(tr_cat, y_tr, cat_features=cat_cols, sample_weight=w_full, verbose=False)
                    preds.append(m.predict_proba(te_cat)[:, 1])
                    log(f"CatBoost fit done  | seed={seed}")
                return np.mean(preds, axis=0)

            base_params = {
                "iterations": n_iter_base,
                "learning_rate": 0.03,
                "depth": 8,
                "l2_leaf_reg": 10.0,
            }
            tuned_params = {
                "iterations": n_iter_tuned,
                "learning_rate": 0.03,
                "depth": 9,
                "l2_leaf_reg": 12.0,
                "extra": {
                    "rsm": 0.8,
                    "bagging_temperature": 0.5,
                    "random_strength": 0.5,
                    "min_data_in_leaf": 50,
                    "border_count": 128,
                },
            }

            best_iter_base = estimate_best_iter(base_params)
            best_iter_tuned = estimate_best_iter(tuned_params)

            p_pure_cat = fit_cat_preds(base_params, (SEED,), best_iter_base)
            p_cat_ens = fit_cat_preds(base_params, cat_seeds, best_iter_base)
            p_cat_tuned_ens = fit_cat_preds(tuned_params, cat_seeds, best_iter_tuned)

            p_cons = blend_w_xgb * p_xgb + (1.0 - blend_w_xgb) * p_cat_ens
            p_aggr = 0.6 * p_xgb + 0.4 * p_cat_ens
            p_lb = 0.9 * p_cat_tuned_ens + 0.1 * p_xgb
    else:
        p_cons = p_xgb
        p_aggr = p_xgb
        p_pure_cat = p_xgb
        p_cat_ens = p_xgb
        p_lb = p_xgb

    return {
        "conservative": p_cons,
        "aggressive": p_aggr,
        "pure_cat": p_pure_cat,
        "cat_ensemble": p_cat_ens,
        "lb_leaning": p_lb,
    }


def save_submission(path: str, ids: np.ndarray, pred: np.ndarray):
    sub = pd.DataFrame({"FINANCEORDERID": ids, "FPD_15_pred": pred})
    sub = sub.sort_values("FINANCEORDERID").reset_index(drop=True)
    sub.to_csv(path, index=False)


def run_ab(cfg: RunConfig):
    log("A/B runner: comparing time-weighted vs unweighted training.")
    orders, test_ids = load_data()
    orders = prepare_labels(orders, cfg.snapshot_date)
    orders = add_safe_entity_history(orders)
    orders = add_features(orders)

    train_df, test_df = make_train_test(orders, test_ids, cfg)
    train_df, test_df = apply_currency_norm(train_df, test_df)

    variants = [
        ("A_no_time_weight", 0.0),
        ("B_time_weight", cfg.time_weight_alpha),
    ]
    rows = []
    for name, alpha in variants:
        cfg_v = RunConfig(
            min_train_month=cfg.min_train_month,
            max_train_month=cfg.max_train_month,
            fold_months=cfg.fold_months,
            snapshot_date=cfg.snapshot_date,
            remove_sideload=cfg.remove_sideload,
            use_catboost=cfg.use_catboost,
            xgb_seeds=cfg.xgb_seeds,
            cat_seeds=cfg.cat_seeds,
            time_weight_alpha=alpha,
        )
        log(f"A/B variant: {name} (alpha={alpha})")
        res = rolling_oof(train_df, test_df, cfg_v)
        rows.append(
            {
                "variant": name,
                "alpha": alpha,
                "oof_auc_xgb": res["oof_auc_xgb"],
                "oof_auc_cat": None if np.isnan(res["oof_auc_cat"]) else res["oof_auc_cat"],
                "oof_auc_blend": res["oof_auc_blend"],
                "blend_w_xgb": res["blend_w_xgb"],
            }
        )

    report = pd.DataFrame(rows)
    print(report)
    report.to_csv("ab_summary.csv", index=False)
    log("Saved ab_summary.csv")


def main():
    cfg = parse_args()
    log(f"Config: {cfg}")
    if cfg.ab:
        run_ab(cfg)
        return
    orders, test_ids = load_data()
    orders = prepare_labels(orders, cfg.snapshot_date)
    orders = add_safe_entity_history(orders)
    orders = add_features(orders)

    train_df, test_df = make_train_test(orders, test_ids, cfg)
    train_df, test_df = apply_currency_norm(train_df, test_df)
    log(f"Training rows (maturity-clean): {len(train_df):,} | Test rows: {len(test_df):,}")

    result = rolling_oof(train_df, test_df, cfg)
    log(
        f"OOF AUC => XGB={result['oof_auc_xgb']:.5f} | "
        f"CAT={result['oof_auc_cat'] if np.isnan(result['oof_auc_cat']) else round(result['oof_auc_cat'],5)} | "
        f"Blend={result['oof_auc_blend']:.5f} (w_xgb={result['blend_w_xgb']:.2f})"
    )

    preds = final_train_predict(
        train_df,
        test_df,
        use_cat=result["use_cat"],
        blend_w_xgb=result["blend_w_xgb"],
        xgb_seeds=cfg.xgb_seeds,
        cat_seeds=cfg.cat_seeds,
        best_iters_xgb=result["best_iters_xgb"],
        best_iters_cat=result["best_iters_cat"],
        time_weight_alpha=cfg.time_weight_alpha,
        min_train_month=cfg.min_train_month,
        max_train_month=cfg.max_train_month,
    )

    save_submission("final_submission_best_conservative.csv", test_df["FINANCEORDERID"].values, preds["conservative"])
    save_submission("final_submission_best_aggressive.csv", test_df["FINANCEORDERID"].values, preds["aggressive"])
    save_submission("final_submission_best_pure_cat.csv", test_df["FINANCEORDERID"].values, preds["pure_cat"])
    save_submission("final_submission_best_cat_ensemble.csv", test_df["FINANCEORDERID"].values, preds["cat_ensemble"])
    save_submission("final_submission_best_lb_leaning.csv", test_df["FINANCEORDERID"].values, preds["lb_leaning"])
    # Final pick based on observed LB performance
    save_submission("final_submission_final.csv", test_df["FINANCEORDERID"].values, preds["cat_ensemble"])

    summary = {
        "config": cfg.__dict__,
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "fold_report": result["fold_report"],
        "oof_auc_xgb": result["oof_auc_xgb"],
        "oof_auc_cat": None if np.isnan(result["oof_auc_cat"]) else result["oof_auc_cat"],
        "oof_auc_blend": result["oof_auc_blend"],
        "blend_w_xgb": result["blend_w_xgb"],
        "use_cat": result["use_cat"],
        "best_iters_xgb": result["best_iters_xgb"],
        "best_iters_cat": result["best_iters_cat"],
        "lb_leaning_xgb_weight": 0.1,
        "time_weight_alpha": cfg.time_weight_alpha,
        "final_choice": "cat_ensemble",
    }
    with open("best_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log("Saved final_submission_best_conservative.csv")
    log("Saved final_submission_best_aggressive.csv")
    log("Saved final_submission_final.csv")
    log("Saved best_summary.json")


if __name__ == "__main__":
    main()
