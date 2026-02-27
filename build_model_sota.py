"""
SOTA-style leakage-safe pipeline for PayJoy FPD challenge.

Highlights:
- Maturity-clean labels (only train rows with observable FPD by snapshot date)
- Rolling out-of-time validation (months 9/10/11 by default)
- CatBoost primary model on raw categoricals
- XGBoost companion model on numeric + time-safe target encodings
- OOF-based blend weight search
- Final outputs: conservative and aggressive submissions
"""

import os
import json
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
    remove_sideload: bool = True
    snapshot_date: str = "2025-12-01T00:00:00Z"
    use_catboost: bool = True


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--min-train-month", type=int, default=6)
    p.add_argument("--max-train-month", type=int, default=11)
    p.add_argument("--fold-months", type=str, default="9,10,11")
    p.add_argument("--snapshot-date", type=str, default="2025-12-01T00:00:00Z")
    p.add_argument("--keep-sideload", action="store_true")
    p.add_argument("--no-catboost", action="store_true")
    a = p.parse_args()
    fold_months = tuple(int(x) for x in a.fold_months.split(",") if x.strip())
    return RunConfig(
        min_train_month=a.min_train_month,
        max_train_month=a.max_train_month,
        fold_months=fold_months,
        remove_sideload=not a.keep_sideload,
        snapshot_date=a.snapshot_date,
        use_catboost=not a.no_catboost,
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
    ]:
        orders[c] = pd.to_datetime(orders[c], utc=True)
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


def add_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("TRANSACTIONTIME").reset_index(drop=True).copy()

    # Entity volume/tenure (safe at order time)
    for entity, prefix in [("MERCHANTID", "merch"), ("CLERK_ID", "clerk"), ("ADMINID", "admin")]:
        g = out.groupby(entity, sort=False)
        out[f"{prefix}_cum_orders"] = g.cumcount().astype(np.int32)
        out[f"{prefix}_first_tx"] = g["TRANSACTIONTIME"].transform("min")
        out[f"{prefix}_age_days"] = (
            (out["TRANSACTIONTIME"] - out[f"{prefix}_first_tx"]).dt.total_seconds() / 86400.0
        )
        out[f"{prefix}_order_velocity"] = out[f"{prefix}_cum_orders"] / (out[f"{prefix}_age_days"] + 1.0)
        out.drop(columns=[f"{prefix}_first_tx"], inplace=True)

    # Time/order features
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

    # Currency-robust ratios + normalized amounts
    out["down_pmt_ratio"] = out["DOWN_PAYMENT_AMOUNT"] / (out["PURCHASE_AMOUNT"] + 1.0)
    out["finance_pmt_ratio"] = out["FINANCE_AMOUNT"] / (out["PURCHASE_AMOUNT"] + 1.0)
    out["total_due_ratio"] = out["TOTAL_DUE"] / (out["PURCHASE_AMOUNT"] + 1.0)
    out["finance_down_ratio"] = out["FINANCE_AMOUNT"] / (out["DOWN_PAYMENT_AMOUNT"] + 1.0)
    out["interest_ratio"] = (out["TOTAL_DUE"] - out["FINANCE_AMOUNT"]) / (out["FINANCE_AMOUNT"] + 1.0)

    for c in ["FINANCE_AMOUNT", "PURCHASE_AMOUNT", "TOTAL_DUE", "DOWN_PAYMENT_AMOUNT"]:
        out[f"log_{c.lower()}"] = np.log1p(out[c].clip(lower=0))
        med = out.groupby("CURRENCY")[c].transform("median")
        out[f"{c.lower()}_ccy_norm"] = out[c] / (med + 1.0)

    # Verification scores (weak signal but keep for model to decide)
    sc = ["FACE_RECOGNITION_SCORE", "IDVALIDATION_OVERALL_SCORE", "LIVENESS_SCORE", "OVERALL_SCORE"]
    out["score_mean"] = out[sc].mean(axis=1)
    out["score_min"] = out[sc].min(axis=1)
    out["score_std"] = out[sc].std(axis=1)
    return out


def make_train_test(
    df: pd.DataFrame, test_ids: pd.DataFrame, cfg: RunConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    is_test = df["FINANCEORDERID"].isin(set(test_ids["FINANCEORDERID"]))
    train_mask = (
        df["label_matured"]
        & df["tx_month"].between(cfg.min_train_month, cfg.max_train_month)
    )
    if cfg.remove_sideload:
        bad = df["LOCK_NAME"].str.contains("SideLoad|KnoxHack", case=False, na=False)
        train_mask = train_mask & (~bad)
    out = df[train_mask | is_test].copy()
    train_df = out[out["FPD_15"].notna() & out["label_matured"]].copy()
    test_df = out[out["FINANCEORDERID"].isin(set(test_ids["FINANCEORDERID"]))].copy()
    return train_df, test_df


def fit_monthwise_te(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    te_cols: List[str],
    smooth: float = 200.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit TE on train only, apply to val/test (for XGB numeric branch)."""
    tr = train_df.copy()
    va = val_df.copy()
    te = test_df.copy()
    gm = tr["FPD_15"].mean()
    for c in te_cols:
        s = tr.groupby(c)["FPD_15"].agg(["mean", "count"])
        m = (s["count"] * s["mean"] + smooth * gm) / (s["count"] + smooth)
        col = f"{c}_te"
        va[col] = va[c].map(m).fillna(gm).astype(np.float32)
        te[col] = te[c].map(m).fillna(gm).astype(np.float32)
    return va, te


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
        "MERCHANTID",
        "CLERK_ID",
        "ADMINID",
        "USER_STATE",
        "MERCHANT_STATE",
    }
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def get_catboost_specs(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = [
        "COUNTRY",
        "STATE",
        "CITY",
        "MODEL",
        "MANUFACTURER",
        "LOCK_PRODUCT",
        "CURRENCY",
        "MERCHANTID",
        "CLERK_ID",
        "ADMINID",
    ]
    num_cols = [c for c in get_xgb_features(df) if not c.endswith("_te")]
    feature_cols = num_cols + cat_cols
    return feature_cols, cat_cols


def train_xgb(X_tr, y_tr, X_va, y_va) -> Tuple[object, np.ndarray, float]:
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    m = xgb.XGBClassifier(
        n_estimators=2400,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=25,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=6.0,
        gamma=0.2,
        scale_pos_weight=spw,
        tree_method="hist",
        random_state=SEED,
        n_jobs=1,
        eval_metric="auc",
        early_stopping_rounds=120,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    p = m.predict_proba(X_va)[:, 1]
    auc = float(roc_auc_score(y_va, p))
    return m, p, auc


def train_catboost(X_tr, y_tr, X_va, y_va, cat_cols: List[str]):
    try:
        from catboost import CatBoostClassifier
    except Exception:
        return None, None, np.nan

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
        verbose=False,
    )
    m.fit(
        X_tr,
        y_tr,
        cat_features=cat_cols,
        eval_set=(X_va, y_va),
        use_best_model=True,
        verbose=False,
    )
    p = m.predict_proba(X_va)[:, 1]
    auc = float(roc_auc_score(y_va, p))
    return m, p, auc


def rolling_oof(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: RunConfig):
    log(f"Rolling OOF on months: {cfg.fold_months}")
    te_cols = ["COUNTRY", "STATE", "CITY", "MODEL", "MANUFACTURER", "LOCK_PRODUCT", "CURRENCY"]

    oof_xgb = np.zeros(len(train_df), dtype=np.float32)
    oof_cat = np.zeros(len(train_df), dtype=np.float32)
    valid_mask = np.zeros(len(train_df), dtype=bool)

    xgb_models = []
    cat_models = []
    fold_report = []
    use_cat = cfg.use_catboost

    for m in cfg.fold_months:
        tr_idx = np.where(train_df["tx_month"].values < m)[0]
        va_idx = np.where(train_df["tx_month"].values == m)[0]
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue

        tr = train_df.iloc[tr_idx].copy()
        va = train_df.iloc[va_idx].copy()
        te_va, _ = fit_monthwise_te(tr, va, test_df, te_cols=te_cols)
        tr_te, _ = fit_monthwise_te(tr, tr, test_df, te_cols=te_cols)

        xgb_cols = get_xgb_features(pd.concat([tr_te, te_va], axis=0))
        med = tr_te[xgb_cols].median()
        Xtr_x = tr_te[xgb_cols].fillna(med).astype(np.float32)
        Xva_x = te_va[xgb_cols].fillna(med).astype(np.float32)
        y_tr = tr["FPD_15"].astype(int).values
        y_va = va["FPD_15"].astype(int).values

        xgb_m, p_xgb, auc_xgb = train_xgb(Xtr_x, y_tr, Xva_x, y_va)
        oof_xgb[va_idx] = p_xgb
        xgb_models.append((xgb_m, xgb_cols, med))

        auc_cat = np.nan
        if use_cat:
            cat_cols_all, cat_cols = get_catboost_specs(pd.concat([tr, va], axis=0))
            tr_cat = tr[cat_cols_all].copy()
            va_cat = va[cat_cols_all].copy()
            for c in cat_cols:
                tr_cat[c] = tr_cat[c].astype(str).fillna("NA")
                va_cat[c] = va_cat[c].astype(str).fillna("NA")
            cat_m, p_cat, auc_cat = train_catboost(tr_cat, y_tr, va_cat, y_va, cat_cols=cat_cols)
            if cat_m is None:
                use_cat = False
            else:
                oof_cat[va_idx] = p_cat
                cat_models.append((cat_m, cat_cols_all, cat_cols))

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
    }


def final_train_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, use_cat: bool, blend_w_xgb: float):
    te_cols = ["COUNTRY", "STATE", "CITY", "MODEL", "MANUFACTURER", "LOCK_PRODUCT", "CURRENCY"]
    test_for_xgb, _ = fit_monthwise_te(train_df, test_df, test_df, te_cols=te_cols)
    train_for_xgb, _ = fit_monthwise_te(train_df, train_df, test_df, te_cols=te_cols)

    xgb_cols = get_xgb_features(pd.concat([train_for_xgb, test_for_xgb], axis=0))
    med = train_for_xgb[xgb_cols].median()
    Xtr_x = train_for_xgb[xgb_cols].fillna(med).astype(np.float32)
    Xte_x = test_for_xgb[xgb_cols].fillna(med).astype(np.float32)
    y_tr = train_df["FPD_15"].astype(int).values
    xgb_m, _, _ = train_xgb(Xtr_x, y_tr, Xtr_x.iloc[: min(50000, len(Xtr_x))], y_tr[: min(50000, len(y_tr))])
    p_xgb = xgb_m.predict_proba(Xte_x)[:, 1]

    if use_cat:
        cat_cols_all, cat_cols = get_catboost_specs(pd.concat([train_df, test_df], axis=0))
        tr_cat = train_df[cat_cols_all].copy()
        te_cat = test_df[cat_cols_all].copy()
        for c in cat_cols:
            tr_cat[c] = tr_cat[c].astype(str).fillna("NA")
            te_cat[c] = te_cat[c].astype(str).fillna("NA")
        cat_m, _, _ = train_catboost(tr_cat, y_tr, tr_cat.iloc[: min(50000, len(tr_cat))], y_tr[: min(50000, len(y_tr))], cat_cols)
        if cat_m is not None:
            p_cat = cat_m.predict_proba(te_cat)[:, 1]
            p_cons = blend_w_xgb * p_xgb + (1.0 - blend_w_xgb) * p_cat
            p_aggr = 0.6 * p_xgb + 0.4 * p_cat
        else:
            p_cons = p_xgb
            p_aggr = p_xgb
    else:
        p_cons = p_xgb
        p_aggr = p_xgb

    return p_cons, p_aggr


def save_submission(path: str, ids: np.ndarray, pred: np.ndarray):
    sub = pd.DataFrame({"FINANCEORDERID": ids, "FPD_15_pred": pred})
    sub = sub.sort_values("FINANCEORDERID").reset_index(drop=True)
    sub.to_csv(path, index=False)


def main():
    cfg = parse_args()
    log(f"Config: {cfg}")
    orders, test_ids = load_data()
    orders = prepare_labels(orders, cfg.snapshot_date)
    orders = add_safe_features(orders)

    train_df, test_df = make_train_test(orders, test_ids, cfg)
    log(f"Training rows (maturity-clean): {len(train_df):,} | Test rows: {len(test_df):,}")

    result = rolling_oof(train_df, test_df, cfg)
    log(
        f"OOF AUC => XGB={result['oof_auc_xgb']:.5f} | "
        f"CAT={result['oof_auc_cat'] if np.isnan(result['oof_auc_cat']) else round(result['oof_auc_cat'],5)} | "
        f"Blend={result['oof_auc_blend']:.5f} (w_xgb={result['blend_w_xgb']:.2f})"
    )

    p_cons, p_aggr = final_train_predict(
        train_df,
        test_df,
        use_cat=result["use_cat"],
        blend_w_xgb=result["blend_w_xgb"],
    )

    save_submission("final_submission_sota_conservative.csv", test_df["FINANCEORDERID"].values, p_cons)
    save_submission("final_submission_sota_aggressive.csv", test_df["FINANCEORDERID"].values, p_aggr)

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
    }
    with open("sota_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log("Saved final_submission_sota_conservative.csv")
    log("Saved final_submission_sota_aggressive.csv")
    log("Saved sota_summary.json")


if __name__ == "__main__":
    main()
