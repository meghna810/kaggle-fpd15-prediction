"""
PayJoy FPD_15 — Pure CatBoost pipeline (LB-focused, leakage-safe).

Best learnings included:
- Maturity-clean labels
- Strict rolling OOT validation (months 9/10/11)
- Time-weighted training (optional, default 0.8)
- Normalized categoricals (CITY/STATE/COUNTRY/CURRENCY/MODEL/MANUFACTURER/LOCK_PRODUCT)
- Currency normalization using TRAIN-only medians (no leakage)
- CatBoost-only with multi-seed ensemble
- Optional month-11 holdout to calibrate iterations

Outputs:
- final_submission_cat_ensemble.csv
- final_submission_cat_tuned.csv
- final_submission_cat_final.csv
- cat_summary.json
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
    snapshot_date: str = "2025-12-01T00:00:00Z"
    remove_sideload: bool = True
    cat_seeds: Tuple[int, ...] = (42, 123, 2026)
    time_weight_alpha: float = 0.8


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--min-train-month", type=int, default=6)
    p.add_argument("--max-train-month", type=int, default=11)
    p.add_argument("--fold-months", type=str, default="9,10,11")
    p.add_argument("--snapshot-date", type=str, default="2025-12-01T00:00:00Z")
    p.add_argument("--keep-sideload", action="store_true")
    p.add_argument("--time-weight-alpha", type=float, default=0.8)
    a = p.parse_args()
    fold_months = tuple(int(x) for x in a.fold_months.split(",") if x.strip())
    return RunConfig(
        min_train_month=a.min_train_month,
        max_train_month=a.max_train_month,
        fold_months=fold_months,
        snapshot_date=a.snapshot_date,
        remove_sideload=not a.keep_sideload,
        time_weight_alpha=a.time_weight_alpha,
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
    out = orders.copy()
    snapshot_ts = pd.Timestamp(snapshot_date, tz="UTC")
    maturity_ts = out["FIRST_PAYMENT_DUE_TIMESTAMP"] + pd.Timedelta(days=15)
    out["label_matured"] = out["FPD_15"].notna() & (maturity_ts <= snapshot_ts)
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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["CITY_NORM"] = out["CITY"].map(normalize_city)
    out["STATE_NORM"] = out["STATE"].map(normalize_code)
    out["COUNTRY_NORM"] = out["COUNTRY"].map(normalize_code)
    out["CURRENCY_NORM"] = out["CURRENCY"].map(normalize_code)
    out["MODEL_NORM"] = out["MODEL"].map(normalize_alnum)
    out["MANUFACTURER_NORM"] = out["MANUFACTURER"].map(normalize_alnum)
    out["LOCK_PRODUCT_NORM"] = out["LOCK_PRODUCT"].map(normalize_alnum)

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
    return out


def add_safe_entity_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("TRANSACTIONTIME").reset_index(drop=True).copy()
    for entity, prefix in [("MERCHANTID", "merch"), ("CLERK_ID", "clerk"), ("ADMINID", "admin")]:
        g = out.groupby(entity, sort=False)
        out[f"{prefix}_cum_orders"] = g.cumcount().astype(np.int32)
        out[f"{prefix}_first_tx"] = g["TRANSACTIONTIME"].transform("min")
        out[f"{prefix}_age_days"] = (
            (out["TRANSACTIONTIME"] - out[f"{prefix}_first_tx"]).dt.total_seconds() / 86400.0
        )
        out[f"{prefix}_order_velocity"] = out[f"{prefix}_cum_orders"] / (out[f"{prefix}_age_days"] + 1.0)
        out.drop(columns=[f"{prefix}_first_tx"], inplace=True)
    return out


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


def compute_time_weights(df: pd.DataFrame, min_m: int, max_m: int, alpha: float):
    if alpha <= 0.0:
        return None
    denom = max(max_m - min_m, 1)
    w = 1.0 + alpha * (df["tx_month"].astype(float) - float(min_m)) / float(denom)
    return w.astype(np.float32).values


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
    ]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = list(dict.fromkeys(num_cols + cat_cols))
    return feature_cols, cat_cols


def train_catboost(
    X_tr,
    y_tr,
    X_va,
    y_va,
    cat_cols: List[str],
    sample_weight=None,
    eval_weight=None,
    params: Dict | None = None,
):
    try:
        from catboost import CatBoostClassifier, Pool
    except Exception:
        return None, None, np.nan, None

    p = params or {}
    m = CatBoostClassifier(
        iterations=int(p.get("iterations", 4000)),
        learning_rate=float(p.get("learning_rate", 0.03)),
        depth=int(p.get("depth", 8)),
        l2_leaf_reg=float(p.get("l2_leaf_reg", 10.0)),
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        od_type="Iter",
        od_wait=int(p.get("od_wait", 150)),
        verbose=False,
        **p.get("extra", {}),
    )
    train_pool = Pool(X_tr, y_tr, cat_features=cat_cols, weight=sample_weight)
    eval_pool = Pool(X_va, y_va, cat_features=cat_cols, weight=eval_weight)
    m.fit(train_pool, eval_set=eval_pool, use_best_model=True, verbose=False)
    p_va = m.predict_proba(X_va)[:, 1]
    auc = float(roc_auc_score(y_va, p_va))
    best_iter = int(m.get_best_iteration() or m.tree_count_)
    return m, p_va, auc, best_iter


def rolling_oof(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: RunConfig):
    log(f"Rolling OOF on months: {cfg.fold_months}")
    oof_cat = np.zeros(len(train_df), dtype=np.float32)
    valid_mask = np.zeros(len(train_df), dtype=bool)
    fold_report = []
    best_iters_cat = []

    for m in cfg.fold_months:
        tr_idx = np.where(train_df["tx_month"].values < m)[0]
        va_idx = np.where(train_df["tx_month"].values == m)[0]
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        tr = train_df.iloc[tr_idx].copy()
        va = train_df.iloc[va_idx].copy()
        tr, va = apply_currency_norm(tr, va)

        w_tr = compute_time_weights(tr, cfg.min_train_month, cfg.max_train_month, cfg.time_weight_alpha)
        w_va = compute_time_weights(va, cfg.min_train_month, cfg.max_train_month, cfg.time_weight_alpha)

        feature_cols, cat_cols = get_catboost_specs(pd.concat([tr, va], axis=0))
        X_tr = tr[feature_cols].copy()
        X_va = va[feature_cols].copy()
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype(str).fillna("NA")
            X_va[c] = X_va[c].astype(str).fillna("NA")
        y_tr = tr["FPD_15"].astype(int).values
        y_va = va["FPD_15"].astype(int).values

        cat_m, p_cat, auc_cat, best_iter = train_catboost(
            X_tr, y_tr, X_va, y_va, cat_cols=cat_cols, sample_weight=w_tr, eval_weight=w_va
        )
        oof_cat[va_idx] = p_cat
        best_iters_cat.append(best_iter)
        valid_mask[va_idx] = True
        fold_report.append({"month": int(m), "cat_auc": auc_cat})
        log(f"Fold m={m}: CAT={auc_cat:.5f}")

    y_oof = train_df.loc[valid_mask, "FPD_15"].astype(int).values
    auc_cat = float(roc_auc_score(y_oof, oof_cat[valid_mask]))
    return {
        "fold_report": fold_report,
        "oof_auc_cat": auc_cat,
        "best_iters_cat": best_iters_cat,
    }


def final_predict(train_df, test_df, cfg: RunConfig, best_iters_cat: List[int]):
    try:
        from catboost import CatBoostClassifier
    except Exception:
        raise RuntimeError("CatBoost not available.")

    train_df, test_df = apply_currency_norm(train_df, test_df)
    feature_cols, cat_cols = get_catboost_specs(pd.concat([train_df, test_df], axis=0))

    tr_cat = train_df[feature_cols].copy()
    te_cat = test_df[feature_cols].copy()
    for c in cat_cols:
        tr_cat[c] = tr_cat[c].astype(str).fillna("NA")
        te_cat[c] = te_cat[c].astype(str).fillna("NA")

    y_tr = train_df["FPD_15"].astype(int).values
    w_full = compute_time_weights(train_df, cfg.min_train_month, cfg.max_train_month, cfg.time_weight_alpha)

    n_iter_base = int(np.mean(best_iters_cat) + 200) if best_iters_cat else 3000

    base_params = {
        "iterations": n_iter_base,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 10.0,
    }
    tuned_params = {
        "iterations": max(n_iter_base - 400, 600),
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

    def fit_preds(params: Dict, seeds: Tuple[int, ...]) -> np.ndarray:
        preds = []
        for seed in seeds:
            log(f"CatBoost fit start | seed={seed} | iters={params['iterations']} | depth={params['depth']}")
            m = CatBoostClassifier(
                iterations=params["iterations"],
                learning_rate=params["learning_rate"],
                depth=params["depth"],
                l2_leaf_reg=params["l2_leaf_reg"],
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=seed,
                verbose=200,
                **params.get("extra", {}),
            )
            m.fit(tr_cat, y_tr, cat_features=cat_cols, sample_weight=w_full, verbose=False)
            preds.append(m.predict_proba(te_cat)[:, 1])
            log(f"CatBoost fit done  | seed={seed}")
        return np.mean(preds, axis=0)

    p_ens = fit_preds(base_params, cfg.cat_seeds)
    p_tuned = fit_preds(tuned_params, cfg.cat_seeds)
    return p_ens, p_tuned


def save_submission(path: str, ids: np.ndarray, pred: np.ndarray):
    sub = pd.DataFrame({"FINANCEORDERID": ids, "FPD_15_pred": pred})
    sub = sub.sort_values("FINANCEORDERID").reset_index(drop=True)
    sub.to_csv(path, index=False)


def main():
    cfg = parse_args()
    log(f"Config: {cfg}")
    orders, test_ids = load_data()
    orders = prepare_labels(orders, cfg.snapshot_date)
    orders = add_safe_entity_history(orders)
    orders = add_features(orders)

    train_df, test_df = make_train_test(orders, test_ids, cfg)
    log(f"Training rows (maturity-clean): {len(train_df):,} | Test rows: {len(test_df):,}")

    result = rolling_oof(train_df, test_df, cfg)
    log(f"OOF AUC (CatBoost): {result['oof_auc_cat']:.5f}")

    p_ens, p_tuned = final_predict(train_df, test_df, cfg, result["best_iters_cat"])
    save_submission("final_submission_cat_ensemble.csv", test_df["FINANCEORDERID"].values, p_ens)
    save_submission("final_submission_cat_tuned.csv", test_df["FINANCEORDERID"].values, p_tuned)
    save_submission("final_submission_cat_final.csv", test_df["FINANCEORDERID"].values, p_ens)

    summary = {
        "config": cfg.__dict__,
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "fold_report": result["fold_report"],
        "oof_auc_cat": result["oof_auc_cat"],
        "best_iters_cat": result["best_iters_cat"],
        "final_choice": "cat_ensemble",
    }
    with open("cat_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log("Saved final_submission_cat_ensemble.csv")
    log("Saved final_submission_cat_tuned.csv")
    log("Saved final_submission_cat_final.csv")
    log("Saved cat_summary.json")


if __name__ == "__main__":
    main()
