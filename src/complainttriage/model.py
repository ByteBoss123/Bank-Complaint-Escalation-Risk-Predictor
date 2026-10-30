"""Train on older complaints, choose on 2024, test on 2025-2026 (the most recent, scored once).

Candidates (logistic regression, so every driver can be read off the coefficients):
  structured only  — product details, issue, company, state, channel, tags, narrative length/amounts
  text only        — TF-IDF of the consumer's narrative (words and word pairs)
  structured + text
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL, NUMERIC, build, load

ROOT = Path(__file__).resolve().parents[2]


def make(kind: str, C: float = 1.0) -> Pipeline:
    parts = []
    if kind in ("structured", "both"):
        parts += [("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
                  ("num", StandardScaler(), NUMERIC)]
    if kind in ("text", "both"):
        parts += [("txt", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200000,
                                          sublinear_tf=True, stop_words="english"), "narrative")]
    return Pipeline([("prep", ColumnTransformer(parts)),
                     ("clf", LogisticRegression(C=C, max_iter=3000, solver="liblinear"))])


def capacity(y, p, frac):
    k = int(round(len(y) * frac))
    top = np.argsort(-p, kind="stable")[:k]
    hits = int(y[top].sum())
    return {"review_share": frac, "reviewed": k, "relief_cases_found": hits, "precision": round(hits / k, 4),
            "recall": round(hits / int(y.sum()), 4), "lift_vs_random": round(hits / k / y.mean(), 2)}


CANDIDATES = [("structured only", "structured", 1.0), ("text only", "text", 4.0),
              ("structured + text", "both", 1.0)]


def prepare(raw_path) -> dict:
    df = load(raw_path)
    yr = df["date_received"].dt.year
    tr, va, te = df[yr <= 2023], df[yr == 2024], df[yr >= 2025]
    Xtr, top = build(tr)
    Xva, _ = build(va, top)
    Xte, _ = build(te, top)
    meta = {"n": int(len(df)), "date_min": str(df.date_received.min().date()), "date_max": str(df.date_received.max().date())}
    return {"tr": tr, "va": va, "te": te, "Xtr": Xtr, "Xva": Xva, "Xte": Xte, "meta": meta}


def fit_candidate(prep: dict, kind: str, C: float) -> dict:
    m = make(kind, C).fit(prep["Xtr"], prep["tr"]["y"])
    p_va = m.predict_proba(prep["Xva"])[:, 1]
    p_te = m.predict_proba(prep["Xte"])[:, 1]
    names = m.named_steps["prep"].get_feature_names_out()
    return {"p_va": p_va, "p_te": p_te, "coef": pd.Series(m.named_steps["clf"].coef_[0], index=names),
            "val": {"auroc": round(roc_auc_score(prep["va"]["y"], p_va), 4),
                    "pr_auc": round(average_precision_score(prep["va"]["y"], p_va), 4)}}


def report(prep: dict, fits: dict) -> dict:
    tr, va, te = prep["tr"], prep["va"], prep["te"]
    val = {n: f["val"] for n, f in fits.items()}
    best = max(val, key=lambda k: val[k]["auroc"])
    y = te["y"].to_numpy()
    p = fits[best]["p_te"]
    test_all = {n: round(roc_auc_score(y, f["p_te"]), 4) for n, f in fits.items()}

    comp = te["company"].to_numpy()
    by_company = {}
    for c in te["company"].value_counts().head(6).index:
        mk = comp == c
        if y[mk].sum() >= 30 and (1 - y[mk]).sum() >= 30:
            by_company[c] = {"complaints": int(mk.sum()), "relief_rate": round(float(y[mk].mean()), 4),
                             "auroc": round(roc_auc_score(y[mk], p[mk]), 4)}
    years = te["date_received"].dt.year.to_numpy()
    by_year = {}
    for v in sorted(set(years)):
        mk = years == v
        by_year[int(v)] = {"complaints": int(mk.sum()), "relief_rate": round(float(y[mk].mean()), 4),
                           "auroc": round(roc_auc_score(y[mk], p[mk]), 4)}

    coef = fits[best]["coef"]
    is_txt = np.array([n.startswith("txt__") for n in coef.index])
    drivers = {}
    if is_txt.any():
        txt = coef[is_txt]
        drivers["text_raises_relief_odds"] = [n[5:] for n in txt.sort_values(ascending=False).head(15).index]
        drivers["text_lowers_relief_odds"] = [n[5:] for n in txt.sort_values().head(15).index]
    if (~is_txt).any():
        st = coef[~is_txt]
        drivers["structured_raises"] = [n.split("__", 1)[1] for n in st.sort_values(ascending=False).head(8).index]
        drivers["structured_lowers"] = [n.split("__", 1)[1] for n in st.sort_values().head(8).index]

    q = np.quantile(p, np.linspace(0, 1, 11))
    deciles = []
    for lo, hi in zip(q[:-1], q[1:]):
        mk = (p >= lo) & (p <= hi)
        deciles.append({"mean_predicted": round(float(p[mk].mean()), 4), "observed": round(float(y[mk].mean()), 4)})

    meta = prep["meta"]
    return {
        "data": {"source": "CFPB Consumer Complaint Database, product = Checking or savings account",
                 "complaints": meta["n"], "date_range": [meta["date_min"], meta["date_max"]],
                 "split": {"train_2017_2023": int(len(tr)), "validation_2024": int(len(va)), "test_2025_2026": int(len(te))},
                 "relief_rate": {"train": round(float(tr.y.mean()), 4), "validation": round(float(va.y.mean()), 4),
                                 "test": round(float(te.y.mean()), 4)}},
        "validation": val, "chosen_model": best,
        "test": {"auroc": round(roc_auc_score(y, p), 4), "pr_auc": round(average_precision_score(y, p), 4),
                 "pr_auc_random": round(float(y.mean()), 4), "brier": round(brier_score_loss(y, p), 4),
                 "auroc_all_candidates": test_all, "calibration_deciles": deciles},
        "review_capacity": [capacity(y, p, f) for f in (0.05, 0.10, 0.20)],
        "test_auroc_by_company": by_company, "test_by_year": by_year, "drivers": drivers,
    }


def run(raw_path, out: Path = ROOT / "results") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    prep = prepare(raw_path)
    fits = {name: fit_candidate(prep, kind, C) for name, kind, C in CANDIDATES}
    r = report(prep, fits)
    (out / "results.json").write_text(json.dumps(r, indent=2))
    return r


if __name__ == "__main__":
    r = run(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "complaints_raw.csv")
    print(json.dumps(r, indent=1)[:6000])
