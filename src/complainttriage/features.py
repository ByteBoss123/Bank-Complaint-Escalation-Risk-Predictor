"""Intake-time features from the CFPB export.

Only fields known when a complaint arrives are used. Excluded on purpose because they are set
after the company responds: "Company public response", "Timely response?", and the outcome itself.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

TARGET_VALUE = "Closed with monetary relief"
CATEGORICAL = ["sub_product", "issue", "sub_issue", "company_top", "state", "submitted_via", "tags"]
NUMERIC = ["narrative_words_log", "redaction_share", "max_dollar_log", "dollar_mentions"]
DOLLAR = re.compile(r"\{\$([\d,]+(?:\.\d+)?)\}")      # CFPB formats amounts as {$1,234.00}


def load(path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_").replace("?", "") for c in df.columns]
    df = df[df["company_response_to_consumer"] != "Untimely response"].copy()   # no outcome yet
    df["y"] = (df["company_response_to_consumer"] == TARGET_VALUE).astype(int)
    df["date_received"] = pd.to_datetime(df["date_received"], utc=True)
    df["narrative"] = df["consumer_complaint_narrative"].fillna("")
    return df.reset_index(drop=True)


def _max_dollar(text: str) -> float:
    vals = [float(v.replace(",", "")) for v in DOLLAR.findall(text)]
    return max(vals) if vals else 0.0


def build(df: pd.DataFrame, top_companies: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    if top_companies is None:
        top_companies = df["company"].value_counts().head(30).index.tolist()
    X = pd.DataFrame(index=df.index)
    X["sub_product"] = df["sub_product"].fillna("Missing")
    X["issue"] = df["issue"].fillna("Missing")
    X["sub_issue"] = df["sub_issue"].fillna("Missing")
    X["company_top"] = df["company"].where(df["company"].isin(top_companies), "Other")
    X["state"] = df["state"].fillna("Missing")
    X["submitted_via"] = df["submitted_via"].fillna("Missing")
    X["tags"] = df["tags"].fillna("None")
    words = df["narrative"].str.split().str.len().fillna(0)
    X["narrative_words_log"] = np.log1p(words)
    X["redaction_share"] = df["narrative"].str.count("XX") / words.clip(lower=1)
    amounts = df["narrative"].map(_max_dollar)
    X["max_dollar_log"] = np.log1p(amounts)
    X["dollar_mentions"] = df["narrative"].str.count(r"\{\$").clip(upper=20)
    X["narrative"] = df["narrative"]
    return X, top_companies
