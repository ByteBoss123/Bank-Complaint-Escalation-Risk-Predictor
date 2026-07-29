"""
ComplaintTriage - data cleaning + feature engineering
Input:  data/complaints_raw.csv  (pulled from the CFPB Consumer Complaint Database API,
        see README.md for the exact curl command)
Output: data/complaints_features.csv  (ready for the SAS modeling step)

Real CFPB schema fields used (confirmed against a live API pull on 2026-07-29):
    product, sub_product, issue, sub_issue, company, company_response,
    complaint_what_happened, has_narrative, timely, submitted_via, state,
    zip_code, tags, date_received, date_sent_to_company, complaint_id

NOTE: the CFPB discontinued the `consumer_disputed` field in 2017 following a
rule change. There is no dispute flag in current data. Target variable here is
`monetary_relief` = 1 if company_response == "Closed with monetary relief", else 0.
"""

import pandas as pd
import numpy as np
import re

RAW_PATH = "../data/complaints_raw.csv"
OUT_PATH = "../data/complaints_features.csv"

NEGATIVE_WORDS = {
    "unacceptable", "fraud", "fraudulent", "scam", "stolen", "theft", "lied",
    "lying", "illegal", "threatened", "threat", "harassment", "harass",
    "never", "refuse", "refused", "denied", "unauthorized", "disgusting",
    "terrible", "horrible", "worst", "angry", "furious", "victim",
}

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    if "company_response" not in df.columns:
        raise KeyError("Expected 'company_response' column not found in raw export.")
    df["monetary_relief"] = (
        df["company_response"].astype(str).str.strip() == "Closed with monetary relief"
    ).astype(int)
    return df


def structured_features(df: pd.DataFrame) -> pd.DataFrame:
    df["timely_flag"] = (df["timely"].astype(str).str.strip() == "Yes").astype(int)
    df["submitted_web"] = (df["submitted_via"].astype(str).str.strip() == "Web").astype(int)
    df["has_tag"] = df["tags"].notna().astype(int)  # e.g. Servicemember, Older American

    # complaint duration: date_sent_to_company - date_received, in days
    for col in ("date_received", "date_sent_to_company"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    if {"date_received", "date_sent_to_company"}.issubset(df.columns):
        df["days_to_route"] = (
            (df["date_sent_to_company"] - df["date_received"]).dt.total_seconds() / 86400
        ).clip(lower=0)
    else:
        df["days_to_route"] = np.nan

    df["state"] = df.get("state", pd.Series(dtype=str)).astype(str).str.upper().str.strip()
    return df


def text_features(df: pd.DataFrame) -> pd.DataFrame:
    narrative = df.get("complaint_what_happened", pd.Series([""] * len(df))).fillna("").astype(str)

    df["narrative_length"] = narrative.str.len()
    df["narrative_word_count"] = narrative.str.split().apply(len)

    # crude redaction-density check: CFPB masks PII as repeated "XXXX" tokens
    df["redaction_count"] = narrative.str.count(r"\bXXXX\b")

    # lightweight negative-sentiment proxy (no external NLP lib dependency here;
    # swap in VADER/TextBlob if available in the environment)
    lower_narrative = narrative.str.lower()
    df["negative_word_count"] = lower_narrative.apply(
        lambda t: sum(t.count(w) for w in NEGATIVE_WORDS)
    )
    df["negative_word_rate"] = df["negative_word_count"] / df["narrative_word_count"].replace(0, np.nan)
    df["negative_word_rate"] = df["negative_word_rate"].fillna(0)

    df["has_narrative_flag"] = (narrative.str.len() > 0).astype(int)
    return df


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "complaint_id", "product", "sub_product", "issue", "sub_issue",
        "state", "company", "timely_flag", "submitted_web", "has_tag",
        "days_to_route", "narrative_length", "narrative_word_count",
        "redaction_count", "negative_word_count", "negative_word_rate",
        "has_narrative_flag", "monetary_relief",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()
    out = out.dropna(subset=["monetary_relief"])
    return out


def main():
    df = load_raw(RAW_PATH)
    df = build_target(df)
    df = structured_features(df)
    df = text_features(df)
    out = finalize(df)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows, {out.shape[1]} columns to {OUT_PATH}")
    print("Target balance:")
    print(out["monetary_relief"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
