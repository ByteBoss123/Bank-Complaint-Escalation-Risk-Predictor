"""Feature checks on a few hand-written rows (shape and leakage checks only, no results claimed)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from complainttriage.features import CATEGORICAL, NUMERIC, build, load


def _raw(tmp_path):
    rows = [
        {"Date received": "2024-01-02T10:00:00.000Z", "Product": "Checking or savings account", "Sub-product": "Checking account",
         "Issue": "Managing an account", "Sub-issue": "Deposits and withdrawals",
         "Consumer complaint narrative": "They took {$1,250.00} and {$40.00} from XXXX account", "Company public response": "x",
         "Company": "BANK A", "State": "VA", "ZIP code": "22030", "Tags": None, "Submitted via": "Web",
         "Date sent to company": "2024-01-02", "Company response to consumer": "Closed with monetary relief",
         "Timely response?": "Yes", "Complaint ID": 1},
        {"Date received": "2024-01-03T10:00:00.000Z", "Product": "Checking or savings account", "Sub-product": None,
         "Issue": "Opening an account", "Sub-issue": None, "Consumer complaint narrative": "slow service",
         "Company public response": None, "Company": "BANK B", "State": None, "ZIP code": None, "Tags": "Servicemember",
         "Submitted via": "Web", "Date sent to company": "2024-01-03", "Company response to consumer": "Closed with explanation",
         "Timely response?": "Yes", "Complaint ID": 2},
        {"Date received": "2024-01-04T10:00:00.000Z", "Product": "Checking or savings account", "Sub-product": None,
         "Issue": "x", "Sub-issue": None, "Consumer complaint narrative": "pending", "Company public response": None,
         "Company": "BANK B", "State": "VA", "ZIP code": None, "Tags": None, "Submitted via": "Web",
         "Date sent to company": "2024-01-04", "Company response to consumer": "Untimely response",
         "Timely response?": "No", "Complaint ID": 3},
    ]
    p = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_load_drops_rows_without_an_outcome_and_sets_target(tmp_path):
    df = load(_raw(tmp_path))
    assert df["complaint_id"].tolist() == [1, 2]
    assert df["y"].tolist() == [1, 0]


def test_features_use_only_intake_fields(tmp_path):
    df = load(_raw(tmp_path))
    X, top = build(df)
    assert list(X.columns) == CATEGORICAL + NUMERIC + ["narrative"]
    leaky = {"company_public_response", "timely_response", "company_response_to_consumer", "y"}
    assert not leaky & set(X.columns)
    assert X.loc[0, "max_dollar_log"] == np.log1p(1250.0)
    assert X.loc[0, "dollar_mentions"] == 2
    assert not X[CATEGORICAL + NUMERIC].isna().any().any()
    X2, _ = build(df.assign(company="NEW BANK"), top)
    assert (X2["company_top"] == "Other").all()


RESULTS = Path(__file__).parents[1] / "results" / "results.json"


def test_reported_split_is_by_time():
    if not RESULTS.exists():
        return
    r = json.loads(RESULTS.read_text())
    s = r["data"]["split"]
    assert s["train_2017_2023"] + s["validation_2024"] + s["test_2025_2026"] == r["data"]["complaints"]
