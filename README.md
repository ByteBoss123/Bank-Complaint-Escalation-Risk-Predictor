# ComplaintTriage

Predicts whether a CFPB consumer complaint against a bank ends up "Closed with
monetary relief" — a proxy for which incoming complaints deserve priority
review before they become regulatory or reputational risk.

Built as a portfolio project targeting the Truist Financial Data Scientist I/II
posting (structured + unstructured data, Python + SAS, banking domain).

## Why this target, not "disputed"

The CFPB Consumer Complaint Database used to carry a `consumer_disputed` flag.
It was discontinued after a 2017 rule change and does not exist in current API
pulls (confirmed directly against a live query on 2026-07-29). The target here
is `monetary_relief` (`company_response == "Closed with monetary relief"`),
which is a real, current column — not a stand-in dressed up to look like the
old one.

## Data

Source: [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/),
public, updated daily, no API key required.

Pull your own filtered CSV:

```bash
curl -G "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/" \
  --data-urlencode "product=Checking or savings account" \
  --data-urlencode "has_narrative=true" \
  --data-urlencode "size=5000" \
  --data-urlencode "format=csv" \
  -o data/complaints_raw.csv
```

Swap `product` for other bank-relevant categories as needed
(`Credit card`, `Bank account or service`, `Money transfer, virtual currency, or money service`).
Note: the API's `frm` offset parameter is currently broken upstream (confirmed
via a CFPB GitHub issue) — for a larger sample, pull multiple date-windowed
CSVs (`date_received_min` / `date_received_max`) instead of relying on
pagination.

## Pipeline

1. `python/clean_and_features.py` — cleans the raw export and builds:
   - **Structured features**: product, state, timely-response flag, submission
     channel, tag presence (e.g. Servicemember), days from receipt to routing
   - **Text-derived features**: narrative length, word count, PII-redaction
     density (CFPB masks personal info as `XXXX`), a lightweight negative-word
     rate
2. `sas/model_complaint_triage.sas` — fits two models in SAS (SAS OnDemand for
   Academics, free):
   - Baseline: structured features only
   - Full: structured + text-derived features
   - Plus a `PROC HPFOREST` nonlinear check
   - Compares AUC between baseline and full model

## Honesty note

Whatever AUC lift the SAS run produces from adding text features is the
number that goes in the writeup — no target lift is assumed before running
it. If the text features don't help, that's a reportable finding too.

## Status

- [x] Schema confirmed against live CFPB API
- [x] Cleaning + feature engineering script
- [x] SAS modeling script
- [ ] Real data pull (blocked in this sandbox — no network path to
      files.consumerfinance.gov; run the curl command above locally)
- [ ] Model results + stakeholder one-pager
