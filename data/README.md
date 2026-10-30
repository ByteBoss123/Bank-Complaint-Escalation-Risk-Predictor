# Data

`complaints_raw.csv` is an export from the public [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/):
product **Checking or savings account**, complaints **with a consumer narrative**, pulled 2026-07-29.

| | |
|---|---|
| Rows | 185,043 (184,902 after dropping 141 "Untimely response" rows that have no outcome) |
| Date received | 2017-04-24 to 2026-07-09 |
| SHA-256 | `578cd403e13b574752a2d0b26f293093148791ef1a3b624656f6d03ca332a9af` |

The file (282 MB) is not committed. To rebuild it, export the same filter from the CFPB search page as CSV,
or pull date-windowed CSVs from the public API (its `frm` offset parameter is unreliable, so window by
`date_received_min` / `date_received_max` instead of paging). A fresh pull will include newer complaints and
won't match the checksum exactly.
