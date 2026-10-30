/* ComplaintTriage - SAS modeling step
   Run in SAS OnDemand for Academics (free, real SAS).
   Input: complaints_features.csv (produced by python/clean_and_features.py)
   Target: monetary_relief (1 = complaint closed with monetary relief, 0 = not)

   Two models are fit and compared:
     1. PROC LOGISTIC  - structured features only (baseline)
     2. PROC LOGISTIC  - structured + text-derived features (full model)
   A PROC HPFOREST run is included as a nonlinear check against the same
   feature sets.

   Honesty note: whatever lift numbers this run produces go into the
   portfolio writeup as-is. No target performance number is assumed in
   advance.
*/

libname ctriage "~/complainttriage/data";

proc import datafile="~/complainttriage/data/complaints_features.csv"
    out=ctriage.complaints
    dbms=csv
    replace;
    getnames=yes;
run;

/* 80/20 train-test split */
proc surveyselect data=ctriage.complaints
    out=ctriage.split
    outall
    samprate=0.8
    seed=20260729;
run;

data ctriage.train ctriage.test;
    set ctriage.split;
    if selected = 1 then output ctriage.train;
    else output ctriage.test;
run;

/* ---------- Model 1: structured features only (baseline) ---------- */
proc logistic data=ctriage.train outmodel=ctriage.mod_structured;
    class product state timely_flag submitted_web has_tag / param=ref;
    model monetary_relief(event='1') = product state timely_flag
          submitted_web has_tag days_to_route;
    output out=ctriage.pred_structured p=pred_structured;
run;

proc logistic inmodel=ctriage.mod_structured;
    score data=ctriage.test out=ctriage.score_structured;
run;

/* ---------- Model 2: structured + text-derived features (full model) ---------- */
proc logistic data=ctriage.train outmodel=ctriage.mod_full;
    class product state timely_flag submitted_web has_tag has_narrative_flag / param=ref;
    model monetary_relief(event='1') = product state timely_flag
          submitted_web has_tag days_to_route
          narrative_length narrative_word_count redaction_count
          negative_word_count negative_word_rate has_narrative_flag;
    output out=ctriage.pred_full p=pred_full;
run;

proc logistic inmodel=ctriage.mod_full;
    score data=ctriage.test out=ctriage.score_full;
run;

/* ---------- Nonlinear check: HP Forest on the full feature set ---------- */
proc hpforest data=ctriage.train maxtrees=200 vars_to_try=5;
    target monetary_relief / level=binary;
    input product state timely_flag submitted_web has_tag has_narrative_flag
          / level=nominal;
    input days_to_route narrative_length narrative_word_count redaction_count
          negative_word_count negative_word_rate / level=interval;
    score out=ctriage.score_forest;
    save file="~/complainttriage/output/forest_model.bin";
run;

/* ---------- Compare AUC: structured-only baseline vs full model ---------- */
proc logistic data=ctriage.score_structured;
    model monetary_relief(event='1') = pred_structured;
    roc; roccontrast;
    ods output ROCAssociation=ctriage.auc_structured;
run;

proc logistic data=ctriage.score_full;
    model monetary_relief(event='1') = pred_full;
    roc; roccontrast;
    ods output ROCAssociation=ctriage.auc_full;
run;

proc print data=ctriage.auc_structured; title "Baseline (structured only) AUC"; run;
proc print data=ctriage.auc_full; title "Full model (structured + text) AUC"; run;
