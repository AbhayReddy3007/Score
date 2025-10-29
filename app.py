# app.py
import re
import json
from collections import Counter
from typing import Optional

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Single-Drug Overall Scorer", layout="wide")
st.title("Single-Drug Overall Scorer — values + scores (max 20)")

uploaded = st.file_uploader("Upload annotated CSV (all abstracts belong to one drug)", type=["csv"])
if not uploaded:
    st.info("Upload a CSV (annotated by extractor or manually). Helpful column names (case-insensitive): "
            "weight_loss_pct, a1c_reduction_pct, alt_reduction_pct, alt_reduction_flag, "
            "mash_resolution_with_no_worsening_pct, mash_resolution_pct, mash_fibrosis_worsening_pct, mash_resolution.")
    st.stop()

# utility: find column case-insensitively
def find_col(df, *names):
    cmap = {c.lower().strip(): c for c in df.columns}
    for n in names:
        k = n.lower().strip()
        if k in cmap:
            return cmap[k]
    return None

# utility: parse numeric percent-ish strings robustly
def parse_percentish(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "na", "none"):
        return None
    # find first float in the string (may include %)
    m = re.search(r"-?\d+\.?\d*", s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

# scoring functions (exact thresholds you gave)
def score_weight(v):
    if v is None:
        return 1
    if v >= 22.0:
        return 5
    if 16.0 <= v <= 21.9:
        return 4
    if 10.0 <= v <= 15.9:
        return 3
    if 5.0 <= v <= 9.9:
        return 2
    return 1

def score_a1c(v):
    if v is None:
        return 1
    if v >= 2.2:
        return 5
    if 1.8 <= v <= 2.1:
        return 4
    if 1.2 <= v <= 1.7:
        return 3
    if 0.8 <= v <= 1.1:
        return 2
    return 1

def score_alt_pct(v):
    # v is percent reduction from baseline
    if v is None:
        return 1
    # 5: >50%
    if v > 50.0:
        return 5
    if 30.0 <= v <= 50.0:
        return 4
    if 15.0 <= v <= 29.0:
        return 3
    if 0.0 < v < 15.0:
        return 2
    # v <= 0 -> no reduction / increase
    return 1

def score_alt_flag_val(cat):
    if cat is None:
        return 1
    c = str(cat).strip().lower()
    if c in ("yes","y"):
        # map yes -> moderate (3) by fallback convention
        return 3
    if c in ("no","n"):
        return 1
    return 2  # unclear/mixed

# MASH scoring as you specified (returns score, meta dict)
def mash_score_from_aggregates(no_worse_pct, res_pct, fib_worse_pct, cat_mode):
    # helper to coerce numeric
    def to_float(v):
        try:
            return None if v is None else float(v)
        except Exception:
            return None

    nw = to_float(no_worse_pct)
    rp = to_float(res_pct)
    fw = to_float(fib_worse_pct)

    # 1) explicit no-worse %
    if nw is not None:
        if nw >= 50.0:
            return 5, {"method":"no_worse_pct","value":nw}
        if nw >= 30.0:
            return 4, {"method":"no_worse_pct","value":nw}
        if nw > 0.0:
            return 3, {"method":"no_worse_pct","value":nw}
        return 1, {"method":"no_worse_pct","value":nw}

    # 2) estimate from res_pct and fibrosis worsening
    if rp is not None and fw is not None:
        est_no_worse = max(0.0, rp - fw)
        if est_no_worse >= 50.0:
            return 5, {"method":"estimated_no_worse","value":est_no_worse,"res_pct":rp,"fib_worsen_pct":fw}
        if est_no_worse >= 30.0:
            return 4, {"method":"estimated_no_worse","value":est_no_worse,"res_pct":rp,"fib_worsen_pct":fw}
        if est_no_worse > 0.0:
            # some resolution but fibrosis worsened for some -> 3
            return 3, {"method":"estimated_no_worse","value":est_no_worse,"res_pct":rp,"fib_worsen_pct":fw}
        return 1, {"method":"estimated_no_worse","value":est_no_worse,"res_pct":rp,"fib_worsen_pct":fw}

    # 3) res_pct only, fibrosis unknown
    if rp is not None:
        if rp >= 50.0:
            return 5, {"method":"res_pct_no_fib_info","value":rp}
        if rp >= 30.0:
            return 4, {"method":"res_pct_no_fib_info","value":rp}
        if rp > 0.0:
            return 3, {"method":"res_pct_no_fib_info","value":rp}
        return 1, {"method":"res_pct_no_fib_info","value":rp}

    # 4) categorical fallback
    if cat_mode is None:
        return 2, {"method":"no_info","value":None}
    c = str(cat_mode).strip().lower()
    if c in ("mixed","mixed results","ambiguous","inconclusive","unclear"):
        return 2, {"method":"categorical_mode","value":c}
    if c in ("yes","y","resolved","resolution"):
        return 3, {"method":"categorical_mode","value":c}  # fibrosis unknown -> 3
    if c in ("no","n"):
        return 1, {"method":"categorical_mode","value":c}
    return 2, {"method":"categorical_mode","value":c}

# Read uploaded CSV
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.markdown("### Data preview (first 6 rows)")
st.dataframe(df.head(6))

# find columns
wcol = find_col(df := df, *["weight_loss_pct","weight loss pct","weight_loss","weight"])
a1c_col = find_col(df, *["a1c_reduction_pct","a1c reduction pct","a1c_reduction","a1c"])
alt_num_col = find_col(df, *["alt_reduction_pct","alt reduction pct","alt_reduction","alt"])
alt_flag_col = find_col(df, *["alt_reduction_flag","alt_flag","alt reduction flag"])
mash_no_worse_col = find_col(df, *["mash_resolution_with_no_worsening_pct","mash_resolution_no_worse_pct","mash_resolution_no_worsen_pct"])
mash_res_pct_col = find_col(df, *["mash_resolution_pct","mash resolution pct","mash_resolution"])
mash_fib_worse_col = find_col(df, *["mash_fibrosis_worsening_pct","mash_fibrosis_worsen_pct","mash_fibrosis_worsened_pct"])
mash_cat_col = find_col(df, *["mash_resolution","mash resolution","mash"])

# helper to compute mean of parsed numeric column
def mean_parsed_percent(df, colname):
    if colname is None or colname not in df.columns:
        return None
    vals = df[colname].apply(parse_percentish).dropna()
    return float(vals.mean()) if len(vals)>0 else None

# compute aggregated values across all rows
mean_weight = mean_parsed_percent(df, wcol)
mean_a1c = mean_parsed_percent(df, a1c_col)
mean_alt = mean_parsed_percent(df, alt_num_col)

mean_mash_no_worse = mean_parsed_percent(df, mash_no_worse_col)
mean_mash_res_pct = mean_parsed_percent(df, mash_res_pct_col)
mean_mash_fib_worse = mean_parsed_percent(df, mash_fib_worse_col)

# categorical mode for mash and alt flag
def categorical_mode(df, colname):
    if colname is None or colname not in df.columns:
        return None, 0
    vals = df[colname].astype(str).fillna("").str.strip().str.lower()
    vals = [v for v in vals if v != ""]
    if not vals:
        return None, 0
    ctr = Counter(vals)
    most = ctr.most_common()
    if len(most) > 1 and most[0][1] == most[1][1]:
        # tie => ambiguous
        return "unclear", most[0][1]
    return most[0][0], most[0][1]

agg_mash_cat, mash_cat_count = categorical_mode(df, mash_cat_col)
agg_alt_flag, alt_flag_count = categorical_mode(df, alt_flag_col)

# compute MASH score
mash_score_val, mash_meta = mash_score_from_aggregates(mean_mash_no_worse, mean_mash_res_pct, mean_mash_fib_worse, agg_mash_cat)

# compute other endpoint scores
weight_score = score_weight(mean_weight)
a1c_score = score_a1c(mean_a1c)

if mean_alt is not None:
    alt_score = score_alt_pct(mean_alt)
else:
    # fallback to alt flag mapping if available
    if agg_alt_flag is not None:
        alt_score = score_alt_flag_val(agg_alt_flag)
    else:
        alt_score = 1

total_score = int(weight_score + a1c_score + mash_score_val + alt_score)

# Display aggregated numeric values and scores
st.markdown("## Aggregated values (across all abstracts)")
vals = {
    "mean_weight_loss_pct": mean_weight,
    "mean_a1c_reduction_pct": mean_a1c,
    "mean_alt_reduction_pct": mean_alt,
    "mean_mash_resolution_pct": mean_mash_res_pct,
    "mean_mash_fibrosis_worsening_pct": mean_mash_fib_worse,
    "mean_mash_resolution_no_worse_pct": mean_mash_no_worse,
    "agg_mash_categorical_mode": agg_mash_cat,
    "agg_alt_flag_mode": agg_alt_flag
}
st.json({k:(round(v,3) if isinstance(v,(int,float)) else v) for k,v in vals.items()})

st.markdown("## Scores")
scores = {
    "weight_score": weight_score,
    "a1c_score": a1c_score,
    "mash_score": int(mash_score_val),
    "alt_score": int(alt_score),
    "total_score": total_score,
    "max_score": 20
}
st.json(scores)

st.markdown("### Human-friendly summary")
summary_lines = [
    f"Mean weight loss: {round(mean_weight,2) if mean_weight is not None else 'N/A'}%  -> weight score = {weight_score}/5",
    f"Mean A1c reduction: {round(mean_a1c,2) if mean_a1c is not None else 'N/A'}%  -> A1c score = {a1c_score}/5",
    f"Mean ALT reduction: {round(mean_alt,2) if mean_alt is not None else 'N/A'}%  -> ALT score = {alt_score}/5",
    f"MASH: score = {mash_score_val}/5  (meta = {mash_meta})"
]
for ln in summary_lines:
    st.write(ln)

# Offer downloads
out_obj = {
    "aggregated_values": vals,
    "scores": scores,
    "mash_meta": mash_meta
}
st.download_button("Download JSON result", data=json.dumps(out_obj, indent=2).encode("utf-8"), file_name="single_drug_score.json", mime="application/json")
st.download_button("Download CSV summary", data=pd.DataFrame([ {**vals, **scores} ]).to_csv(index=False).encode("utf-8"), file_name="single_drug_score.csv", mime="text/csv")

st.success("Done — aggregated values and scores computed for the single drug.")
