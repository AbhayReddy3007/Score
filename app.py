# app.py
"""
Overall drug scorer (one total score per drug, max 20) using exact thresholds provided.

Run:
    pip install streamlit pandas
    streamlit run app.py
"""
import streamlit as st
import pandas as pd
import json
from collections import Counter

st.set_page_config(page_title="Drug Scorer (overall)", layout="wide")
st.title("Drug Scorer — overall score per drug (max 20)")

uploaded = st.file_uploader("Upload annotated CSV (prefer a 'drug' column).", type=["csv"])
if not uploaded:
    st.info("Upload a CSV (annotated by extractor or manually). Helpful column names (case-insensitive): "
            "drug, weight_loss_pct, a1c_reduction_pct, alt_reduction_pct, alt_reduction_flag, "
            "mash_resolution_with_no_worsening_pct, mash_resolution_pct, mash_fibrosis_worsening_pct, mash_resolution.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.write("Detected columns:", ", ".join(df.columns.tolist()))

# case-insensitive column finder
col_map = {c.lower().strip(): c for c in df.columns}
def find(*names):
    for n in names:
        k = n.lower().strip()
        if k in col_map:
            return col_map[k]
    return None

drug_col = find("drug")
wcol = find("weight_loss_pct", "weight loss pct", "weight_loss", "weight")
a1ccol = find("a1c_reduction_pct", "a1c reduction pct", "a1c_reduction", "a1c")
alt_num_col = find("alt_reduction_pct", "alt reduction pct", "alt_reduction", "alt")
alt_flag_col = find("alt_reduction_flag", "alt_flag", "alt reduction flag")
mash_no_worse_col = find("mash_resolution_with_no_worsening_pct", "mash_resolution_no_worse_pct", "mash_resolution_no_worsen_pct")
mash_res_pct_col = find("mash_resolution_pct", "mash resolution pct", "mash_resolution")
mash_fib_worse_col = find("mash_fibrosis_worsening_pct", "mash_fibrosis_worsen_pct", "mash_fibrosis_worsened_pct")
mash_cat_col = find("mash_resolution", "mash resolution", "mash")

st.markdown("### Preview (first 5 rows)")
st.dataframe(df.head(5))

st.markdown("---")
st.write("Scoring uses fixed thresholds (no UI adjustments).")

# Threshold functions (exact rules provided)
def score_weight_mean(v):
    if v is None:
        return 1
    try:
        v = float(v)
    except Exception:
        return 1
    if v >= 22.0:
        return 5
    if v >= 16.0:
        return 4
    if v >= 10.0:
        return 3
    if v >= 5.0:
        return 2
    return 1

def score_a1c_mean(v):
    if v is None:
        return 1
    try:
        v = float(v)
    except Exception:
        return 1
    if v >= 2.2:
        return 5
    if v >= 1.8:
        return 4
    if v >= 1.2:
        return 3
    if v >= 0.8:
        return 2
    return 1

def score_alt_mean_percent(v):
    # v = mean percent reduction from baseline
    if v is None:
        return 1
    try:
        v = float(v)
    except Exception:
        return 1
    # note: "5: >50%" so strictly >50
    if v > 50.0:
        return 5
    if 30.0 <= v <= 50.0:
        return 4
    if 15.0 <= v <= 29.0:
        return 3
    if 0.0 < v < 15.0:
        return 2
    # v <= 0 -> no reduction or increase
    return 1

# fallback mapping if only alt flag exists
def score_alt_flag(c):
    if not isinstance(c, str):
        c = str(c) if pd.notna(c) else ""
    c = c.strip().lower()
    if c in ("yes", "y"):
        # conservative mapping -> moderate signal
        return 3
    if c in ("no", "n"):
        return 1
    return 2  # unclear/mixed

# MASH scoring according to your rule
def mash_score_from_group(group):
    """
    Priority:
    1) If `mash_resolution_with_no_worsening_pct` available -> use it:
       >=50 ->5, >=30 ->4, >0 ->3, 0 ->1
    2) Else if mash_resolution_pct and mash_fibrosis_worsening_pct available:
       estimate no_worse = max(0, res_pct - fib_worsen_pct) and apply same thresholds
       if est_no_worse >0 but some fibrosis worsening exists -> give 3 per rule
    3) Else if mash_resolution_pct only -> use res_pct but fibrosis unknown:
       >=50 ->5, >=30 ->4, >0 ->3, 0 ->1
    4) Else categorical mode on mash_resolution: map mixed->2, yes->3 (fibrosis unknown), no->1, unclear->2
    """
    # helper to compute mean of a numeric column in group
    def mean_of(colname):
        if colname is None or colname not in group.columns:
            return None
        vals = pd.to_numeric(group[colname], errors="coerce").dropna()
        return float(vals.mean()) if len(vals)>0 else None

    no_worse = mean_of(mash_no_worse_col)
    res_pct = mean_of(mash_res_pct_col)
    fib_worse = mean_of(mash_fib_worse_col)

    # 1) explicit no-worse %
    if no_worse is not None:
        if no_worse >= 50.0:
            return 5, {"method":"no_worse_pct","value":no_worse}
        if no_worse >= 30.0:
            return 4, {"method":"no_worse_pct","value":no_worse}
        if no_worse > 0.0:
            return 3, {"method":"no_worse_pct","value":no_worse}
        return 1, {"method":"no_worse_pct","value":no_worse}

    # 2) estimate from res_pct and fibrosis worsening
    if res_pct is not None and fib_worse is not None:
        est_no_worse = max(0.0, res_pct - fib_worse)
        # if fibrosis worsening exists but resolution > 0 -> this falls into 3 if est_no_worse >0
        if est_no_worse >= 50.0:
            return 5, {"method":"estimated_no_worse","est":est_no_worse,"res_pct":res_pct,"fib_worse":fib_worse}
        if est_no_worse >= 30.0:
            return 4, {"method":"estimated_no_worse","est":est_no_worse,"res_pct":res_pct,"fib_worse":fib_worse}
        if est_no_worse > 0.0:
            # some resolution but fibrosis worsened for some -> 3
            return 3, {"method":"estimated_no_worse","est":est_no_worse,"res_pct":res_pct,"fib_worse":fib_worse}
        return 1, {"method":"estimated_no_worse","est":est_no_worse,"res_pct":res_pct,"fib_worse":fib_worse}

    # 3) res_pct only, fibrosis unknown
    if res_pct is not None:
        if res_pct >= 50.0:
            return 5, {"method":"res_pct_no_fib_info","value":res_pct}
        if res_pct >= 30.0:
            return 4, {"method":"res_pct_no_fib_info","value":res_pct}
        if res_pct > 0.0:
            return 3, {"method":"res_pct_no_fib_info","value":res_pct}
        return 1, {"method":"res_pct_no_fib_info","value":res_pct}

    # 4) categorical mode
    if mash_cat_col is None or mash_cat_col not in group.columns:
        return 2, {"method":"no_info","value":None}
    cats = group[mash_cat_col].astype(str).fillna("").str.strip().str.lower()
    cats = [c for c in cats if c != ""]
    if not cats:
        return 2, {"method":"no_info","value":None}
    ctr = Counter(cats)
    most = ctr.most_common()
    if len(most) > 1 and most[0][1] == most[1][1]:
        # tie => ambiguous
        return 2, {"method":"categorical_mode_tie","value":most}
    top = most[0][0]
    if top in ("mixed","mixed results","ambiguous","inconclusive"):
        return 2, {"method":"categorical_mode","value":top}
    if top in ("yes","y","resolved","resolution"):
        return 3, {"method":"categorical_mode","value":top}  # fibrosis unknown
    if top in ("no","n"):
        return 1, {"method":"categorical_mode","value":top}
    return 2, {"method":"categorical_mode","value":top}

# Group and compute
if st.button("Compute overall scores per drug (fixed thresholds)"):
    # group key
    if drug_col is None:
        df["_drug_group"] = "ALL_DATA"
        group_key = "_drug_group"
    else:
        group_key = drug_col

    rows = []
    for name, group in df.groupby(group_key):
        rec = {"drug": name}

        # mean helper
        def mean_or_none(colname):
            if colname is None or colname not in group.columns:
                return None
            vals = pd.to_numeric(group[colname], errors="coerce").dropna()
            return float(vals.mean()) if len(vals)>0 else None

        mean_w = mean_or_none(wcol)
        mean_a1c = mean_or_none(a1ccol)
        mean_alt = mean_or_none(alt_num_col)

        # compute endpoint scores
        w_score = score_weight_mean(mean_w)
        a1c_score = score_a1c_mean(mean_a1c)

        # ALT: numeric preferred, else flag mapping
        if mean_alt is not None:
            alt_score = score_alt_mean_percent(mean_alt)
        else:
            # fallback to categorical flag if present
            if alt_flag_col is not None and alt_flag_col in group.columns:
                # mode of flag
                flags = group[alt_flag_col].astype(str).fillna("").str.strip().str.lower()
                flags = [f for f in flags if f != ""]
                if flags:
                    ctr = Counter(flags)
                    # if tie -> unclear -> score 2
                    most = ctr.most_common()
                    if len(most) > 1 and most[0][1] == most[1][1]:
                        alt_score = 2
                    else:
                        topf = most[0][0]
                        alt_score = score_alt_flag(topf)
                else:
                    alt_score = 1
            else:
                alt_score = 1

        # MASH score using priority rules
        mash_score_val, mash_meta = mash_score_from_group(group)

        total = int(w_score + a1c_score + mash_score_val + alt_score)
        rec.update({
            "mean_weight_loss_pct": round(mean_w,3) if mean_w is not None else None,
            "mean_a1c_reduction_pct": round(mean_a1c,3) if mean_a1c is not None else None,
            "mean_alt_reduction_pct": round(mean_alt,3) if mean_alt is not None else None,
            "mean_mash_resolution_pct": round(mean_or_none(mash_res_pct_col),3) if mean_or_none(mash_res_pct_col) is not None else None,
            "mean_mash_fibrosis_worsening_pct": round(mean_or_none(mash_fib_worse_col),3) if mean_or_none(mash_fib_worse_col) is not None else None,
            "mean_mash_no_worse_pct": round(mean_or_none(mash_no_worse_col),3) if mean_or_none(mash_no_worse_col) is not None else None,
            "weight_score": int(w_score),
            "a1c_score": int(a1c_score),
            "mash_score": int(mash_score_val),
            "alt_score": int(alt_score),
            "total_score": total,
            "max_score": 20,
            "mash_score_meta": json.dumps(mash_meta)
        })
        rows.append(rec)

    outdf = pd.DataFrame(rows).sort_values("total_score", ascending=False).reset_index(drop=True)
    st.markdown("### Overall scores per drug")
    st.dataframe(outdf)

    st.markdown("### Top drug(s)")
    st.table(outdf.head(10)[["drug","total_score","weight_score","a1c_score","mash_score","alt_score"]])

    st.download_button("Download overall scores CSV", data=outdf.to_csv(index=False).encode("utf-8"), file_name="overall_scores_by_drug.csv", mime="text/csv")
    st.download_button("Download overall scores JSON", data=json.dumps(rows, indent=2).encode("utf-8"), file_name="overall_scores_by_drug.json", mime="application/json")

    st.success("Done — scoring applied with your exact thresholds.")
