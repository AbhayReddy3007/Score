# app.py
"""
Streamlit app — batch dataset-level extractor per drug using Gemini 2.5-flash.

Behavior:
- Upload CSV or Excel with columns:
    * a drug column (case-insensitive: 'drug', 'drug_name', or 'treatment')
    * an 'abstract' column (case-insensitive)
- The app groups rows by drug, concatenates abstracts for each drug, sends one prompt
  per drug to Gemini to extract four dataset-level numbers:
    - average_weight_loss_pct
    - average_a1c_reduction_pct
    - mash_highest_resolution_pct
    - alt_highest_reduction_pct
  then computes scores for each and returns a table + downloadable CSV/JSON.

Requirements:
    pip install streamlit pandas openpyxl google-genai
Run:
    streamlit run app.py
"""
import os
import json
import time
from typing import Optional, Any, Dict, List

import streamlit as st
import pandas as pd

# lazy import genai to give helpful error if missing
try:
    from google import genai
except Exception:
    genai = None

# CONFIG
MODEL_NAME = "gemini-2.5-flash"
OUTPUT_JSON = "batch_outcomes.json"
OUTPUT_CSV = "batch_outcomes.csv"
DEFAULT_SLEEP = 0.25  # seconds between model calls (adjust if you hit rate limits)

# Prompt template per drug: returns a single JSON object
DATASET_PROMPT_TEMPLATE = (
    "You are an extractor for clinical research abstracts. I will provide a numbered list of "
    "abstracts from multiple studies for a single drug.\n\n"
    "Read all abstracts and return exactly one JSON object (single-line) with the following keys:\n\n"
    "1) average_weight_loss_pct -> numeric mean of reported weight-loss percentages across abstracts that report it (null if none).\n\n"
    "2) average_a1c_reduction_pct -> numeric mean of reported A1c absolute reductions (percentage points) across abstracts that report it (null if none).\n\n"
    "3) mash_highest_resolution_pct -> the HIGHEST percentage of patients reported in any study to have MASH resolution (percent from baseline). Return null if none reported.\n\n"
    "4) alt_highest_reduction_pct -> the HIGHEST percentage reduction in ALT from baseline reported in any abstract (percent). Return null if none reported.\n\n"
    "IMPORTANT: Return ONLY the single JSON object and NOTHING else. Do not include commentary.\n\n"
    "Now analyze these abstracts:\n\n"
    "-----\n"
    "{all_abstracts}\n"
    "-----\n"
)

# ----------------------------
# Helpers (parsing + scoring)
# ----------------------------
def extract_json(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    if s.startswith("```") and s.endswith("```"):
        parts = s.split("\n")
        if len(parts) > 2:
            inner = "\n".join(parts[1:-1]).strip()
            if inner:
                s = inner
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        try:
            return json.loads(s)
        except Exception:
            return None
    candidate = s[start:end+1]
    try:
        return json.loads(candidate)
    except Exception:
        try:
            cleaned = candidate.replace(",}", "}")
            return json.loads(cleaned)
        except Exception:
            return None

def safe_to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        try:
            return float(val)
        except Exception:
            return None
    if isinstance(val, str):
        s = val.strip()
        if s == "" or s.lower() in {"null","none","n/a","na"}:
            return None
        s = s.replace("%","").replace(",","")
        try:
            return float(s)
        except Exception:
            return None
    return None

def normalize_yes_no_unclear(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, bool):
        return "yes" if val else "no"
    s = str(val).strip().lower()
    if s in {"yes","y","true","t"}:
        return "yes"
    if s in {"no","n","false","f"}:
        return "no"
    if s in {"unclear","unknown","maybe","ambiguous","unsure","conflicting","mixed"}:
        return "unclear"
    if "worsen" in s or "progress" in s or "fibrosis increased" in s or "fibrosis progression" in s:
        return "yes"
    if "no worsen" in s or "no progression" in s or "no fibrosis worsening" in s:
        return "no"
    return "unclear"

# scoring functions (identical to your rules)
def score_weight_loss(pct: Optional[float]) -> int:
    if pct is None:
        return 0
    if pct >= 22:
        return 5
    if 16 <= pct <= 21.9:
        return 4
    if 10 <= pct <= 15.9:
        return 3
    if 5 <= pct <= 9.9:
        return 2
    return 1

def score_a1c_reduction(pct: Optional[float]) -> int:
    if pct is None:
        return 0
    if pct >= 2.2:
        return 5
    if 1.8 <= pct <= 2.1:
        return 4
    if 1.2 <= pct <= 1.7:
        return 3
    if 0.8 <= pct <= 1.1:
        return 2
    return 1

def score_mash(highest_pct: Optional[float], fibrosis_status: Optional[str]) -> int:
    pct = safe_to_float(highest_pct)
    fib = normalize_yes_no_unclear(fibrosis_status) if fibrosis_status is not None else None
    if pct is None or (isinstance(pct,float) and pct == 0.0):
        return 1
    if fib is None or fib == "unclear":
        return 2
    if fib == "no":
        if pct >= 50:
            return 5
        if pct >= 30:
            return 4
        return 2
    if fib == "yes":
        return 3
    return 2

def score_alt(highest_pct: Optional[float]) -> int:
    pct = safe_to_float(highest_pct)
    if pct is None or (isinstance(pct,float) and pct <= 0.0):
        return 1
    if pct > 50:
        return 5
    if 30 <= pct <= 50:
        return 4
    if 15 <= pct <= 29:
        return 3
    if 0 < pct < 15:
        return 2
    return 1

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Batch per-drug extractor", layout="wide")
st.title("Batch per-drug dataset-level extractor — Gemini 2.5-flash")
st.markdown(
    "- Upload a CSV or Excel with one abstract per row and a drug identifier column.\n"
    "- The app will group by drug and compute the four numbers + their scores for each drug."
)

uploaded_file = st.file_uploader("Upload CSV or Excel (xlsx)", type=["csv","xlsx","xls"])
if not uploaded_file:
    st.info("Upload a CSV or Excel file to begin. Required columns: a drug column and 'abstract'.")
    st.stop()

# read file into DataFrame
try:
    if str(uploaded_file.name).lower().endswith((".xls",".xlsx")):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read uploaded file: {e}")
    st.stop()

# find drug column
drug_col = None
for c in df.columns:
    if c.strip().lower() in {"drug","drug_name","treatment","drugname"}:
        drug_col = c
        break
if drug_col is None:
    st.error("Could not find a drug column. Please include a column named 'drug' or 'drug_name' (case-insensitive).")
    st.stop()

# find abstract column
abstract_col = None
for c in df.columns:
    if c.strip().lower() == "abstract":
        abstract_col = c
        break
if abstract_col is None:
    st.error("CSV/Excel must contain an 'abstract' column (case-insensitive).")
    st.stop()

st.success(f"Found drug column '{drug_col}' and abstract column '{abstract_col}' — {len(df)} rows.")

# parameters
per_call_sleep = st.number_input("Seconds to sleep between model calls (per drug)", value=DEFAULT_SLEEP, min_value=0.0, step=0.05)
max_drugs = st.number_input("Max number of unique drugs to process (0 = all)", value=0, min_value=0)

# API key and dependency
api_key = None
try:
    api_key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
except Exception:
    api_key = None
if not api_key:
    api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("No Gemini API key found. Set GEMINI_API_KEY in Streamlit secrets or environment variables.")
    st.stop()
if genai is None:
    st.error("Missing dependency: google-genai. Install with `pip install google-genai`.")
    st.stop()

# start processing
if st.button("Run batch extraction per drug"):
    client = genai.Client(api_key=api_key)

    # group by drug → list of abstracts per drug
    grouped = df.groupby(df[drug_col].astype(str))[abstract_col].apply(list).to_dict()
    unique_drugs = list(grouped.keys())
    if max_drugs and max_drugs > 0:
        unique_drugs = unique_drugs[:int(max_drugs)]

    n = len(unique_drugs)
    st.write(f"Processing {n} drugs...")
    progress = st.progress(0)

    results: List[Dict[str, Any]] = []
    for i, drug in enumerate(unique_drugs, start=1):
        abstracts = grouped[drug]
        # build combined numbered abstracts
        combined = "\n\n".join([f"{idx+1}. {' '.join(str(a).split())}" for idx,a in enumerate(abstracts)])
        prompt = DATASET_PROMPT_TEMPLATE.replace("{all_abstracts}", combined)

        # call model
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        except Exception as e:
            st.warning(f"API error for drug '{drug}': {e}")
            # append row with NaNs and continue
            results.append({
                "drug": drug,
                "average_weight_loss_pct_normalized": None,
                "weight_score_points": None,
                "average_a1c_reduction_pct_normalized": None,
                "a1c_score_points": None,
                "mash_highest_resolution_pct_normalized": None,
                "mash_score_points": None,
                "alt_highest_reduction_pct_normalized": None,
                "alt_score_points": None,
                "raw_model_output": None
            })
            progress.progress(int(i/n*100))
            time.sleep(per_call_sleep)
            continue

        # robustly extract text from response
        resp_text = ""
        if hasattr(response, "text") and response.text:
            resp_text = response.text
        else:
            try:
                if getattr(response, "candidates", None):
                    cand = response.candidates[0]
                    parts = getattr(cand, "content", None)
                    if parts and getattr(parts, "parts", None):
                        resp_text = "".join([p.text for p in parts.parts if getattr(p, "text", None)])
                    else:
                        resp_text = str(response)
                else:
                    resp_text = str(response)
            except Exception:
                resp_text = str(response)

        parsed = extract_json(resp_text)
        if parsed is None:
            st.warning(f"Could not parse JSON for drug '{drug}'. Saving raw output for inspection.")
            results.append({
                "drug": drug,
                "average_weight_loss_pct_normalized": None,
                "weight_score_points": None,
                "average_a1c_reduction_pct_normalized": None,
                "a1c_score_points": None,
                "mash_highest_resolution_pct_normalized": None,
                "mash_score_points": None,
                "alt_highest_reduction_pct_normalized": None,
                "alt_score_points": None,
                "raw_model_output": resp_text
            })
            progress.progress(int(i/n*100))
            time.sleep(per_call_sleep)
            continue

        # extract numeric fields (with fallback key detection)
        def get_first_key_like(d: dict, substrings: List[str]):
            for k in d.keys():
                kl = k.lower()
                if all(sub in kl for sub in substrings):
                    return d.get(k)
            return None

        avg_wt = safe_to_float(parsed.get("average_weight_loss_pct") or get_first_key_like(parsed, ["average","weight","loss","pct"]) )
        avg_a1c = safe_to_float(parsed.get("average_a1c_reduction_pct") or get_first_key_like(parsed, ["average","a1c","reduction","pct"]) )
        mash_highest = parsed.get("mash_highest_resolution_pct") or get_first_key_like(parsed, ["mash","highest","resolution","pct"])
        alt_highest = parsed.get("alt_highest_reduction_pct") or get_first_key_like(parsed, ["alt","highest","reduction","pct"])

        # optional fibrosis field (to compute MASH score correctly)
        mash_fibrosis_raw = parsed.get("mash_worsening_of_fibrosis") or get_first_key_like(parsed, ["mash","worsen","fibrosis"])

        mash_fibrosis_norm = normalize_yes_no_unclear(mash_fibrosis_raw) if mash_fibrosis_raw is not None else None

        # compute scores
        weight_points = score_weight_loss(avg_wt)
        a1c_points = score_a1c_reduction(avg_a1c)
        mash_points = score_mash(mash_highest, mash_fibrosis_norm)
        alt_points = score_alt(alt_highest)

        results.append({
            "drug": drug,
            "average_weight_loss_pct_normalized": avg_wt,
            "weight_score_points": weight_points,
            "average_a1c_reduction_pct_normalized": avg_a1c,
            "a1c_score_points": a1c_points,
            "mash_highest_resolution_pct_normalized": safe_to_float(mash_highest),
            "mash_score_points": mash_points,
            "alt_highest_reduction_pct_normalized": safe_to_float(alt_highest),
            "alt_score_points": alt_points,
            "raw_model_output": resp_text
        })

        progress.progress(int(i/n*100))
        # polite sleep
        time.sleep(per_call_sleep)

    # assemble results dataframe
    out_df = pd.DataFrame(results)
    st.subheader("Results (first 50 rows)")
    st.dataframe(out_df.head(50))

    # download buttons
    st.download_button("Download CSV", data=out_df.to_csv(index=False).encode("utf-8"), file_name=OUTPUT_CSV, mime="text/csv")
    st.download_button("Download JSON", data=out_df.to_json(orient="records", indent=2).encode("utf-8"), file_name=OUTPUT_JSON, mime="application/json")

    st.success("Batch extraction complete.")
