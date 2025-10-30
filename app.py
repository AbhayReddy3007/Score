# app.py
"""
Streamlit app — dataset-level extractor using Gemini 2.5-flash.

Instructions:
- Put your Gemini API key in Streamlit secrets (.streamlit/secrets.toml):
    GEMINI_API_KEY = "your_real_gemini_api_key_here"
  Or export GEMINI_API_KEY as an environment variable (fallback).

- Install dependencies:
    pip install streamlit pandas google-genai

- Run:
    streamlit run app.py
"""
import os
import json
from typing import Optional, Any

import streamlit as st
import pandas as pd

# lazy import genai to give helpful error if missing
try:
    from google import genai
except Exception:
    genai = None

# CONFIG
MODEL_NAME = "gemini-2.5-flash"
OUTPUT_JSON = "overall_outcomes.json"

# Prompt used for the whole dataset: ask for a single JSON object output
DATASET_PROMPT_TEMPLATE = (
    "You are an extractor for clinical research abstracts. I will provide a numbered list of "
    "abstracts from multiple studies.\n"
    "Read all abstracts and answer with exactly one JSON object (single-line) that contains the "
    "following keys:\n\n"
    "1) average_weight_loss_pct -> numeric mean of reported weight-loss percentages across the "
    "abstracts that report it. If no abstracts report weight-loss %, return null.\n\n"
    "2) average_a1c_reduction_pct -> numeric mean of reported A1c absolute reductions (percentage "
    "points) across abstracts that report it. If abstracts report relative percent reductions treat "
    "them as percent values. If no abstracts report A1c change, return null.\n\n"
    "3) mash_resolution_counts -> object with integer counts of abstracts that reported MASH/NASH "
    "resolution as yes/no/unclear. Example: {\"yes\": 10, \"no\": 3, \"unclear\": 187}\n\n"
    "4) alt_reduction_counts -> object with integer counts of abstracts reporting ALT "
    "reduction/normalization as yes/no/unclear. Example: {\"yes\": 12, \"no\": 5, \"unclear\": 183}\n\n"
    "5) notes -> a short one-sentence caveat if needed (e.g., many abstracts lack numeric details), "
    "otherwise an empty string.\n\n"
    "If possible, also provide average_alt_reduction_pct (numeric mean of reported ALT %-reduction from baseline) "
    "or counts in ALT-% buckets (e.g., count_>50, count_30_50, count_15_29, count_1_15, count_<=0) if you can extract them. "
    "Return only the JSON object and nothing else.\n\n"
    "Now analyze these abstracts:\n\n"
    "-----\n"
    "{all_abstracts}\n"
    "-----\n"
)

# ----------------------------
# Helper functions
# ----------------------------
def extract_json(text: str) -> Optional[dict]:
    """Extract a single JSON object from model text output as robustly as possible."""
    if not isinstance(text, str):
        return None
    s = text.strip()

    # remove code fences if present
    if s.startswith("```") and s.endswith("```"):
        parts = s.split("\n")
        if len(parts) > 2:
            inner = "\n".join(parts[1:-1]).strip()
            if inner:
                s = inner

    # find first { and last }
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        # try direct json load of entire string
        try:
            return json.loads(s)
        except Exception:
            return None

    candidate = s[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        # small cleanup attempts
        try:
            cleaned = candidate.replace(",}", "}")
            return json.loads(cleaned)
        except Exception:
            return None


def safe_to_float(val: Any) -> Optional[float]:
    """Try to convert value to float. Return None if not possible."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        # guard against NaN
        try:
            return float(val)
        except Exception:
            return None
    if isinstance(val, str):
        s = val.strip()
        if s == "" or s.lower() in {"null", "none", "n/a", "na"}:
            return None
        s = s.replace("%", "")
        # remove commas in numbers like "1,234.5"
        s = s.replace(",", "")
        try:
            return float(s)
        except Exception:
            return None
    return None


# --- Weight loss scoring (existing) ---
def score_weight_loss(pct: Optional[float]) -> int:
    """
    Weight loss % scoring:
      5: >=22
      4: 16 - 21.9
      3: 10 - 15.9
      2: 5 - 9.9
      1: <5
    If pct is None -> return 0
    """
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


# --- A1c scoring (existing) ---
def score_a1c_reduction(pct: Optional[float]) -> int:
    """
    A1c Reduction % scoring (absolute percentage points):
      5: >=2.2
      4: 1.8 - 2.1
      3: 1.2 - 1.7
      2: 0.8 - 1.1
      1: <0.8
    If pct is None -> return 0
    """
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


# --- ALT reduction scoring (new) ---
def score_alt_avg_pct(avg_pct: Optional[float]) -> int:
    """
    ALT reduction scoring by percent reduction from baseline:
      5: >50% reduction from baseline
      4: >30 - 50% reduction from baseline
      3: >15 - 29% reduction from baseline
      2: >0 - 15% reduction from baseline
      1: No reduction or increase (<= 0)
    Notes:
      - If avg_pct is None -> return 0
      - The original user text had "2:>15% reduction from baseline" which is ambiguous.
        Here we interpret ranges to be mutually exclusive and cover the space:
          5: avg > 50
          4: 30 < avg <= 50
          3: 15 < avg <= 30
          2: 0 < avg <= 15
          1: avg <= 0
    """
    if avg_pct is None:
        return 0
    if avg_pct > 50:
        return 5
    if avg_pct > 30:
        return 4
    if avg_pct > 15:
        return 3
    if avg_pct > 0:
        return 2
    return 1


def per_study_alt_score(pct: Optional[float]) -> Optional[int]:
    """Map a single study's ALT % reduction to a 1-5 score (None if pct is None)."""
    if pct is None:
        return None
    if pct > 50:
        return 5
    if pct > 30:
        return 4
    if pct > 15:
        return 3
    if pct > 0:
        return 2
    return 1


# ----------------------------
# Streamlit UI and main flow
# ----------------------------
st.set_page_config(page_title="Dataset-level Outcome Extractor", layout="wide")
st.title("Dataset-level Outcome Extractor — Gemini 2.5-flash")
st.write(
    "Upload a CSV with a column named `abstract` (case-insensitive). "
    "The app will analyze all abstracts together and return dataset-level outcomes."
)

uploaded_file = st.file_uploader("Upload CSV file with abstracts", type=["csv"])
if not uploaded_file:
    st.info("Upload a CSV to begin. Example header: 'abstract'")
    st.stop()

# Read CSV
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read uploaded CSV: {e}")
    st.stop()

# Locate abstract column (case-insensitive)
abstract_col = None
for c in df.columns:
    if c.strip().lower() == "abstract":
        abstract_col = c
        break

if abstract_col is None:
    st.error("CSV must contain an 'abstract' column (case-insensitive).")
    st.stop()

st.success(f"Found abstract column '{abstract_col}' — {len(df)} rows.")
st.markdown("#### Preview (first 5 abstracts)")
for i, val in enumerate(df[abstract_col].astype(str).head(5), start=1):
    st.write(f"{i}. {' '.join(val.split())}")

# Check if CSV contains per-study ALT reduction % column (case-insensitive)
alt_pct_col = None
for c in df.columns:
    if c.strip().lower() in {"alt_reduction_pct", "alt_reduction_%", "alt_pct", "alt_percent"}:
        alt_pct_col = c
        break

if alt_pct_col:
    st.info(f"Found ALT percentage column '{alt_pct_col}'. App will compute ALT scoring from CSV values when available.")
else:
    st.info("No per-study ALT percentage column found in CSV. The app will try to use Gemini's average_alt_reduction_pct (if returned).")

# Prepare combined numbered abstracts
all_abstracts = []
for i, a in enumerate(df[abstract_col].astype(str), start=1):
    text = " ".join(a.split())  # collapse whitespace
    all_abstracts.append(f"{i}. {text}")
combined = "\n\n".join(all_abstracts)

# API key: check Streamlit secrets then environment
api_key = None
try:
    api_key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
except Exception:
    api_key = None

if not api_key:
    api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error(
        "No Gemini API key found. Set GEMINI_API_KEY in Streamlit secrets (.streamlit/secrets.toml) "
        "or as an environment variable."
    )
    st.stop()

if genai is None:
    st.error("Missing dependency: google-genai. Install with `pip install google-genai`.")
    st.stop()

# Option: include ALT in total score
include_alt_in_total = st.checkbox("Include ALT endpoint in total points (adds up to +5)", value=False)

# Button to analyze dataset
if st.button("Analyze dataset"):
    client = genai.Client(api_key=api_key)
    prompt = DATASET_PROMPT_TEMPLATE.replace("{all_abstracts}", combined)

    with st.spinner("Sending dataset to Gemini and waiting for response — this may take a moment..."):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        except Exception as e:
            st.error(f"API error when calling Gemini: {e}")
            st.stop()

    # Extract response text robustly
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

    st.subheader("Raw model output")
    st.code(resp_text)

    parsed = extract_json(resp_text)
    if parsed is None:
        st.error("Could not parse JSON from model response. Inspect the raw output above.")
        st.stop()

    # Existing keys we expect
    expected_keys = [
        "average_weight_loss_pct",
        "average_a1c_reduction_pct",
        "mash_resolution_counts",
        "alt_reduction_counts",
        "notes",
    ]
    missing = [k for k in expected_keys if k not in parsed]
    if missing:
        st.warning(f"Model returned JSON but missing expected keys: {missing} — displaying what was returned.")

    # --- Gather/compute values for weight, a1c, alt ---
    # Weight (from model)
    avg_wt = safe_to_float(parsed.get("average_weight_loss_pct"))
    weight_points = score_weight_loss(avg_wt)

    # A1c (from model)
    avg_a1c = safe_to_float(parsed.get("average_a1c_reduction_pct"))
    a1c_points = score_a1c_reduction(avg_a1c)

    # ALT: priority order for determining ALT score:
    # 1) If CSV contains per-study ALT pct column -> compute from CSV (preferred).
    # 2) Else if model returned average_alt_reduction_pct -> score that.
    # 3) Else if model returned counts per ALT-% bucket (like count_>50 etc.) -> compute weighted/majority score.
    # 4) Else fall back to alt_reduction_counts yes/no/unclear -> cannot reliably score by percent; set to 0.
    alt_points = 0
    alt_avg_used = None
    alt_method = "none"

    # 1) CSV per-study column
    if alt_pct_col:
        # convert to numeric
        df_alt_numeric = df[alt_pct_col].apply(safe_to_float)
        non_na = df_alt_numeric.dropna()
        if len(non_na) > 0:
            # option A: compute dataset average percent and score it
            avg_alt_pct = float(non_na.mean())
            alt_points = score_alt_avg_pct(avg_alt_pct)
            alt_avg_used = avg_alt_pct
            alt_method = f"csv_column:{alt_pct_col}"
        else:
            # column present but no numeric values
            alt_method = f"csv_column:{alt_pct_col}_no_numeric"

    # 2) try average_alt_reduction_pct from model
    if alt_method == "none" or alt_method.startswith("csv_column") and alt_method.endswith("_no_numeric"):
        avg_alt_from_model = None
        # model might return a dedicated key
        if "average_alt_reduction_pct" in parsed:
            avg_alt_from_model = safe_to_float(parsed.get("average_alt_reduction_pct"))
        # sometimes model might embed it under another name; check a few possibilities:
        if avg_alt_from_model is None:
            for k in parsed.keys():
                if k.lower().startswith("average") and "alt" in k.lower():
                    avg_alt_from_model = safe_to_float(parsed.get(k))
                    if avg_alt_from_model is not None:
                        break

        if avg_alt_from_model is not None:
            alt_points = score_alt_avg_pct(avg_alt_from_model)
            alt_avg_used = avg_alt_from_model
            alt_method = "model_average_alt_reduction_pct"

    # 3) try bucket counts returned by model (count_>50, count_30_50, count_15_29, count_1_15, count_<=0)
    if alt_method in {"none", "csv_column:alt_reduction_pct_no_numeric", "csv_column:alt_reduction_pct"}:
        # check for bucket keys in parsed JSON
        bucket_keys = {
            "count_>50": 5,
            "count_>50%": 5,
            "count_30_50": 4,
            "count_30-50": 4,
            "count_15_29": 3,
            "count_15-29": 3,
            "count_1_15": 2,
            "count_1-15": 2,
            "count_<=0": 1,
            "count_< =0": 1,
            "count_le_0": 1,
        }
        # gather counts per bucket if present
        bucket_counts = {}
        for k in parsed.keys():
            key_lower = k.lower().replace(" ", "_")
            if key_lower in [bk.lower() for bk in bucket_keys.keys()]:
                # exact match normalized
                bucket_counts[key_lower] = int(parsed.get(k)) if parsed.get(k) is not None else 0
        # More flexible detection: find keys that contain 'count' and a percent bracket
        if not bucket_counts:
            for k, v in parsed.items():
                if isinstance(v, (int, float)):
                    kl = k.lower()
                    if "count" in kl and (">50" in kl or "30" in kl or "15" in kl or "<=0" in kl or "1-15" in kl):
                        bucket_counts[kl] = int(v)
        if bucket_counts:
            # map detected bucket keys to scores using heuristics; choose the bucket with highest count
            mapped = []
            for k, v in bucket_counts.items():
                # map key to score
                score = None
                if ">50" in k:
                    score = 5
                elif "30" in k and "50" in k:
                    score = 4
                elif "15" in k and ("29" in k or "30" in k):
                    score = 3
                elif "1" in k and "15" in k:
                    score = 2
                elif "<=0" in k or "<0" in k or "no" in k:
                    score = 1
                if score is not None:
                    mapped.append((score, int(v)))
            if mapped:
                # choose the score with maximum count
                mapped_sorted = sorted(mapped, key=lambda x: x[1], reverse=True)
                alt_points = int(mapped_sorted[0][0])
                alt_method = "model_bucket_counts"
                # compute a weighted average score (optional) and set alt_avg_used to None
                alt_avg_used = None

    # 4) fallback: alt_reduction_counts yes/no/unclear exists but cannot map to percent bins
    if alt_method == "none":
        if isinstance(parsed.get("alt_reduction_counts"), dict):
            # We cannot reliably determine percent-based bin from yes/no/unclear counts.
            alt_points = 0
            alt_method = "alt_reduction_counts_only"
        else:
            alt_points = 0
            alt_method = "no_alt_info"

    # Build parsed_with_scores JSON (non-destructive)
    parsed_with_scores = dict(parsed)  # shallow copy
    parsed_with_scores.update(
        {
            "average_weight_loss_pct_normalized": avg_wt,
            "average_a1c_reduction_pct_normalized": avg_a1c,
            "weight_score_points": weight_points,
            "a1c_score_points": a1c_points,
            "alt_score_points": alt_points,
            "alt_avg_used": alt_avg_used,
            "alt_detection_method": alt_method,
        }
    )

    # Compute total points: if include_alt_in_total is checked, include alt_points
    total_points = weight_points + a1c_points
    max_points = 10  # weight (5) + a1c (5)
    if include_alt_in_total:
        total_points += alt_points
        max_points += 5

    parsed_with_scores.update({"total_points": total_points, "max_points": max_points})

    # Display parsed JSON and scores
    st.subheader("Parsed JSON (dataset-level outcomes + scores)")
    st.json(parsed_with_scores)

    # Show a small summary box for scores
    st.subheader("Scoring summary")
    st.markdown(
        f"- Weight-loss average (model): **{avg_wt if avg_wt is not None else 'N/A'}**  → points: **{weight_points} / 5**\n"
        f"- A1c reduction average (model): **{avg_a1c if avg_a1c is not None else 'N/A'}**  → points: **{a1c_points} / 5**\n"
        f"- ALT reduction: method **{alt_method}**, used average: **{alt_avg_used if alt_avg_used is not None else 'N/A'}** → points: **{alt_points} / 5**\n"
        f"- **Total points: {total_points} / {max_points}**"
    )

    # Download button (includes scores)
    out_bytes = json.dumps(parsed_with_scores, indent=2).encode("utf-8")
    st.download_button(label="Download outcomes JSON (with scores)", data=out_bytes, file_name=OUTPUT_JSON, mime="application/json")

    st.success("Analysis complete.")
