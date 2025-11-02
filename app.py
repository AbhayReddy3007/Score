# app.py
"""
Streamlit app — dataset-level extractor using Gemini 2.5-flash.

This version returns ONLY:
- average_weight_loss_pct_normalized
- weight_score_points
- average_a1c_reduction_pct_normalized
- a1c_score_points
- mash_highest_resolution_pct_normalized
- mash_score_points
- alt_highest_reduction_pct_normalized
- alt_score_points

It asks the model to read all abstracts and return dataset-level values.
Scoring rules are hard-coded per user specification.

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

# Prompt: request only the four dataset-level numbers (and optionally fibrosis worsening for MASH scoring)
DATASET_PROMPT_TEMPLATE = (
    "You are an extractor for clinical research abstracts. I will provide a numbered list of "
    "abstracts from multiple studies.\n\n"
    "Read all abstracts and return exactly one JSON object (single-line) with the following keys:\n\n"
    "1) average_weight_loss_pct -> numeric mean of reported weight-loss percentages across abstracts that report it (null if none).\n\n"
    "2) average_a1c_reduction_pct -> numeric mean of reported A1c absolute reductions (percentage points) across abstracts that report it (null if none).\n\n"
    "3) mash_highest_resolution_pct -> the HIGHEST percentage of patients reported in any study to have MASH resolution (percent from baseline). Return null if none reported.\n\n"
    "4) alt_highest_reduction_pct -> the HIGHEST percentage reduction in ALT from baseline reported in any abstract (percent). Return null if none reported.\n\n"
    "Optionally (only if easy to extract), you may include:\n"
    " - mash_worsening_of_fibrosis -> 'yes'/'no'/'unclear' indicating whether any abstracts reported worsening of fibrosis.\n\n"
    "IMPORTANT: Return ONLY the single JSON object and NOTHING else. Do not include extra commentary.\n\n"
    "Now analyze these abstracts:\n\n"
    "-----\n"
    "{all_abstracts}\n"
    "-----\n"
)

# ----------------------------
# Helper functions
# ----------------------------
def extract_json(text: str) -> Optional[dict]:
    """Extract a single JSON object from model text output robustly."""
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
        # try direct json load
        try:
            return json.loads(s)
        except Exception:
            return None

    candidate = s[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
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
        try:
            return float(val)
        except Exception:
            return None
    if isinstance(val, str):
        s = val.strip()
        if s == "" or s.lower() in {"null", "none", "n/a", "na"}:
            return None
        s = s.replace("%", "").replace(",", "")
        try:
            return float(s)
        except Exception:
            return None
    return None


def normalize_yes_no_unclear(val: Any) -> Optional[str]:
    """Normalize model-provided fibrosis worsening strings to 'yes'/'no'/'unclear' or None."""
    if val is None:
        return None
    if isinstance(val, bool):
        return "yes" if val else "no"
    s = str(val).strip().lower()
    if s in {"yes", "y", "true", "t"}:
        return "yes"
    if s in {"no", "n", "false", "f"}:
        return "no"
    if s in {"unclear", "unknown", "maybe", "ambiguous", "unsure", "conflicting", "mixed"}:
        return "unclear"
    if "worsen" in s or "progress" in s or "fibrosis increased" in s or "fibrosis progression" in s:
        return "yes"
    if "no worsen" in s or "no progression" in s or "no fibrosis worsening" in s or "no progression noted" in s:
        return "no"
    return "unclear"


# --- Scoring functions (user-specified bins) ---
def score_weight_loss(pct: Optional[float]) -> int:
    """Weight loss % scoring:
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


def score_a1c_reduction(pct: Optional[float]) -> int:
    """A1c scoring (absolute %-points):
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


def score_mash(highest_pct: Optional[float], fibrosis_status: Optional[str]) -> int:
    """
    MASH scoring per user's earlier rule:
      5: >=50% resolution with no worsening of fibrosis
      4: >=30% resolution with no worsening of fibrosis
      3: resolution reported but some worsening of fibrosis
      2: mixed or ambiguous data on resolution (or fibrosis info missing)
      1: No resolution
    If highest_pct is None -> 1 (No resolution)
    If fibrosis_status missing -> treat as ambiguous -> 2 (unless no resolution)
    """
    pct = safe_to_float(highest_pct)
    fib = normalize_yes_no_unclear(fibrosis_status) if fibrosis_status is not None else None

    if pct is None or (isinstance(pct, float) and pct == 0.0):
        return 1

    if fib is None or fib == "unclear":
        return 2

    if fib == "no":
        if pct >= 50:
            return 5
        if pct >= 30:
            return 4
        # resolution <30 but fibrosis explicitly not worsened -> treat as ambiguous/mixed
        return 2

    if fib == "yes":
        # resolution reported but fibrosis worsened in some patients
        return 3

    return 2


def score_alt(highest_pct: Optional[float]) -> int:
    """
    ALT scoring (based on highest reported ALT %-reduction across abstracts):
      5: >50% reduction from baseline
      4: 30 - 50% reduction from baseline
      3: 15 - 29% reduction from baseline
      2: <15% reduction from baseline (but >0)
      1: No reduction from baseline (<=0 or none reported)
    If highest_pct is None -> 1
    """
    pct = safe_to_float(highest_pct)
    if pct is None or (isinstance(pct, float) and pct <= 0.0):
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
# Streamlit UI and main flow
# ----------------------------
st.set_page_config(page_title="Dataset-level Outcome Extractor", layout="wide")
st.title("Dataset-level Outcome Extractor — minimal outputs + scores")
st.write(
    "Upload a CSV with a column named `abstract` (case-insensitive). "
    "The app will ask Gemini to read all abstracts and return four dataset-level numbers "
    "and their scores."
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

    # Extract the four numeric fields (and optional fibrosis field)
    avg_wt = safe_to_float(parsed.get("average_weight_loss_pct"))
    avg_a1c = safe_to_float(parsed.get("average_a1c_reduction_pct"))
    mash_highest = parsed.get("mash_highest_resolution_pct")
    alt_highest = parsed.get("alt_highest_reduction_pct")

    # try fallback key detection if keys are present under different names
    if mash_highest is None:
        for k in parsed.keys():
            kl = k.lower()
            if "mash" in kl and ("highest" in kl or "max" in kl) and "pct" in kl:
                mash_highest = parsed.get(k)
                break
            if "mash" in kl and "resolution" in kl and "pct" in kl:
                mash_highest = parsed.get(k)
                break

    if alt_highest is None:
        for k in parsed.keys():
            kl = k.lower()
            if "alt" in kl and ("highest" in kl or "max" in kl) and "pct" in kl:
                alt_highest = parsed.get(k)
                break
            if "alt" in kl and "reduction" in kl and "pct" in kl:
                alt_highest = parsed.get(k)
                break

    # optional fibrosis worsening info for MASH scoring
    mash_fibrosis_raw = parsed.get("mash_worsening_of_fibrosis")
    # normalize
    mash_fibrosis_norm = normalize_yes_no_unclear(mash_fibrosis_raw) if mash_fibrosis_raw is not None else None

    # Compute scores
    weight_points = score_weight_loss(avg_wt)
    a1c_points = score_a1c_reduction(avg_a1c)
    mash_points = score_mash(mash_highest, mash_fibrosis_norm)
    alt_points = score_alt(alt_highest)

    # Build minimal JSON with only required normalized numbers and scores
    out = {
        "average_weight_loss_pct_normalized": avg_wt,
        "weight_score_points": weight_points,
        "average_a1c_reduction_pct_normalized": avg_a1c,
        "a1c_score_points": a1c_points,
        "mash_highest_resolution_pct_normalized": safe_to_float(mash_highest),
        "mash_score_points": mash_points,
        "alt_highest_reduction_pct_normalized": safe_to_float(alt_highest),
        "alt_score_points": alt_points,
    }

    st.subheader("Parsed outputs (numbers + scores)")
    st.json(out)

    # Download button (includes only the minimal fields)
    out_bytes = json.dumps(out, indent=2).encode("utf-8")
    st.download_button(label="Download minimal outcomes JSON", data=out_bytes, file_name=OUTPUT_JSON, mime="application/json")

    st.success("Analysis complete.")
