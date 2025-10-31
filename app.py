# app.py
"""
Streamlit app — dataset-level extractor using Gemini 2.5-flash.

This version implements MASH resolution logic:
- The model is asked to find the highest reported % of patients with MASH resolution
  across all abstracts and whether any abstracts report worsening of fibrosis.
- Scoring for MASH follows the user's rules:
    5: >=50% resolution with no worsening of fibrosis
    4: >=30% resolution with no worsening of fibrosis
    3: resolution reported but some worsening of fibrosis
    2: mixed or ambiguous data on resolution
    1: No resolution
- The app still extracts dataset-level weight-loss and A1c averages and scores them.

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

# Prompt: request dataset-level averages for weight/A1c and MASH highest resolution + fibrosis worsening
DATASET_PROMPT_TEMPLATE = (
    "You are an extractor for clinical research abstracts. I will provide a numbered list of "
    "abstracts from multiple studies.\n\n"
    "Read all abstracts and return exactly one JSON object (single-line) with the following keys:\n\n"
    "1) average_weight_loss_pct -> numeric mean of reported weight-loss percentages across abstracts that report it (null if none).\n\n"
    "2) average_a1c_reduction_pct -> numeric mean of reported A1c absolute reductions (percentage points) across abstracts that report it (null if none).\n\n"
    "3) mash_highest_resolution_pct -> a single numeric value equal to the HIGHEST percentage of patients reported in any study to have MASH resolution (percent from baseline). "
    "If no studies report a percent, return null. If a study reports a range (e.g., 20-30%), provide your best single-number estimate (prefer midpoint) and document in 'mash_notes'.\n\n"
    "4) mash_worsening_of_fibrosis -> one of the strings 'yes', 'no', or 'unclear' indicating whether any abstracts reported worsening of fibrosis in patients (choose 'yes' if any worsening is reported, 'no' if none report worsening, 'unclear' if ambiguous).\n\n"
    "5) mash_notes -> a short one-sentence caveat describing how you interpreted resolution percentages and fibrosis worsening (e.g., 'one study reported 60% resolution but also mentioned fibrosis progression in 5%').\n\n"
    "6) mash_resolution_counts -> optional object with integer counts (yes/no/unclear) for resolution if you provide them.\n\n"
    "7) notes -> a short one-sentence caveat for other endpoints if needed; otherwise an empty string.\n\n"
    "IMPORTANT: Return only the single JSON object and nothing else.\n\n"
    "Now analyze these abstracts (they are numbered to match your output if needed):\n\n"
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
    if s in {"unclear", "unknown", "maybe", "ambiguous", "unsure"}:
        return "unclear"
    # sometimes models return longer phrases
    if "worsen" in s or "progress" in s or "fibrosis increased" in s or "fibrosis progression" in s:
        return "yes"
    if "no worsen" in s or "no progression" in s or "no fibrosis worsening" in s or "no progression noted" in s:
        return "no"
    # default to 'unclear' if unsure
    return "unclear"

# --- Weight loss scoring ---
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


# --- A1c scoring ---
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


# --- MASH scoring according to the user's rules ---
def score_mash(highest_pct: Optional[float], fibrosis_status: Optional[str], ambiguous_flag: Optional[bool]) -> int:
    """
    Scoring rules (user-specified):
      5: >=50% resolution with no worsening of fibrosis
      4: >=30% resolution with no worsening of fibrosis
      3: resolution reported but some worsening of fibrosis
      2: mixed or ambiguous data on resolution
      1: No resolution

    Logic implemented:
      - If ambiguous_flag is True -> 2
      - If highest_pct is None or highest_pct == 0 -> 1
      - If fibrosis_status == 'no' and highest_pct >= 50 -> 5
      - If fibrosis_status == 'no' and highest_pct >= 30 -> 4
      - If highest_pct and fibrosis_status == 'yes' -> 3
      - If highest_pct and fibrosis_status == 'unclear' -> 2
      - Otherwise fallback to 2 if conflicting info, else 1
    """
    # ambiguous marker (model explicitly says 'mixed' or 'ambiguous')
    if ambiguous_flag:
        return 2

    # normalize
    pct = safe_to_float(highest_pct)
    fib = normalize_yes_no_unclear(fibrosis_status)

    if pct is None or (isinstance(pct, float) and pct == 0.0):
        # nothing reported as resolution
        return 1

    # pct is numeric and >0
    if fib == "no":
        if pct >= 50:
            return 5
        if pct >= 30:
            return 4
        # pct <30 but fibrosis explicitly not worsened -> treat as ambiguous/mixed (2)
        return 2
    elif fib == "yes":
        # resolution reported but fibrosis worsened in some patients
        return 3
    elif fib == "unclear" or fib is None:
        return 2

    # fallback
    return 2

# ----------------------------
# Streamlit UI and main flow
# ----------------------------
st.set_page_config(page_title="Dataset-level Outcome Extractor", layout="wide")
st.title("Dataset-level Outcome Extractor — Gemini 2.5-flash (MASH resolution scoring)")
st.write(
    "Upload a CSV with a column named `abstract` (case-insensitive). "
    "The app will ask Gemini to read all abstracts and return dataset-level "
    "averages for weight and A1c, and MASH highest-resolution + fibrosis-worsening "
    "info used to compute a MASH score."
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

    st.subheader("Model-extracted fields (raw parsed JSON)")
    st.json(parsed)

    # ------------------------
    # Compute scores (weight, A1c, MASH)
    # ------------------------
    avg_wt = safe_to_float(parsed.get("average_weight_loss_pct"))
    weight_points = score_weight_loss(avg_wt)

    avg_a1c = safe_to_float(parsed.get("average_a1c_reduction_pct"))
    a1c_points = score_a1c_reduction(avg_a1c)

    # MASH fields
    # Model should return 'mash_highest_resolution_pct' (numeric), 'mash_worsening_of_fibrosis' (yes/no/unclear),
    # optionally 'mash_ambiguous' (True/False) or we infer ambiguity from notes.
    mash_highest_pct = parsed.get("mash_highest_resolution_pct")
    mash_worsening_raw = parsed.get("mash_worsening_of_fibrosis")
    mash_notes = parsed.get("mash_notes") if parsed.get("mash_notes") is not None else parsed.get("notes")

    # Some models may use different keys; try to detect alternates
    if mash_highest_pct is None:
        # try other possible key names
        for k in parsed.keys():
            kl = k.lower()
            if "highest" in kl and "mash" in kl and "pct" in kl:
                mash_highest_pct = parsed.get(k)
                break
            if "resolution" in kl and "pct" in kl and "mash" in kl:
                mash_highest_pct = parsed.get(k)
                break

    if mash_worsening_raw is None:
        for k in parsed.keys():
            kl = k.lower()
            if "worsen" in kl or "fibrosis" in kl:
                mash_worsening_raw = parsed.get(k)
                break

    # ambiguous detection: if model explicitly returned a field 'mash_ambiguous' or 'mash_mixed' or notes contain 'mixed'/'ambiguous'
    mash_ambiguous_flag = False
    if isinstance(parsed.get("mash_ambiguous"), bool):
        mash_ambiguous_flag = parsed.get("mash_ambiguous")
    elif isinstance(parsed.get("mash_mixed"), bool):
        mash_ambiguous_flag = parsed.get("mash_mixed")
    else:
        # look for keywords in notes
        note_text = ""
        if mash_notes:
            note_text = str(mash_notes).lower()
        if "mixed" in note_text or "ambiguous" in note_text or "conflicting" in note_text:
            mash_ambiguous_flag = True

    # normalize highest pct to float if possible
    mash_highest_pct_norm = safe_to_float(mash_highest_pct)
    mash_worsening_norm = normalize_yes_no_unclear(mash_worsening_raw)

    mash_points = score_mash(mash_highest_pct_norm, mash_worsening_norm, mash_ambiguous_flag)

    # Build output JSON with scores and mash diagnostics
    parsed_with_scores = dict(parsed)  # shallow copy
    parsed_with_scores.update(
        {
            "average_weight_loss_pct_normalized": avg_wt,
            "average_a1c_reduction_pct_normalized": avg_a1c,
            "weight_score_points": weight_points,
            "a1c_score_points": a1c_points,
            "mash_highest_resolution_pct_normalized": mash_highest_pct_norm,
            "mash_worsening_of_fibrosis_normalized": mash_worsening_norm,
            "mash_ambiguous_flag": mash_ambiguous_flag,
            "mash_notes_normalized": mash_notes,
            "mash_score_points": mash_points,
        }
    )

    # Totals: by default weight + a1c = 10 (unchanged)
    total_points = weight_points + a1c_points
    max_points = 10

    parsed_with_scores.update({"total_points": total_points, "max_points": max_points})

    st.subheader("Parsed JSON (dataset-level outcomes + scores + MASH diagnostics)")
    st.json(parsed_with_scores)

    # Scoring summary display
    st.subheader("Scoring summary")
    st.markdown(
        f"- Weight-loss average (model): **{avg_wt if avg_wt is not None else 'N/A'}**  → points: **{weight_points} / 5**\n"
        f"- A1c reduction average (model): **{avg_a1c if avg_a1c is not None else 'N/A'}**  → points: **{a1c_points} / 5**\n"
        f"- MASH highest resolution (model): **{mash_highest_pct_norm if mash_highest_pct_norm is not None else 'N/A'}**\n"
        f"- MASH fibrosis worsening (model): **{mash_worsening_norm if mash_worsening_norm is not None else 'N/A'}**\n"
        f"- MASH ambiguous flag: **{mash_ambiguous_flag}**\n"
        f"- MASH score: **{mash_points} / 5**\n"
        f"- **Total points (weight + A1c): {total_points} / {max_points}**"
    )

    # Download button (includes scores)
    out_bytes = json.dumps(parsed_with_scores, indent=2).encode("utf-8")
    st.download_button(label="Download outcomes JSON (with scores)", data=out_bytes, file_name=OUTPUT_JSON, mime="application/json")

    st.success("Analysis complete.")
