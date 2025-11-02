# app.py
"""
Streamlit app — dataset-level extractor using Gemini 2.5-flash.

This version:
- Extracts dataset-level averages for weight and A1c.
- Implements MASH scoring (highest resolution % across abstracts + fibrosis worsening).
- Implements ALT scoring (highest ALT %-reduction across abstracts) with user-provided bins.
- Returns diagnostics and scores for each endpoint. By default total_points = weight + A1c (10).
  ALT and MASH scores are computed and returned separately; include them in totals if you ask.

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
from typing import Optional, Any, Dict

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

# Prompt: request dataset-level averages and MASH + ALT highest-reported percentages + diagnostics
DATASET_PROMPT_TEMPLATE = (
    "You are an extractor for clinical research abstracts. I will provide a numbered list of "
    "abstracts from multiple studies.\n\n"
    "Read all abstracts and return exactly one JSON object (single-line) with the following keys:\n\n"
    "1) average_weight_loss_pct -> numeric mean of reported weight-loss percentages across abstracts that report it (null if none).\n\n"
    "2) average_a1c_reduction_pct -> numeric mean of reported A1c absolute reductions (percentage points) across abstracts that report it (null if none).\n\n"
    "MASH-specific (single dataset-level fields):\n"
    "3) mash_highest_resolution_pct -> the HIGHEST percentage of patients reported in any study to have MASH resolution (percent from baseline). Return null if none reported.\n\n"
    "4) mash_worsening_of_fibrosis -> one of 'yes','no','unclear' indicating whether any abstracts reported worsening of fibrosis (choose 'yes' if any worsening is reported, 'no' if none, 'unclear' if ambiguous).\n\n"
    "5) mash_notes -> brief caveat about MASH extraction (one sentence) or empty string.\n\n"
    "ALT-specific (single dataset-level fields):\n"
    "6) alt_highest_reduction_pct -> the HIGHEST percentage reduction in ALT from baseline reported in any abstract (percent). Return null if none reported. If ranges are reported, provide your best single-number estimate (prefer midpoint) and document in 'alt_notes'.\n\n"
    "7) alt_notes -> brief caveat about ALT extraction (one sentence) or empty string.\n\n"
    "8) mash_resolution_counts -> optional object with integer counts of abstracts that reported MASH resolution yes/no/unclear.\n\n"
    "9) alt_reduction_counts -> optional object with integer counts of abstracts that reported ALT reduction yes/no/unclear.\n\n"
    "10) notes -> a short one-sentence caveat for other endpoints if needed; otherwise an empty string.\n\n"
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
    """Normalize model-provided yes/no/unclear-like values to 'yes'/'no'/'unclear' or None."""
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
    # look for keywords
    if "worsen" in s or "progress" in s or "fibrosis increased" in s or "fibrosis progression" in s:
        return "yes"
    if "no worsen" in s or "no progression" in s or "no fibrosis worsening" in s or "no progression noted" in s:
        return "no"
    # default to 'unclear'
    return "unclear"

# --- Weight loss scoring ---
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


# --- A1c scoring ---
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


# --- MASH scoring (user rules) ---
def score_mash(highest_pct: Optional[float], fibrosis_status: Optional[str], ambiguous_flag: Optional[bool]) -> int:
    """
    MASH scoring rules:
      5: >=50% resolution with no worsening of fibrosis
      4: >=30% resolution with no worsening of fibrosis
      3: resolution reported but some worsening of fibrosis
      2: mixed or ambiguous data on resolution
      1: No resolution
    """
    if ambiguous_flag:
        return 2

    pct = safe_to_float(highest_pct)
    fib = normalize_yes_no_unclear(fibrosis_status)

    if pct is None or (isinstance(pct, float) and pct == 0.0):
        return 1

    if fib == "no":
        if pct >= 50:
            return 5
        if pct >= 30:
            return 4
        return 2  # resolution <30 but no fibrosis worsening -> treat as mixed
    if fib == "yes":
        return 3
    if fib == "unclear" or fib is None:
        return 2

    return 2


# --- ALT scoring (user bins for highest ALT %-reduction) ---
def score_alt_highest_pct(highest_pct: Optional[float], ambiguous_flag: Optional[bool]) -> int:
    """
    ALT scoring (based on highest reported ALT %-reduction across abstracts):
      5: >50% reduction from baseline
      4: 30 - 50% reduction from baseline
      3: 15 - 29% reduction from baseline
      2: <15% reduction from baseline (but >0)
      1: No reduction from baseline (<=0 or none reported)

    ambiguous_flag: if True -> treat as score 2 (mixed/ambiguous)
    """
    if ambiguous_flag:
        return 2
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
st.title("Dataset-level Outcome Extractor — Gemini 2.5-flash (MASH & ALT scoring)")
st.write(
    "Upload a CSV with a column named `abstract` (case-insensitive). "
    "The app will ask Gemini to read all abstracts and return dataset-level "
    "averages for weight and A1c, plus MASH and ALT highest reported percentages used to compute scores."
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
    # Compute scores (weight, A1c, MASH, ALT)
    # ------------------------
    avg_wt = safe_to_float(parsed.get("average_weight_loss_pct"))
    weight_points = score_weight_loss(avg_wt)

    avg_a1c = safe_to_float(parsed.get("average_a1c_reduction_pct"))
    a1c_points = score_a1c_reduction(avg_a1c)

    # MASH fields detection & scoring
    mash_highest = parsed.get("mash_highest_resolution_pct")
    mash_worsening_raw = parsed.get("mash_worsening_of_fibrosis")
    mash_notes = parsed.get("mash_notes") if parsed.get("mash_notes") is not None else parsed.get("notes")

    # fallback key detection for MASH highest pct & worsening
    if mash_highest is None:
        for k in parsed.keys():
            kl = k.lower()
            if "mash" in kl and "highest" in kl and "pct" in kl:
                mash_highest = parsed.get(k)
                break
            if "highest" in kl and "resolution" in kl and "mash" in kl:
                mash_highest = parsed.get(k)
                break

    if mash_worsening_raw is None:
        for k in parsed.keys():
            kl = k.lower()
            if "worsen" in kl or "fibrosis" in kl:
                mash_worsening_raw = parsed.get(k)
                break

    # ambiguous detection for MASH
    mash_ambiguous_flag = False
    if isinstance(parsed.get("mash_ambiguous"), bool):
        mash_ambiguous_flag = parsed.get("mash_ambiguous")
    elif isinstance(parsed.get("mash_mixed"), bool):
        mash_ambiguous_flag = parsed.get("mash_mixed")
    else:
        note_text = ""
        if mash_notes:
            note_text = str(mash_notes).lower()
        if "mixed" in note_text or "ambiguous" in note_text or "conflicting" in note_text:
            mash_ambiguous_flag = True

    mash_highest_norm = safe_to_float(mash_highest)
    mash_worsening_norm = normalize_yes_no_unclear(mash_worsening_raw)
    mash_points = score_mash(mash_highest_norm, mash_worsening_norm, mash_ambiguous_flag)

    # ALT fields detection & scoring (mirror MASH logic but only percent-based)
    alt_highest = parsed.get("alt_highest_reduction_pct")
    alt_notes = parsed.get("alt_notes") if parsed.get("alt_notes") is not None else parsed.get("notes")

    # fallback key detection for ALT highest pct
    if alt_highest is None:
        for k in parsed.keys():
            kl = k.lower()
            if "alt" in kl and ("highest" in kl or "max" in kl) and "pct" in kl:
                alt_highest = parsed.get(k)
                break
            if "alt" in kl and "reduction" in kl and "pct" in kl:
                alt_highest = parsed.get(k)
                break

    # ambiguous detection for ALT
    alt_ambiguous_flag = False
    if isinstance(parsed.get("alt_ambiguous"), bool):
        alt_ambiguous_flag = parsed.get("alt_ambiguous")
    elif isinstance(parsed.get("alt_mixed"), bool):
        alt_ambiguous_flag = parsed.get("alt_mixed")
    else:
        alt_note_text = ""
        if alt_notes:
            alt_note_text = str(alt_notes).lower()
        if "mixed" in alt_note_text or "ambiguous" in alt_note_text or "conflicting" in alt_note_text:
            alt_ambiguous_flag = True

    alt_highest_norm = safe_to_float(alt_highest)
    alt_points = score_alt_highest_pct(alt_highest_norm, alt_ambiguous_flag)

    # Build output JSON with scores and diagnostics
    parsed_with_scores = dict(parsed)  # shallow copy
    parsed_with_scores.update(
        {
            "average_weight_loss_pct_normalized": avg_wt,
            "average_a1c_reduction_pct_normalized": avg_a1c,
            "weight_score_points": weight_points,
            "a1c_score_points": a1c_points,
            "mash_highest_resolution_pct_normalized": mash_highest_norm,
            "mash_worsening_of_fibrosis_normalized": mash_worsening_norm,
            "mash_ambiguous_flag": mash_ambiguous_flag,
            "mash_notes_normalized": mash_notes,
            "mash_score_points": mash_points,
            "alt_highest_reduction_pct_normalized": alt_highest_norm,
            "alt_ambiguous_flag": alt_ambiguous_flag,
            "alt_notes_normalized": alt_notes,
            "alt_score_points": alt_points,
        }
    )

    # Totals: default remains weight + a1c = 10 (unchanged)
    total_points = weight_points + a1c_points
    max_points = 10

    parsed_with_scores.update({"total_points": total_points, "max_points": max_points})

    st.subheader("Parsed JSON (dataset-level outcomes + scores + diagnostics)")
    st.json(parsed_with_scores)

    # Scoring summary display
    st.subheader("Scoring summary")
    st.markdown(
        f"- Weight-loss average (model): **{avg_wt if avg_wt is not None else 'N/A'}**  → points: **{weight_points} / 5**\n"
        f"- A1c reduction average (model): **{avg_a1c if avg_a1c is not None else 'N/A'}**  → points: **{a1c_points} / 5**\n\n"
        f"- MASH highest resolution (model): **{mash_highest_norm if mash_highest_norm is not None else 'N/A'}**\n"
        f"  - Fibrosis worsening: **{mash_worsening_norm if mash_worsening_norm is not None else 'N/A'}**\n"
        f"  - Ambiguous flag: **{mash_ambiguous_flag}**\n"
        f"  - MASH score: **{mash_points} / 5**\n\n"
        f"- ALT highest reduction (model): **{alt_highest_norm if alt_highest_norm is not None else 'N/A'}**\n"
        f"  - Ambiguous flag: **{alt_ambiguous_flag}**\n"
        f"  - ALT score: **{alt_points} / 5**\n\n"
        f"- **Total points (weight + A1c): {total_points} / {max_points}**"
    )

    # Download button (includes scores)
    out_bytes = json.dumps(parsed_with_scores, indent=2).encode("utf-8")
    st.download_button(label="Download outcomes JSON (with scores)", data=out_bytes, file_name=OUTPUT_JSON, mime="application/json")

    st.success("Analysis complete.")
