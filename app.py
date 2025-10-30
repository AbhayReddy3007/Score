
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

# Prompt: request dataset-level averages for weight loss and A1c only (no ALT)
DATASET_PROMPT_TEMPLATE = (
    "You are an extractor for clinical research abstracts. I will provide a numbered list of "
    "abstracts from multiple studies.\n\n"
    "Read all abstracts and return exactly one JSON object (single-line) with the following keys:\n\n"
    "1) average_weight_loss_pct -> numeric mean of reported weight-loss percentages across abstracts that report it (null if none).\n\n"
    "2) average_a1c_reduction_pct -> numeric mean of reported A1c absolute reductions (percentage points) across abstracts that report it (null if none).\n\n"
    "3) mash_resolution_counts -> object with integer counts of abstracts that reported MASH/NASH resolution as yes/no/unclear.\n\n"
    "4) notes -> a short one-sentence caveat if needed; otherwise an empty string.\n\n"
    "IMPORTANT: Do NOT include any ALT-related fields. Return only the single JSON object and nothing else.\n\n"
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


# ----------------------------
# Streamlit UI and main flow
# ----------------------------
st.set_page_config(page_title="Dataset-level Outcome Extractor", layout="wide")
st.title("Dataset-level Outcome Extractor — Gemini 2.5-flash (no ALT endpoint)")
st.write(
    "Upload a CSV with a column named `abstract` (case-insensitive). "
    "The app will ask Gemini to read all abstracts and return dataset-level "
    "averages for weight loss and A1c only."
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
    # Compute scores (weight + A1c only)
    # ------------------------
    avg_wt = safe_to_float(parsed.get("average_weight_loss_pct"))
    weight_points = score_weight_loss(avg_wt)

    avg_a1c = safe_to_float(parsed.get("average_a1c_reduction_pct"))
    a1c_points = score_a1c_reduction(avg_a1c)

    # Build output JSON with scores
    parsed_with_scores = dict(parsed)  # shallow copy
    parsed_with_scores.update(
        {
            "average_weight_loss_pct_normalized": avg_wt,
            "average_a1c_reduction_pct_normalized": avg_a1c,
            "weight_score_points": weight_points,
            "a1c_score_points": a1c_points,
        }
    )

    # Total points: weight (5) + a1c (5) = 10 max
    total_points = weight_points + a1c_points
    max_points = 10

    parsed_with_scores.update({"total_points": total_points, "max_points": max_points})

    st.subheader("Parsed JSON (dataset-level outcomes + scores)")
    st.json(parsed_with_scores)

    # Scoring summary display
    st.subheader("Scoring summary")
    st.markdown(
        f"- Weight-loss average (model): **{avg_wt if avg_wt is not None else 'N/A'}**  → points: **{weight_points} / 5**\n"
        f"- A1c reduction average (model): **{avg_a1c if avg_a1c is not None else 'N/A'}**  → points: **{a1c_points} / 5**\n"
        f"- **Total points: {total_points} / {max_points}**"
    )

    # Download button (includes scores)
    out_bytes = json.dumps(parsed_with_scores, indent=2).encode("utf-8")
    st.download_button(label="Download outcomes JSON (with scores)", data=out_bytes, file_name=OUTPUT_JSON, mime="application/json")

    st.success("Analysis complete.")
