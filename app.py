# streamlit_gemini_abstract_extractor.py
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
import time
from typing import Optional

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
DEFAULT_SLEEP = 0.25

# Prompt used for the whole dataset: ask for a single JSON object output
DATASET_PROMPT_TEMPLATE = """
You are an extractor for clinical research abstracts. I will provide a numbered list of abstracts from multiple studies.
Read all abstracts and answer with exactly one JSON object (single-line) that contains the following keys:

1) average_weight_loss_pct -> numeric mean of reported weight-loss percentages across the abstracts that report it. If no abstracts report weight-loss %, return null.

2) average_a1c_reduction_pct -> numeric mean of reported A1c absolute reductions (percentage points) across abstracts that report it. If abstracts report relative percent reductions treat them as percent values. If no abstracts report A1c change, return null.

3) mash_resolution_counts -> object with integer counts of abstracts that reported MASH/NASH resolution as yes/no/unclear. Example: {"yes": 10, "no": 3, "unclear": 187}

4) alt_reduction_counts -> object with integer counts of abstracts reporting ALT reduction/normalization as yes/no/unclear. Example: {"yes": 12, "no": 5, "unclear": 183}

5) notes -> a short one-sentence caveat if needed (e.g., many abstracts lack numeric details), otherwise an empty string.

Return only the JSON object and nothing else.

Now analyze these abstracts:

-----
{all_abstracts}
-----
"""

# Helper: extract JSON object from text (robust)
def extract_json(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    # remove code fences if present
    if s.startswith("```") and s.endswith("``"):
        inner = "
".join(s.split("
")[1:-1]).strip()
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
    candidate = s[start:end+1]
    try:
        return json.loads(candidate)
    except Exception:
        # small cleanup attempts
        try:
            cleaned = candidate.replace(",}", "}")
            return json.loads(cleaned)
        except Exception:
            return None

# Streamlit UI (minimal)
st.set_page_config(page_title="Dataset-level Outcome Extractor", layout="wide")
st.title("Dataset-level Outcome Extractor — Gemini 2.5-flash")
st.write("Upload a CSV with a column named `abstract` (case-insensitive). The app will analyze all abstracts together and return dataset-level outcomes.")

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
combined = "

".join(all_abstracts)

# API key: check Streamlit secrets then environment
api_key = None
try:
    api_key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
except Exception:
    api_key = None

if not api_key:
    api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("No Gemini API key found. Set GEMINI_API_KEY in Streamlit secrets (.streamlit/secrets.toml) or as an environment variable.")
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

    expected_keys = ["average_weight_loss_pct", "average_a1c_reduction_pct", "mash_resolution_counts", "alt_reduction_counts", "notes"]
    missing = [k for k in expected_keys if k not in parsed]
    if missing:
        st.warning(f"Model returned JSON but missing expected keys: {missing} — displaying what was returned.")

    st.subheader("Parsed JSON (dataset-level outcomes)")
    st.json(parsed)

    # Download button
    out_bytes = json.dumps(parsed, indent=2).encode("utf-8")
    st.download_button(label="Download outcomes JSON", data=out_bytes, file_name=OUTPUT_JSON, mime="application/json")

    st.success("Analysis complete.")
