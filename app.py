# streamlit_gemini_abstract_extractor.py
"""
Streamlit app to extract overall structured outcomes from a set of research abstracts using Gemini 2.5-flash.

Changes from previous version:
- Reads **all abstracts at once** and sends a single prompt to Gemini asking for dataset-level/overall outcomes.
- Removes the settings sidebar (no settings window).
- Reads API key ONLY from Streamlit secrets (recommended) or the GEMINI_API_KEY environment variable (fallback). If no key is found the app errors.

Features:
- Upload a CSV with a single column named `abstract` (case-insensitive).
- The app concatenates all abstracts and asks Gemini to return a single JSON object containing dataset-level fields:
  * average_weight_loss_pct (number or null)
  * average_a1c_reduction_pct (number or null)
  * mash_resolution_counts (object with counts: {"yes": int, "no": int, "unclear": int})
  * alt_resolution_counts (object with counts: {"yes": int, "no": int, "unclear": int})
- Displays the returned JSON and allows downloading it as a small results file.

Security note: Store your key in Streamlit secrets: .streamlit/secrets.toml with
    GEMINI_API_KEY = "your_real_gemini_api_key_here"

Requirements:
    pip install streamlit pandas google-genai

Run:
    streamlit run streamlit_gemini_abstract_extractor.py

"""

import os
import json
import time
from typing import Optional

import streamlit as st
import pandas as pd

# lazy import genai so we can show a friendly error
try:
    from google import genai
except Exception:
    genai = None

# ---------------- CONFIG ----------------
MODEL_NAME = "gemini-2.5-flash"
OUTPUT_JSON = "overall_outcomes.json"
DEFAULT_SLEEP = 0.25

# ---------------- PROMPT (for the whole dataset) ----------------
DATASET_PROMPT_TEMPLATE = r"""
You are an extractor for clinical research abstracts. I will provide a numbered list of abstracts from multiple studies.
Read all abstracts and answer with exactly one JSON object (single-line) that contains the following keys:

1) average_weight_loss_pct -> a numeric value representing the mean reported weight-loss percentage across the studies that report a weight loss percent. If no studies report weight-loss % return null.

2) average_a1c_reduction_pct -> a numeric value representing the mean reported A1c absolute reduction (percentage points) across the studies that report it. If abstracts report relative percent reductions, treat them as percent values. If no studies report A1c change, return null.

3) mash_resolution_counts -> an object with integer counts of how many abstracts reported MASH/NASH resolution as yes/no/unclear. Format: {"yes": 10, "no": 3, "unclear": 187}

4) alt_resolution_counts -> an object with integer counts for ALT normalization/resolution reported as yes/no/unclear. Format: {"yes": 12, "no": 5, "unclear": 183}

5) notes -> brief string (one sentence max) if there are important caveats (e.g., many abstracts lack numeric details); otherwise an empty string.

Return ONLY the JSON object and nothing else.

Now analyze these abstracts:

-----
{all_abstracts}
-----
"""

# ---------------- HELPERS ----------------

def extract_json(text: str) -> Optional[dict]:
    """Find and parse a JSON object inside a text response."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    # remove markdown fences if present
    if s.startswith("```") and s.endswith("```"):
        s = "
".join(s.split("
")[1:-1]).strip()
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
        # try quick cleanup
        try:
            cleaned = candidate.replace(',}', '}')
            return json.loads(cleaned)
        except Exception:
            return None

# ---------------- STREAMLIT UI (minimal) ----------------
st.set_page_config(page_title="Abstracts — Dataset-level Extractor", layout="wide")
st.title("Dataset-level Outcome Extractor — Gemini 2.5-flash")
st.write("Upload a CSV with a column named `abstract` (case-insensitive). The app will analyze all abstracts together and return dataset-level outcomes.")

uploaded_file = st.file_uploader("Upload CSV file with abstracts", type=["csv"]) 
if not uploaded_file:
    st.info("Upload a CSV to begin. Example header: 'abstract'")
    st.stop()

# read CSV
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read uploaded CSV: {e}")
    st.stop()

# find abstract column
abstract_col = None
for c in df.columns:
    if c.strip().lower() == "abstract":
        abstract_col = c
        break

if abstract_col is None:
    st.error("CSV must contain an 'abstract' column (case-insensitive).")
    st.stop()

st.success(f"Found abstract column '{abstract_col}' — {len(df)} rows.")

# prepare combined text (numbered list)
all_abstracts = []
for i, a in enumerate(df[abstract_col].astype(str), start=1):
    # keep each abstract short in the prompt by trimming very long whitespace
    text = " ".join(a.split())
    all_abstracts.append(f"{i}. {text}")
combined = "

".join(all_abstracts)

st.markdown("### Preview (first 5 abstracts)")
for line in all_abstracts[:5]:
    st.write(line)

# get API key from Streamlit secrets (preferred) or env
api_key = None
try:
    api_key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
except Exception:
    api_key = None
if not api_key:
    api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("No Gemini API key found. Please set GEMINI_API_KEY in Streamlit secrets (.streamlit/secrets.toml) or as an environment variable.")
    st.stop()

if genai is None:
    st.error("Missing dependency: google-genai. Install with `pip install google-genai`.")
    st.stop()

# run single request for the whole dataset
if st.button("Analyze dataset"):
    client = genai.Client(api_key=api_key)
    prompt = DATASET_PROMPT_TEMPLATE.replace("{all_abstracts}", combined)

    with st.spinner("Sending dataset to Gemini and waiting for response — this may take a moment..."):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        except Exception as e:
            st.error(f"API error when calling Gemini: {e}")
            st.stop()

    # extract response text
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

    # basic validation of expected keys
    expected_keys = ["average_weight_loss_pct", "average_a1c_reduction_pct", "mash_resolution_counts", "alt_resolution_counts", "notes"]
    missing = [k for k in expected_keys if k not in parsed]
    if missing:
        st.warning(f"Model returned JSON but missing expected keys: {missing} — displaying what was returned.")

    st.subheader("Parsed JSON (dataset-level outcomes)")
    st.json(parsed)

    # allow download
    out_bytes = json.dumps(parsed, indent=2).encode("utf-8")
    st.download_button(label="Download outcomes JSON", data=out_bytes, file_name=OUTPUT_JSON, mime="application/json")

    st.success("Analysis complete.")

# end of file
