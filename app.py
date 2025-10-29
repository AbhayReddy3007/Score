# app.py
"""
Streamlit app to extract structured outcomes from research abstracts using Gemini 2.5-flash.

Usage:
    1) Put your Gemini key in Streamlit secrets:
       .streamlit/secrets.toml
         GEMINI_API_KEY = "your_real_gemini_api_key_here"

       Or set the environment variable GEMINI_API_KEY (less recommended for deployed apps),
       or paste the key in the sidebar (session-only).

    2) Install deps:
       pip install streamlit pandas google-genai

    3) Run:
       streamlit run app.py
"""
import os
import re
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
DEFAULT_SLEEP = 0.25
OUTPUT_FILENAME = "abstracts_with_outcomes.csv"

PROMPT_TEMPLATE = r"""
You are a precise extractor. Given the abstract of a clinical / research paper, extract the following four outcomes if they are reported in the abstract. Output exactly one valid JSON object (single-line) with these four keys:

1) weight_loss_pct -> numeric percent (e.g., 12.5) if the abstract reports a percentage of weight loss; otherwise null.
2) a1c_reduction_pct -> numeric percent (e.g., 1.2) if the abstract reports an A1c reduction percentage; otherwise null.
   - If A1c reduction is reported as absolute change in % (e.g., "A1c decreased from 8.5% to 7.3%"), compute the absolute percentage-point change (8.5 -> 7.3 -> 1.2) and return 1.2.
   - If A1c is reported as relative percent reduction only (e.g., "10% relative reduction"), return 10.
3) mash_resolution -> one of "yes", "no", or "unclear". "yes" if the abstract explicitly states MASH (or NASH/MASH) resolved or showed resolution; "no" if explicitly states no resolution; "unclear" if not reported or ambiguous.
4) alt_resolution -> one of "yes", "no", or "unclear". "yes" if the abstract explicitly states ALT normalized or resolved; "no" if explicitly states not resolved; "unclear" otherwise.

Example output:
{"weight_loss_pct": 12.5, "a1c_reduction_pct": 1.2, "mash_resolution": "yes", "alt_resolution": "unclear"}

Now extract from this abstract (do not add any extra text, commentary, or explanation — only the JSON object):

-----
{abstract_here}
-----
"""

# ---------------- HELPERS ----------------
def parse_json_from_response(resp_text: str) -> Optional[dict]:
    """Attempt to parse a JSON object embedded in the model response."""
    if not isinstance(resp_text, str):
        return None
    text = resp_text.strip()
    # strip code fences
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.split("\n")[1:-1]).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        try:
            return json.loads(text)
        except Exception:
            return None
    sub = text[start:end+1]
    try:
        return json.loads(sub)
    except Exception:
        # try to fix trailing commas
        try:
            cleaned = re.sub(r",\s*}\s*$", "}", sub)
            return json.loads(cleaned)
        except Exception:
            return None

def normalize_parsed(parsed: dict) -> dict:
    out = {
        "weight_loss_pct": None,
        "a1c_reduction_pct": None,
        "mash_resolution": "unclear",
        "alt_resolution": "unclear",
    }
    if not isinstance(parsed, dict):
        return out

    def to_float(v):
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            m = re.search(r"-?\d+\.?\d*", str(v))
            if m:
                try:
                    return float(m.group(0))
                except Exception:
                    return None
            return None

    out["weight_loss_pct"] = to_float(parsed.get("weight_loss_pct"))
    out["a1c_reduction_pct"] = to_float(parsed.get("a1c_reduction_pct"))

    for key in ("mash_resolution", "alt_resolution"):
        val = parsed.get(key)
        if isinstance(val, str) and val.lower() in ("yes", "no", "unclear"):
            out[key] = val.lower()
        else:
            out[key] = "unclear"
    return out

# ---------------- UI ----------------
st.set_page_config(page_title="Abstract Outcome Extractor — Gemini", layout="wide")
st.title("Abstract Outcome Extractor — Gemini 2.5-flash")
st.write("Upload a CSV with a column named `abstract`. The app will call Gemini and extract structured outcomes per abstract.")

with st.sidebar:
    st.header("Settings")
    st.markdown("**API key source (recommended: Streamlit secrets)**")
    st.write("Order checked: 1) Streamlit secrets, 2) environment variable GEMINI_API_KEY, 3) pasted key (fallback).")
    api_key_input = st.text_input("Paste Gemini API key (hidden)", type="password")
    sleep_sec = st.number_input("Seconds between requests (throttle)", min_value=0.0, max_value=5.0, value=float(DEFAULT_SLEEP), step=0.05)
    st.markdown("---")
    st.caption("Do not upload sensitive unpublished clinical data without permission.")

# uploader
uploaded_file = st.file_uploader("Upload CSV file with abstracts (column name 'abstract')", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV to begin. Example header: 'abstract'")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read uploaded CSV: {e}")
    st.stop()

# find abstract column case-insensitive
abstract_col = None
for c in df.columns:
    if c.strip().lower() == "abstract":
        abstract_col = c
        break

if abstract_col is None:
    st.error("CSV must contain a column named 'abstract' (case-insensitive). Rename and re-upload.")
    st.stop()

st.success(f"Found abstract column '{abstract_col}' — {len(df)} rows.")
st.dataframe(df[[abstract_col]].head(10))

start_button = st.button("Start extraction")
show_raw = st.checkbox("Show raw Gemini output (gemini_raw)", value=False)

if not start_button:
    st.stop()

if genai is None:
    st.error("Missing dependency: google-genai. Install with `pip install google-genai`.")
    st.stop()

# determine key: secrets -> env -> pasted
key_to_use = None
try:
    key_to_use = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
except Exception:
    key_to_use = None

if not key_to_use:
    key_to_use = os.environ.get("GEMINI_API_KEY")
if not key_to_use:
    key_to_use = api_key_input.strip() or None

if not key_to_use:
    st.error("No Gemini API key found. Add to .streamlit/secrets.toml or set GEMINI_API_KEY env var or paste in the sidebar.")
    st.stop()

# init client
try:
    client = genai.Client(api_key=key_to_use)
except Exception as e:
    st.error(f"Failed to initialize Gemini client: {e}")
    st.stop()

# prepare results
results = df.copy()
results["weight_loss_pct"] = None
results["a1c_reduction_pct"] = None
results["mash_resolution"] = None
results["alt_resolution"] = None
results["gemini_raw"] = None

total = len(results)
progress_bar = st.progress(0)
status = st.empty()

for idx, row in results.iterrows():
    status.text(f"Processing {idx+1}/{total}")
    abstract = str(row[abstract_col])
    prompt = PROMPT_TEMPLATE.replace("{abstract_here}", abstract)

    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)

        # extract text robustly
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

        results.at[idx, "gemini_raw"] = resp_text
        parsed = parse_json_from_response(resp_text)
        normalized = normalize_parsed(parsed)

        results.at[idx, "weight_loss_pct"] = normalized["weight_loss_pct"]
        results.at[idx, "a1c_reduction_pct"] = normalized["a1c_reduction_pct"]
        results.at[idx, "mash_resolution"] = normalized["mash_resolution"]
        results.at[idx, "alt_resolution"] = normalized["alt_resolution"]

    except Exception as e:
        results.at[idx, "gemini_raw"] = f"ERROR: {e}"
        results.at[idx, "mash_resolution"] = "unclear"
        results.at[idx, "alt_resolution"] = "unclear"

    progress_bar.progress((idx + 1) / total)
    time.sleep(float(sleep_sec))

status.text("Completed processing.")

# summary
st.markdown("### Summary")
wl_count = results["weight_loss_pct"].notnull().sum()
a1c_count = results["a1c_reduction_pct"].notnull().sum()
mash_yes = (results["mash_resolution"] == "yes").sum()
alt_yes = (results["alt_resolution"] == "yes").sum()

st.write(f"Processed {total} abstracts.")
st.write(f"Weight-loss % extracted for {wl_count} abstracts.")
st.write(f"A1c reduction % extracted for {a1c_count} abstracts.")
st.write(f"MASH resolution reported as 'yes' in {mash_yes} abstracts.")
st.write(f"ALT resolution reported as 'yes' in {alt_yes} abstracts.")

display_cols = [abstract_col, "weight_loss_pct", "a1c_reduction_pct", "mash_resolution", "alt_resolution"]
if show_raw:
    display_cols.append("gemini_raw")
st.dataframe(results[display_cols].head(1000))

# download
csv_bytes = results.to_csv(index=False).encode("utf-8")
st.download_button("Download annotated CSV", data=csv_bytes, file_name=OUTPUT_FILENAME, mime="text/csv")

st.success("Done — download the CSV or inspect results above.")
