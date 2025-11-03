# app.py
"""
Batch per-drug extractor — Gemini 2.5-flash (hardcoded API key)

Replace HARD_CODED_GEMINI_API_KEY with your actual Gemini API key.

Behavior:
- Upload CSV/XLSX with columns: drug (or drug_name) and abstract.
- Groups rows by drug, concatenates abstracts for each drug, calls Gemini per drug.
- Extracts per-study arrays for weight, a1c, mash (with fibrosis flags), plus ALT highest.
- Computes individual scores and assigns endpoint points based on the highest individual score.
- Adds a Line-Of-Treatment (LOT) table with 4 endpoints (Guiding Principle, Clinical Stage,
  Commercial Stage, Key Differentiator).
"""
import os
import json
import re
from typing import Optional, Any, Dict, List

import streamlit as st
import pandas as pd

# Lazy import for better error if missing
try:
    from google import genai
except Exception:
    genai = None

MODEL_NAME = "gemini-2.5-flash"
OUTPUT_JSON = "batch_outcomes.json"
OUTPUT_CSV = "batch_outcomes.csv"

# ---------- HARD-CODED API KEY ----------
# Replace this string with your actual key:
HARD_CODED_GEMINI_API_KEY = "YOUR_HARDCODED_GEMINI_KEY_HERE"
# ----------------------------------------

# Prompt template per drug: return arrays of reported values (per study) + highest values
DATASET_PROMPT_TEMPLATE = (
    "You are an extractor for clinical research abstracts for a single drug. I will provide a numbered list "
    "of abstracts. Read all abstracts and return exactly one JSON object (single-line) with these keys:\n\n"
    # Weight
    "1) weight_loss_values -> array of numeric weight-loss percentages reported across the abstracts (e.g. [12.0, 9.5, 22]). If none reported, return an empty array.\n\n"
    "2) highest_weight_loss_pct -> a single numeric value equal to the HIGHEST reported weight-loss percentage across studies, or null if none.\n\n"
    # A1c
    "3) a1c_reduction_values -> array of numeric absolute A1c reductions (percentage points) reported across abstracts (e.g. [1.2, 0.9, 2.3]). If none reported, return an empty array.\n\n"
    "4) highest_a1c_reduction_pct -> a single numeric value equal to the HIGHEST reported A1c absolute reduction (percentage points), or null if none.\n\n"
    # MASH (per-study values + per-study fibrosis flag)
    "5) mash_values -> array of numeric percentages (per study) representing the percent of patients with MASH resolution reported in each abstract (e.g. [40, 25, 60]). If none reported, return an empty array.\n\n"
    "6) mash_worsening_flags -> array of strings (per study) aligned to mash_values. Each element should be one of 'yes', 'no', or 'unclear' indicating whether that study reported worsening of fibrosis. If a study did not mention fibrosis worsening, use 'unclear'. If no mash_values, return an empty array for this too.\n\n"
    "7) mash_highest_resolution_pct -> the HIGHEST percentage of patients reported to have MASH resolution in any study, or null if none.\n\n"
    # ALT
    "8) alt_highest_reduction_pct -> the HIGHEST percentage reduction in ALT from baseline reported in any abstract, or null if none.\n\n"
    "IMPORTANT: Return ONLY the single JSON object and NOTHING else (no commentary). Use numeric values (no percent signs) in arrays and single numbers. Ensure arrays are JSON arrays and aligned where indicated.\n\n"
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
        if s == "" or s.lower() in {"null", "none", "n/a", "na"}:
            return None
        s = s.replace("%", "").replace(",", "")
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
    if s in {"yes", "y", "true", "t"}:
        return "yes"
    if s in {"no", "n", "false", "f"}:
        return "no"
    if s in {"unclear", "unknown", "maybe", "ambiguous", "unsure", "conflicting", "mixed"}:
        return "unclear"
    if "worsen" in s or "progress" in s or "fibrosis increased" in s or "fibrosis progression" in s:
        return "yes"
    if "no worsen" in s or "no progression" in s or "no fibrosis worsening" in s:
        return "no"
    return "unclear"

# ---- Scoring (your bins) ----
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

def score_mash_individual(pct: Optional[float], fibrosis_flag: Optional[str]) -> int:
    pct = safe_to_float(pct)
    flag = normalize_yes_no_unclear(fibrosis_flag) if fibrosis_flag is not None else None
    if pct is None or (isinstance(pct, float) and pct <= 0.0):
        return 1
    if flag == "yes":
        return 3
    if flag == "no":
        if pct >= 50:
            return 5
        if pct >= 30:
            return 4
        return 2
    return 2

def score_alt(highest_pct: Optional[float]) -> int:
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

# ---------- Line-of-Treatment (LOT) scoring functions ----------
def score_guiding_principle(parsed: dict, abstracts: List[str]) -> int:
    parsed = parsed or {}
    abstracts = abstracts or []
    fulltext = " ".join(str(a) for a in (abstracts or [])).lower()

    gp_parsed = None
    if isinstance(parsed, dict):
        for k in ["guiding_principle", "positioning", "line_of_treatment", "lot_position", "guiding"]:
            v = parsed.get(k)
            if isinstance(v, str) and v.strip():
                gp_parsed = v.strip().lower()
                break
        if gp_parsed is None:
            for k in parsed.keys():
                if "guiding" in k.lower() and isinstance(parsed.get(k), (int, float, str)):
                    gp_parsed = str(parsed.get(k)).lower()
                    break

    if gp_parsed:
        if re.search(r'\b(undisputed|displacing|displace|standard of care|standard-of-care|first[- ]line standard|first line standard)\b', gp_parsed):
            return 5
        if re.search(r'\b(strong first[- ]line|first[- ]line alternative|1st[- ]line alternative|dominant second[- ]line|dominant 2l)\b', gp_parsed):
            return 4
        if re.search(r'\b(second[- ]line|2l|guideline recommended 1l|guideline recommended first[- ]line|sub[- ]population)\b', gp_parsed):
            return 3
        if re.search(r'\b(third[- ]line|3l|after failing|refractory)\b', gp_parsed):
            return 2
        if re.search(r'\b(salvage|niche|late[- ]line|palliative|limited mention)\b', gp_parsed):
            return 1
        if re.search(r'\b1l\b', gp_parsed):
            return 4
        if re.search(r'\b2l\b', gp_parsed):
            return 3
        if re.search(r'\b3l\b', gp_parsed):
            return 2

    if (re.search(r'\b(undisputed|displacing|displace|dominant standard of care|first[- ]line standard|standard of care|standard-of-care)\b', fulltext)
        and re.search(r'\b(guideline|guidelines|recommended|recommendation|soc|top[- ]tier)\b', fulltext)):
        return 5

    if re.search(r'\b(strong first[- ]line|first[- ]line alternative|1st[- ]line alternative|dominant second[- ]line|dominant 2l|capture (a )?(significant|large) share)\b', fulltext):
        return 4

    if re.search(r'\b(second[- ]line|2l|recommended (as )?first[- ]line for|guideline (recommended|recommends).*first[- ]line|for (patients|subgroup|sub-population))\b', fulltext):
        return 3

    if re.search(r'\b(third[- ]line|3l|after (failing|failure|other options)|refractory|used only after)\b', fulltext):
        return 2

    if re.search(r'\b(salvage|niche|late[- ]line|last[- ]line|palliative|limited mention|no mention in guideline|rarely used)\b', fulltext):
        return 1

    return 1

def score_clinical_stage(parsed: dict, abstracts: List[str]) -> int:
    parsed = parsed or {}
    abstracts = abstracts or []
    fulltext = " ".join(str(a) for a in (abstracts or [])).lower()

    cs_parsed = None
    if isinstance(parsed, dict):
        for k in ["clinical_stage", "stage", "development_stage", "clinical_stage_criteria", "stage_position"]:
            v = parsed.get(k)
            if isinstance(v, str) and v.strip():
                cs_parsed = v.strip().lower()
                break
        if cs_parsed is None:
            for k in parsed.keys():
                if "stage" in k.lower() and isinstance(parsed.get(k), (int, float, str)):
                    cs_parsed = str(parsed.get(k)).lower()
                    break

    if cs_parsed:
        if re.search(r'\b(transformative|transformative efficacy|clean safety|new standard of care|new soc|become the new soc|preferred option|expert consensus|draft guideline|preferred in guidelines)\b', cs_parsed):
            return 5
        if re.search(r'\b(competitive 1st|competitive 1l|competitive first[- ]line|go to 2nd|go to 2l|recommended option|included in guidelines|recommended in guidelines)\b', cs_parsed):
            return 4
        if re.search(r'\b(2l|second[- ]line|niche 1l|niche first[- ]line|biomarker|subgroup|guideline subsection)\b', cs_parsed):
            return 3
        if re.search(r'\b(3l|third[- ]line|treatment[- ]resistant|refractory|later line)\b', cs_parsed):
            return 2
        if re.search(r'\b(small population|highly refractory|exhausted all|very small|ultra[- ]rare)\b', cs_parsed):
            return 1

    if (re.search(r'\b(transformative|transformative efficacy|highly effective|dramatic improvement|remarkable efficacy)\b', fulltext)
        and re.search(r'\b(clean safety|well tolerated|favorable safety|low adverse|good safety profile)\b', fulltext)
        and re.search(r'\b(draft guideline|expert consensus|preferred option|recommended as preferred|new standard of care|new soc|standard of care)\b', fulltext)):
        return 5

    if re.search(r'\b(competitive first[- ]line|competitive 1l|competitive 1st|recommended option in guidelines|included in guidelines|recommended in guidelines|go to 2l|go to second[- ]line)\b', fulltext):
        return 4

    if re.search(r'\b(second[- ]line|2l|niche first[- ]line|niche 1l|biomarker positive|biomarker|subgroup|guideline subsection)\b', fulltext):
        return 3

    if re.search(r'\b(third[- ]line|3l|treatment[- ]resistant|refractory|after (failing|failure|other options)|heavily pretreated)\b', fulltext):
        return 2

    if re.search(r'\b(very small|highly refractory|exhausted all other|ultra[- ]rare|limited population|rare disease|salvage)\b', fulltext):
        return 1

    return 1

def score_commercial_stage(parsed: dict, abstracts: List[str]) -> int:
    parsed = parsed or {}
    abstracts = abstracts or []
    fulltext = " ".join(str(a) for a in (abstracts or [])).lower()

    cm_parsed = None
    if isinstance(parsed, dict):
        for k in ["commercial_stage", "commercial_status", "market", "market_stage", "market_share", "prescription", "most_prescribed"]:
            v = parsed.get(k)
            if isinstance(v, str) and v.strip():
                cm_parsed = v.strip().lower()
                break
        if cm_parsed is None:
            for k in parsed.keys():
                if any(tok in k.lower() for tok in ["commercial", "market", "prescribe", "market_share", "most_prescribed"]):
                    val = parsed.get(k)
                    if isinstance(val, (int, float, str)) and str(val).strip():
                        cm_parsed = str(val).lower()
                        break

    if cm_parsed:
        if re.search(r'\b(preferred (first[- ]line|1l|1st[- ]line)|preferred option in guidelines|preferred in guidelines|sole (first[- ]line|1l)|sole soc|sole standard of care|most prescribed (1l|first[- ]line|1st[- ]line))\b', cm_parsed):
            return 5
        if re.search(r'\b(most[- ]prescribed|most prescribed|market leader|market-leading|dominant in 1l)\b', cm_parsed):
            return 5
        if re.search(r'\b(widely adopted|commonly used (1l|first[- ]line)|peer to (the )?soc|peer to soc|peer to standard of care|recommended as an option in guidelines)\b', cm_parsed):
            return 4
        if re.search(r'\b(most[- ]prescribed .*2l|most prescribed .*2l|most prescribed in 2l|dominant 2l)\b', cm_parsed):
            return 4
        if re.search(r'\b(established 2l|established second[- ]line|market share in 2l|recommended 1l for|recommended for (patients|segment|biomarker))\b', cm_parsed):
            return 3
        if re.search(r'\b(primaryly 3l|primarily third[- ]line|used after failure of 1l and 2l|used after failure)\b', cm_parsed):
            return 2
        if re.search(r'\b(salvage|last[- ]resort|last resort|limited use|no formal place|lacks formal place)\b', cm_parsed):
            return 1

    if (re.search(r'\b(preferred (first[- ]line|1l)|preferred in guidelines|preferred option in guidelines|recommended as preferred)\b', fulltext)
        and re.search(r'\b(most prescribed|most[- ]prescribed|market leader|dominant in 1l|widely prescribed)\b', fulltext)):
        return 5

    if re.search(r'\b(widely adopted|commonly used (first[- ]line|1l)|peer to (the )?soc|peer to standard of care|recommended as an option in guidelines)\b', fulltext):
        return 4
    if re.search(r'\b(most (prescribed|used) .*2l|dominant in 2l|widely used in 2l)\b', fulltext):
        return 4

    if re.search(r'\b(established (2nd|second|2l) market share|established 2l|market share in 2l|recommended (as )?1l for|recommended for (biomarker|comorbidity|specific subgroup|segment))\b', fulltext):
        return 3

    if re.search(r'\b(third[- ]line|3l|used only after failure|after failure of 1l|after failure of 2l|after failing other options|used after 1l and 2l)\b', fulltext):
        return 2

    if re.search(r'\b(salvage|last[- ]resort|last resort|limited place|lacks formal place|no formal place|rarely used|limited use)\b', fulltext):
        return 1

    return 1

def score_key_differentiator(parsed: dict, abstracts: List[str]) -> int:
    parsed = parsed or {}
    abstracts = abstracts or []
    fulltext = " ".join(str(a) for a in (abstracts or [])).lower()

    kd_parsed = None
    if isinstance(parsed, dict):
        for k in ["key_differentiator", "differentiator", "unique_feature", "moa", "mechanism", "differentiation"]:
            v = parsed.get(k)
            if isinstance(v, str) and v.strip():
                kd_parsed = v.strip().lower()
                break
        if kd_parsed is None:
            for k in parsed.keys():
                if any(tok in k.lower() for tok in ["different", "differentiator", "moa", "mechanism", "unique"]):
                    val = parsed.get(k)
                    if isinstance(val, (int, float, str)) and str(val).strip():
                        kd_parsed = str(val).lower()
                        break

    if kd_parsed:
        if re.search(r'\b(preferred (standard of care|soc|first[- ]line|1l)|preferred soc|preferred standard of care|de facto standard|de[- ]facto standard of care|de facto soc)\b', kd_parsed):
            return 5
        if re.search(r'\b(preferred in guidelines|recommended as preferred|guideline preferred)\b', kd_parsed):
            return 5

        if re.search(r'\b(dominant (1l|1st|first[- ]line|2l|second[- ]line)|major player|market leader in (1l|2l)|dominant in (1l|2l))\b', kd_parsed):
            return 4

        if re.search(r'\b(established role|clear place in treatment|part of the treatment algorithm|used in the algorithm|not market leader)\b', kd_parsed):
            return 3

        if re.search(r'\b(later[- ]line|reserved for later|used in later line|used in later lines|3l|third[- ]line|later line use)\b', kd_parsed):
            return 2

        if re.search(r'\b(salvage|limited to end|physician discretion|off label|rarely used|no formal place|limited place)\b', kd_parsed):
            return 1

    if (re.search(r'\b(de[- ]facto standard of care|de facto standard of care|de[- ]facto soc|de facto soc|de[- ]facto standard)\b', fulltext)
        and re.search(r'\b(preferred in guidelines|preferred option|guideline preferred|recommended as preferred)\b', fulltext)):
        return 5
    if re.search(r'\b(preferred (standard of care|soc|first[- ]line|1l)|preferred soc|preferred standard of care)\b', fulltext):
        return 5

    if re.search(r'\b(dominant (1l|1st|first[- ]line|2l|second[- ]line)|major player|market leader in (1l|2l)|dominant in (1l|2l))\b', fulltext):
        return 4
    if re.search(r'\b(major player|widely prescribed in early|widely prescribed in 1l|widely prescribed in 2l|market leader)\b', fulltext):
        return 4

    if re.search(r'\b(established role|part of the treatment algorithm|used in the treatment algorithm|has a clear place in the algorithm|not market leader)\b', fulltext):
        return 3
    if re.search(r'\b(recommended for a subgroup|recommended in guideline subsection|used for biomarker positive|used in specific subgroup)\b', fulltext):
        return 3

    if re.search(r'\b(later[- ]line|reserved for later|used in later line|3l|third[- ]line|used after failure)\b', fulltext):
        return 2

    if re.search(r'\b(salvage|physician discretion|off[- ]label|rarely used|no formal place|limited place|limited use|end of treatment pathway)\b', fulltext):
        return 1

    return 1

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Batch per-drug extractor (all drugs)", layout="wide")
st.title("Batch per-drug dataset-level extractor — Gemini 2.5-flash")
st.markdown(
    "- Upload a CSV or Excel with one abstract per row and a drug identifier column.\n"
    "- The app will group by drug and compute the metrics + scores (including LOT) for all drugs."
)

uploaded_file = st.file_uploader("Upload CSV or Excel (xlsx/xls/csv)", type=["csv", "xlsx", "xls"])
if not uploaded_file:
    st.info("Upload a file to begin. Required columns: a drug column and 'abstract'.")
    st.stop()

# Read file
try:
    if str(uploaded_file.name).lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read uploaded file: {e}")
    st.stop()

# Find columns
drug_col = None
for c in df.columns:
    if c.strip().lower() in {"drug", "drug_name", "treatment", "drugname"}:
        drug_col = c
        break
if drug_col is None:
    st.error("Could not find a drug column. Include a column named 'drug' or 'drug_name' (case-insensitive).")
    st.stop()

abstract_col = None
for c in df.columns:
    if c.strip().lower() == "abstract":
        abstract_col = c
        break
if abstract_col is None:
    st.error("File must contain an 'abstract' column (case-insensitive).")
    st.stop()

st.success(f"Found drug column '{drug_col}' and abstract column '{abstract_col}' — {len(df)} rows.")

# Use hard-coded API key
api_key = HARD_CODED_GEMINI_API_KEY
if not api_key or api_key == "YOUR_HARDCODED_GEMINI_KEY_HERE":
    st.error("Hard-coded Gemini API key not set. Edit app.py and replace HARD_CODED_GEMINI_API_KEY with your key.")
    st.stop()

if genai is None:
    st.error("Missing dependency: google-genai. Install with:  pip install google-genai")
    st.stop()

# Run processing
if st.button("Run batch extraction for ALL drugs"):
    client = genai.Client(api_key=api_key)

    grouped = df.groupby(df[drug_col].astype(str))[abstract_col].apply(list).to_dict()
    unique_drugs = list(grouped.keys())
    n = len(unique_drugs)
    st.write(f"Processing {n} drugs…")
    progress = st.progress(0)

    rows: List[Dict[str, Any]] = []
    for i, drug in enumerate(unique_drugs, start=1):
        # ensure every abstract is a string (handles NaN and non-strings safely)
        abstracts = ["" if a is None or (isinstance(a, float) and pd.isna(a)) else str(a) for a in grouped[drug]]

        # build the combined prompt text (safe, uses cleaned strings)
        combined = "\n\n".join([f"{idx+1}. {' '.join(a.split())}" for idx, a in enumerate(abstracts)])
        prompt = DATASET_PROMPT_TEMPLATE.replace("{all_abstracts}", combined)

        # Call model
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        except Exception as e:
            st.warning(f"API error for drug '{drug}': {e}")
            # append row with empty/default fields (no raw_model_output)
            rows.append({
                "drug": drug,
                "weight_values": [],
                "weight_individual_scores": [],
                "weight_score_points": None,
                "a1c_values": [],
                "a1c_individual_scores": [],
                "a1c_score_points": None,
                "mash_values": [],
                "mash_worsening_flags": [],
                "mash_individual_scores": [],
                "mash_score_points": None,
                "alt_highest_reduction_pct_normalized": None,
                "alt_score_points": None,
                "total_points": None,
                # LOT fields
                "guiding_principle_points": None,
                "clinical_stage_points": None,
                "commercial_stage_points": None,
                "key_differentiator_points": None,
                "lot_total_points": None,
                "overall_total_points": None,
                "max_points": 40
            })
            progress.progress(int(i / n * 100))
            continue

        # Extract text
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

        # Small helper to find keys if names differ
        def get_first_key_like(d: dict, substrings: List[str]):
            for k in d.keys():
                kl = k.lower()
                if all(sub in kl for sub in substrings):
                    return d.get(k)
            return None

        # default containers
        weight_values: List[float] = []
        weight_scores: List[int] = []
        weight_points = None

        a1c_values: List[float] = []
        a1c_scores: List[int] = []
        a1c_points = None

        mash_values: List[float] = []
        mash_worsening_flags: List[str] = []
        mash_scores: List[int] = []
        mash_points = None

        alt_highest = None
        alt_points = None

        # Parse model output if dict
        if isinstance(parsed, dict):
            # Weight values: try direct key or fallbacks
            w_vals = parsed.get("weight_loss_values") or get_first_key_like(parsed, ["weight", "loss", "values"])
            if isinstance(w_vals, list):
                for v in w_vals:
                    fv = safe_to_float(v)
                    if fv is not None:
                        weight_values.append(fv)
            # highest weight if provided
            w_high = parsed.get("highest_weight_loss_pct") or get_first_key_like(parsed, ["highest", "weight", "loss", "pct"])
            w_high_norm = safe_to_float(w_high)
            if w_high_norm is not None and (w_high_norm not in weight_values):
                weight_values.append(w_high_norm)

            # A1c values
            a_vals = parsed.get("a1c_reduction_values") or get_first_key_like(parsed, ["a1c", "reduction", "values"])
            if isinstance(a_vals, list):
                for v in a_vals:
                    fv = safe_to_float(v)
                    if fv is not None:
                        a1c_values.append(fv)
            # highest a1c
            a_high = parsed.get("highest_a1c_reduction_pct") or get_first_key_like(parsed, ["highest", "a1c", "reduction", "pct"])
            a_high_norm = safe_to_float(a_high)
            if a_high_norm is not None and (a_high_norm not in a1c_values):
                a1c_values.append(a_high_norm)

            # MASH values and aligned worsening flags
            m_vals = parsed.get("mash_values") or get_first_key_like(parsed, ["mash", "values"])
            m_flags = parsed.get("mash_worsening_flags") or get_first_key_like(parsed, ["mash", "worsen", "flags", "fibrosis"])
            if isinstance(m_vals, list):
                for v in m_vals:
                    fv = safe_to_float(v)
                    if fv is not None:
                        mash_values.append(fv)
            # if flags provided, align or fill with 'unclear'
            if isinstance(m_flags, list):
                for f in m_flags:
                    mash_worsening_flags.append(normalize_yes_no_unclear(f) or "unclear")
            # if model provided highest but not array, include it
            m_high = parsed.get("mash_highest_resolution_pct") or get_first_key_like(parsed, ["mash", "highest", "resolution", "pct"])
            m_high_norm = safe_to_float(m_high)
            if m_high_norm is not None and (m_high_norm not in mash_values):
                mash_values.append(m_high_norm)
                # if flags shorter than values, append 'unclear' for the highest
                if len(mash_worsening_flags) < len(mash_values):
                    mash_worsening_flags.extend(["unclear"] * (len(mash_values) - len(mash_worsening_flags)))

            # ALT highest
            alt_highest = parsed.get("alt_highest_reduction_pct") or get_first_key_like(parsed, ["alt", "highest", "reduction", "pct"])

        # compute individual scores for weight and pick highest as endpoint points
        for v in weight_values:
            weight_scores.append(score_weight_loss(safe_to_float(v)))
        if len(weight_scores) > 0:
            weight_points = max(weight_scores)
        else:
            if 'w_high_norm' in locals() and w_high_norm is not None:
                weight_points = score_weight_loss(w_high_norm)
                if not weight_values:
                    weight_values = [w_high_norm]
                    weight_scores = [weight_points]
            else:
                weight_points = 0

        # compute individual scores for a1c and pick highest
        for v in a1c_values:
            a1c_scores.append(score_a1c_reduction(safe_to_float(v)))
        if len(a1c_scores) > 0:
            a1c_points = max(a1c_scores)
        else:
            if 'a_high_norm' in locals() and a_high_norm is not None:
                a1c_points = score_a1c_reduction(a_high_norm)
                if not a1c_values:
                    a1c_values = [a_high_norm]
                    a1c_scores = [a1c_points]
            else:
                a1c_points = 0

        # compute individual scores for mash using per-study fibrosis flags
        if mash_values and (len(mash_worsening_flags) < len(mash_values)):
            mash_worsening_flags.extend(["unclear"] * (len(mash_values) - len(mash_worsening_flags)))

        for idx, v in enumerate(mash_values):
            flag = mash_worsening_flags[idx] if idx < len(mash_worsening_flags) else "unclear"
            mash_scores.append(score_mash_individual(safe_to_float(v), flag))
        if len(mash_scores) > 0:
            mash_points = max(mash_scores)
        else:
            if 'm_high_norm' in locals() and m_high_norm is not None:
                mash_points = score_mash_individual(m_high_norm, None)
                if not mash_values:
                    mash_values = [m_high_norm]
                    mash_worsening_flags = ["unclear"]
                    mash_scores = [mash_points]
            else:
                mash_points = 0

        # ALT scoring remains based on highest value
        alt_highest_norm = safe_to_float(alt_highest)
        alt_points = score_alt(alt_highest_norm)

        # total points: sum of the four clinical endpoint points (each endpoint max 5) -> max 20
        try:
            total_points = int(weight_points) + int(a1c_points) + int(mash_points) + int(alt_points)
        except Exception:
            total_points = None

        # ---- LOT scoring (use parsed and abstracts) ----
        guiding_principle_points = score_guiding_principle(parsed if isinstance(parsed, dict) else {}, abstracts)
        clinical_stage_points = score_clinical_stage(parsed if isinstance(parsed, dict) else {}, abstracts)
        commercial_stage_points = score_commercial_stage(parsed if isinstance(parsed, dict) else {}, abstracts)
        key_differentiator_points = score_key_differentiator(parsed if isinstance(parsed, dict) else {}, abstracts)

        try:
            lot_total_points = int(guiding_principle_points) + int(clinical_stage_points) + \
                               int(commercial_stage_points) + int(key_differentiator_points)
        except Exception:
            lot_total_points = None

        # overall (clinical + lot) total and max
        overall_total = None
        try:
            overall_total = (None if total_points is None or lot_total_points is None
                             else int(total_points) + int(lot_total_points))
        except Exception:
            overall_total = None

        # append row (no raw_model_output)
        rows.append({
            "drug": drug,
            "weight_values": weight_values,
            "weight_individual_scores": weight_scores,
            "weight_score_points": weight_points,
            "a1c_values": a1c_values,
            "a1c_individual_scores": a1c_scores,
            "a1c_score_points": a1c_points,
            "mash_values": mash_values,
            "mash_worsening_flags": mash_worsening_flags,
            "mash_individual_scores": mash_scores,
            "mash_score_points": mash_points,
            "alt_highest_reduction_pct_normalized": alt_highest_norm,
            "alt_score_points": alt_points,
            "total_points": total_points,
            # LOT fields
            "guiding_principle_points": guiding_principle_points,
            "clinical_stage_points": clinical_stage_points,
            "commercial_stage_points": commercial_stage_points,
            "key_differentiator_points": key_differentiator_points,
            "lot_total_points": lot_total_points,
            "overall_total_points": overall_total,
            "max_points": 40  # clinical (20) + LOT (20)
        })

        progress.progress(int(i / n * 100))

    out_df = pd.DataFrame(rows)

    # stringify list-columns so CSV writes nicely
    def stringify_lists(x):
        if isinstance(x, list):
            return json.dumps(x)
        return x

    out_df_for_export = out_df.copy()
    list_cols = ["weight_values", "weight_individual_scores", "a1c_values", "a1c_individual_scores",
                 "mash_values", "mash_worsening_flags", "mash_individual_scores"]
    for col in list_cols:
        if col in out_df_for_export.columns:
            out_df_for_export[col] = out_df_for_export[col].apply(stringify_lists)

    st.subheader(f"Results (showing {len(out_df)} drugs; lists are JSON strings in CSV)")
    st.dataframe(out_df.head(200))

    st.download_button("Download CSV", data=out_df_for_export.to_csv(index=False).encode("utf-8"),
                       file_name=OUTPUT_CSV, mime="text/csv")
    st.download_button("Download JSON", data=out_df.to_json(orient="records", indent=2).encode("utf-8"),
                       file_name=OUTPUT_JSON, mime="application/json")

    st.success("Batch extraction complete.")
