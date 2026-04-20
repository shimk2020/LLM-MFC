"""
MFC Dataset Description Enricher
=================================
Reads MFC_dataset_imputed.csv, generates LLM descriptions for the three
textual columns (substrate_type, anode_material, cathode_material), and
replaces the original values in-place, saving the result as
MFC_dataset_with_descriptions.csv.

Prompt format
-------------
* substrate_type :
      Single combined prompt → 3-4 semicolon-separated phrases covering
      (solubility, COD, toxicity, biodegradability, fermentability).

* anode_material :
      4 properties queried per material and joined with " , " (comma):
        Physical structure | Surface Properties | Bio-interaction | Electrochemical Properties
      Each block contains 3-4 semicolon-separated phrases.

* cathode_material :
      3 properties queried per material and joined with " , " (comma):
        Physical structure | Surface Properties | Electrochemical Properties
      Each block contains 3-4 semicolon-separated phrases.

De-duplication: each unique value is sent to the API once; results are
broadcast back to all rows containing that value.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import pandas as pd
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV   = os.path.join(SCRIPT_DIR, "MFC_dataset_imputed.csv")
OUTPUT_CSV  = os.path.join(SCRIPT_DIR, "MFC_dataset_with_descriptions.csv")

# Gemini API key is expected alongside the *parent* generation script
API_KEY_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "gemini_api.txt")

MODEL        = "gemini-3.1-pro-preview"
MAX_WORKERS  = 10
MAX_RETRIES  = 5
RETRY_DELAY  = 10   # seconds (multiplied by attempt number on rate-limit)

# Properties used for electrode materials
ANODE_PROPERTIES   = ["Physical structure", "Surface Properties",
                       "Bio-interaction", "Electrochemical Properties"]
CATHODE_PROPERTIES = ["Physical structure", "Surface Properties",
                       "Electrochemical Properties"]

# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are a materials science expert specialising in bioelectrochemical "
    "systems (BES). Your task is to generate compact, ground-truth-style "
    "material descriptions."
)


def load_client() -> genai.Client:
    for kp in (API_KEY_PATH,
               os.path.join(SCRIPT_DIR, "gemini_api.txt")):
        if os.path.exists(kp):
            with open(kp) as f:
                api_key = f.read().strip()
            if api_key:
                print(f"  API key loaded from: {kp}")
                return genai.Client(api_key=api_key)
    raise FileNotFoundError(
        "Gemini API key not found. Place gemini_api.txt in the script "
        "directory or its parent."
    )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_substrate_prompt(name: str) -> str:
    return (
        f"Substrate: {name}\n"
        f"Target Property: substrate characteristics for microbial fuel cells\n\n"
        f"Task:\n"
        f"Write 3 to 4 exact and concise scientific short phrases describing "
        f"only the most critical characteristics of this MFC substrate "
        f"(solubility, COD, toxicity, biodegradability, fermentability).\n\n"
        f"Strict Constraints:\n"
        f"1. Format: single line, phrases separated ONLY by semicolons "
        f"(e.g., highly soluble; non-toxic; moderate COD).\n"
        f"2. Style: telegraphic noun phrases only — no verbs, no articles, "
        f"no filler words.\n"
        f"3. Length: maximum 5 words per phrase.\n"
        f"4. Content: focus on performance-relevant properties only.\n"
        f"5. Prohibited: no bullet points, no introduction, no explanations."
    )


def build_electrode_prompt(name: str, criteria: str) -> str:
    ec_extra = (
        " You MUST explicitly include 'conductivity' and 'resistance'."
        if criteria == "Electrochemical Properties" else ""
    )
    return (
        f"Material: {name}\n"
        f"Target Property: {criteria}\n\n"
        f"Task:\n"
        f"Write 3 to 4 exact and concise scientific short phrases describing "
        f"only the most critical {criteria} characteristics of this electrode "
        f"material in microbial fuel cells.{ec_extra}\n\n"
        f"Strict Constraints:\n"
        f"1. Format: single line, phrases separated ONLY by semicolons "
        f"(e.g., high electrical conductivity; large surface area; "
        f"mesoporous structure).\n"
        f"2. Style: telegraphic noun phrases only — no verbs, no articles, "
        f"no filler words.\n"
        f"3. Length: maximum 5 words per phrase.\n"
        f"4. Content: focus on performance-relevant properties only.\n"
        f"5. Prohibited: no bullet points, no introduction, no explanations."
    )


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

def call_api(client: genai.Client, user_text: str) -> str:
    config = types.GenerateContentConfig(system_instruction=SYSTEM_MSG)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=user_text,
                config=config,
            )
            return response.text.strip()
        except Exception as e:
            msg = str(e)
            is_rate = "429" in msg or "quota" in msg.lower() or "rate" in msg.lower()
            if attempt < MAX_RETRIES and is_rate:
                wait = RETRY_DELAY * attempt
                print(f"    Rate-limit, waiting {wait}s "
                      f"(attempt {attempt}/{MAX_RETRIES}) ...")
                time.sleep(wait)
            elif attempt < MAX_RETRIES:
                time.sleep(2)
            else:
                return f"ERROR: {e}"
    return "ERROR: max retries exceeded"


# ---------------------------------------------------------------------------
# Per-column workers
# ---------------------------------------------------------------------------

def generate_substrate_descriptions(
    client: genai.Client,
    unique_values: List[str],
) -> Dict[str, str]:
    """Returns {name -> description} for substrate_type."""
    tasks: List[Tuple[str, str]] = [
        (name, build_substrate_prompt(name)) for name in unique_values
    ]
    results: Dict[str, str] = {}
    n = len(tasks)
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(call_api, client, prompt): name
            for name, prompt in tasks
        }
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = f"ERROR: {e}"
            completed += 1
            if completed % 10 == 0 or completed == n:
                print(f"  [substrate] {completed}/{n} done")

    return results


def generate_electrode_descriptions(
    client: genai.Client,
    unique_values: List[str],
    properties: List[str],
    label: str,
) -> Dict[str, str]:
    """
    For each unique electrode name, fires one API call per property, then
    joins the per-property results with ' , ' into a single description.

    Returns {name -> combined_description}.
    """
    # Build all (name, property, prompt) triples
    triples: List[Tuple[str, str, str]] = []
    for name in unique_values:
        for prop in properties:
            triples.append((name, prop, build_electrode_prompt(name, prop)))

    # Store results indexed by (name, property)
    raw: Dict[Tuple[str, str], str] = {}
    n = len(triples)
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(call_api, client, prompt): (name, prop)
            for name, prop, prompt in triples
        }
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                raw[key] = future.result()
            except Exception as e:
                raw[key] = f"ERROR: {e}"
            completed += 1
            if completed % 10 == 0 or completed == n:
                print(f"  [{label}] {completed}/{n} API calls done")

    # Combine: for each name join property blocks in the defined order
    combined: Dict[str, str] = {}
    for name in unique_values:
        parts = [raw.get((name, prop), "ERROR: missing") for prop in properties]
        combined[name] = " , ".join(parts)

    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Load dataset -------------------------------------------------------
    df: pd.DataFrame | None = None
    used_enc = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1"):
        try:
            df = pd.read_csv(INPUT_CSV, encoding=enc)
            used_enc = enc
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    if df is None:
        raise ValueError(f"Could not read {INPUT_CSV} with any supported encoding.")
    print(f"Loaded dataset ({used_enc}): {df.shape[0]} rows × {df.shape[1]} cols")

    # --- Load Gemini client -------------------------------------------------
    client = load_client()
    print(f"Using model: {MODEL}\n")

    # -----------------------------------------------------------------------
    # 1. substrate_type
    # -----------------------------------------------------------------------
    print("=== [1/3] Generating substrate_type descriptions ===")
    substrate_unique = df["substrate_type"].dropna().unique().tolist()
    print(f"  Unique substrates: {len(substrate_unique)}")
    substrate_map = generate_substrate_descriptions(client, substrate_unique)
    df["substrate_type"] = df["substrate_type"].map(substrate_map).fillna(df["substrate_type"])
    print("  substrate_type column updated.\n")

    # -----------------------------------------------------------------------
    # 2. anode_material  (4 properties)
    # -----------------------------------------------------------------------
    print("=== [2/3] Generating anode_material descriptions ===")
    anode_unique = df["anode_material"].dropna().unique().tolist()
    print(f"  Unique anodes: {len(anode_unique)}")
    anode_map = generate_electrode_descriptions(
        client, anode_unique, ANODE_PROPERTIES, "anode"
    )
    df["anode_material"] = df["anode_material"].map(anode_map).fillna(df["anode_material"])
    print("  anode_material column updated.\n")

    # -----------------------------------------------------------------------
    # 3. cathode_material  (3 properties — no Bio-interaction)
    # -----------------------------------------------------------------------
    print("=== [3/3] Generating cathode_material descriptions ===")
    cathode_unique = df["cathode_material"].dropna().unique().tolist()
    print(f"  Unique cathodes: {len(cathode_unique)}")
    cathode_map = generate_electrode_descriptions(
        client, cathode_unique, CATHODE_PROPERTIES, "cathode"
    )
    df["cathode_material"] = df["cathode_material"].map(cathode_map).fillna(df["cathode_material"])
    print("  cathode_material column updated.\n")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved enriched dataset to:\n  {OUTPUT_CSV}")
    print("\nDone.")


if __name__ == "__main__":
    main()
