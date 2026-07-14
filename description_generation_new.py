"""
MFC New Dataset – Feature Description Generator
Uses OpenAI Batch API to generate concise scientific descriptions for
anode, cathode, and substrate features from the imputed MFC dataset.

Workflow:
  1. Extract unique materials/substrates from MFC_dataset_new_imputed.csv.
  2. For electrodes: generate descriptions for 4 criteria per unique material without duplication.
  3. For substrates: generate 1 description per unique substrate.
  4. Submit all requests as a single OpenAI batch.
  5. Map generated descriptions back to the dataset rows.
  6. Save output CSV with new description columns.
"""

import json
import os
import time

import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = "gpt-5.5-2026-04-23"
POLL_INTERVAL = 30  # seconds between batch status checks
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}

INPUT_CSV = os.path.join(SCRIPT_DIR, "MFC_dataset_checked_cleaned.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "MFC_dataset_checked_description_cleaned.csv")

# The 4 criteria for electrode descriptions (same as original dataset)
ELECTRODE_CRITERIA = [
    "Physical structure",
    "Surface Properties",
    "Bio-interaction",
    "Electrochemical Properties",
]

# Column mapping
TEXT_COLUMNS = {
    "anode": {
        "col": "anode_material",
        "has_criteria": True,
    },
    "cathode": {
        "col": "cathode_material",
        "has_criteria": True,
    },
    "substrate": {
        "col": "substrate_type",
        "has_criteria": False,
    },
}


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------
def load_client() -> OpenAI:
    key_path = os.path.join(SCRIPT_DIR, "openai_api.txt")
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"API key file not found: {key_path}")
    with open(key_path) as f:
        api_key = f.read().strip()
    if not api_key:
        raise ValueError("API key file is empty.")
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Prompt builder  (UNCHANGED from reference – do not modify prompts)
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are a materials science expert specializing in bioelectrochemical systems (BES). "
    "Your task is to generate compact, ground-truth-style material descriptions."
)


def build_messages(name: str, has_criteria: bool, criteria: str = "") -> list[dict]:
    if has_criteria:
        ec_extra = (
            " You MUST explicitly include 'conductivity' and 'resistance'."
            if criteria == "Electrochemical Properties" else ""
        )
        user_content = (
            f"Material: {name}\n"
            f"Target Property: {criteria}\n\n"
            f"Task:\n"
            f"Write 3 to 4 exact and concise scientific short phrases describing "
            f"only the most critical {criteria} characteristics of this electrode "
            f"material in microbial fuel cells.{ec_extra}\n\n"
            f"Strict Constraints:\n"
            f"1. Format: single line, phrases separated ONLY by semicolons "
            f"(e.g., high electrical conductivity; large surface area; mesoporous structure).\n"
            f"2. Style: telegraphic noun phrases only — no verbs, no articles, no filler words.\n"
            f"3. Length: maximum 5 words per phrase.\n"
            f"4. Content: focus on performance-relevant properties only.\n"
            f"5. Prohibited: no bullet points, no introduction, no explanations."
        )
    else:
        user_content = (
            f"Substrate: {name}\n"
            f"Target Property: substrate characteristics for microbial fuel cells\n\n"
            f"Task:\n"
            f"Write 3 to 4 exact and concise scientific short phrases describing "
            f"only the most critical characteristics of this MFC substrate "
            f"(solubility, COD, toxicity, biodegradability, fermentability).\n\n"
            f"Strict Constraints:\n"
            f"1. Format: single line, phrases separated ONLY by semicolons "
            f"(e.g., highly soluble; non-toxic; moderate COD).\n"
            f"2. Style: telegraphic noun phrases only — no verbs, no articles, no filler words.\n"
            f"3. Length: maximum 5 words per phrase.\n"
            f"4. Content: focus on performance-relevant properties only.\n"
            f"5. Prohibited: no bullet points, no introduction, no explanations."
        )

    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Build batch requests for all unique text values
# ---------------------------------------------------------------------------
def build_all_requests(df: pd.DataFrame) -> list[dict]:
    """
    Build batch API requests for every unique (column, value, criteria) combo.
    custom_id format:
      - electrodes: "{col_type}_{material_name}_{criteria}"
      - substrates: "substrate_{substrate_name}"
    """
    requests = []
    seen = set()

    for col_type, cfg in TEXT_COLUMNS.items():
        col_name = cfg["col"]
        has_criteria = cfg["has_criteria"]
        unique_vals = df[col_name].dropna().unique()

        for name in unique_vals:
            name_str = str(name).strip()
            if not name_str:
                continue

            if has_criteria:
                for crit in ELECTRODE_CRITERIA:
                    cid = f"{col_type}||{name_str}||{crit}"
                    if cid in seen:
                        continue
                    seen.add(cid)

                    messages = build_messages(name_str, True, crit)
                    requests.append({
                        "custom_id": cid,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": MODEL,
                            "messages": messages,
                        },
                    })
            else:
                cid = f"{col_type}||{name_str}"
                if cid in seen:
                    continue
                seen.add(cid)

                messages = build_messages(name_str, False)
                requests.append({
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL,
                        "messages": messages,
                    },
                })

    return requests


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------
def submit_batch(client: OpenAI, requests: list[dict]) -> tuple[str, str, str]:
    """Write JSONL, upload it, create the batch. Returns (batch_id, file_id, local_path)."""
    local_path = os.path.join(SCRIPT_DIR, "new_batch_input.jsonl")
    with open(local_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    with open(local_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch.id, uploaded.id, local_path


def poll_batch(client: OpenAI, batch_id: str):
    """Poll until the batch reaches a terminal status."""
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"  status={batch.status} "
            f"completed={counts.completed}/{counts.total} "
            f"failed={counts.failed}"
        )
        if batch.status in TERMINAL_STATUSES:
            return batch
        print(f"  Waiting {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)


def download_results(client: OpenAI, output_file_id: str) -> dict[str, str]:
    """Parse JSONL output and return {custom_id: generated_text}."""
    content = client.files.content(output_file_id).text
    results = {}
    ok_count = 0
    err_count = 0
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        custom_id = record["custom_id"]
        resp = record.get("response") or {}
        body = resp.get("body") or {}
        if isinstance(body, str):
            body = json.loads(body)
        if resp.get("status_code") == 200:
            text = body["choices"][0]["message"]["content"].strip()
            ok_count += 1
        else:
            # Error detail may be in response.body.error or top-level error field
            body_error = body.get("error") or {}
            top_error = record.get("error") or {}
            code = body_error.get("code") or top_error.get("code") or "unknown"
            msg = body_error.get("message") or top_error.get("message") or ""
            text = f"ERROR: {code} – {msg}"
            err_count += 1
        results[custom_id] = text
    print(f"  Parsed {ok_count} successful / {err_count} failed responses.")
    if ok_count > 0:
        sample_key = next(iter(results))
        print(f"  Sample [{sample_key}]: {results[sample_key][:120]}")
    return results


def cleanup(client: OpenAI, file_id: str, local_path: str) -> None:
    try:
        os.remove(local_path)
    except OSError:
        pass
    try:
        client.files.delete(file_id)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Map descriptions back to dataset rows
# ---------------------------------------------------------------------------
def map_descriptions(df: pd.DataFrame, results: dict[str, str]) -> pd.DataFrame:
    """
    For each row, look up generated descriptions and create new columns:
      - anode_description: all 4 criteria descriptions joined with " | "
      - cathode_description: all 4 criteria descriptions joined with " | "
      - substrate_description: single description
    """
    anode_descs = []
    cathode_descs = []
    substrate_descs = []

    for _, row in df.iterrows():
        # Anode
        anode_name = str(row["anode_material"]).strip()
        anode_parts = []
        for crit in ELECTRODE_CRITERIA:
            cid = f"anode||{anode_name}||{crit}"
            desc = results.get(cid, "MISSING")
            anode_parts.append(desc)
        anode_descs.append(" | ".join(anode_parts))

        # Cathode
        cathode_name = str(row["cathode_material"]).strip()
        cathode_parts = []
        for crit in ELECTRODE_CRITERIA:
            cid = f"cathode||{cathode_name}||{crit}"
            desc = results.get(cid, "MISSING")
            cathode_parts.append(desc)
        cathode_descs.append(" | ".join(cathode_parts))

        # Substrate
        substrate_name = str(row["substrate_type"]).strip()
        cid = f"substrate||{substrate_name}"
        substrate_descs.append(results.get(cid, "MISSING"))

    df["anode_description"] = anode_descs
    df["cathode_description"] = cathode_descs
    df["substrate_description"] = substrate_descs

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Load data
    print(f"Loading imputed dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  {len(df)} rows x {len(df.columns)} cols\n")

    # Count unique values
    for col_type, cfg in TEXT_COLUMNS.items():
        n_unique = df[cfg["col"]].nunique()
        print(f"  Unique {col_type} ({cfg['col']}): {n_unique}")

    # Build requests
    requests = build_all_requests(df)
    print(f"\nTotal batch requests: {len(requests)}")

    # Connect to OpenAI
    client = load_client()
    print(f"Using model: {MODEL}\n")

    # Submit batch
    batch_id, file_id, local_path = submit_batch(client, requests)
    print(f"Batch submitted: {batch_id}")
    print(f"  Requests: {len(requests)}")

    # Poll
    print("\nPolling batch status...")
    batch = poll_batch(client, batch_id)

    if batch.status != "completed":
        print(f"\nBatch did not complete (status={batch.status}). Exiting.")
        cleanup(client, file_id, local_path)
        return

    # Download results
    print("\nDownloading results...")
    results = download_results(client, batch.output_file_id)

    # Map back to dataset
    print("\nMapping descriptions to dataset rows...")
    df = map_descriptions(df, results)

    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[Saved] {OUTPUT_CSV}")
    print(f"  {len(df)} rows x {len(df.columns)} cols")
    print(f"  New columns: anode_description, cathode_description, substrate_description")

    # Also save the per-unique-value results for reference
    ref_path = os.path.join(SCRIPT_DIR, "unique_descriptions_reference.json")
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Reference descriptions saved to: {ref_path}")

    # Cleanup
    cleanup(client, file_id, local_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
