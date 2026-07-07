"""
Cache frozen PubMedBERT embeddings for MFC description columns.

This script only performs deterministic text embedding with a pretrained model.
It does not split data, fit dimensionality reducers, impute numeric values, or
use the regression target. The resulting 768D arrays can be safely cached before
train/test splitting, as long as the embedding model remains frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
DESCRIPTION_COLUMNS = [
    "anode_description",
    "cathode_description",
    "substrate_description",
]
ELECTRODE_ASPECT_LABELS = [
    "Physical structure",
    "Surface properties",
    "Bio-interaction",
    "Electrochemical properties",
]


def default_base_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def format_for_embedding(text: str, column: str, sep_token: str = "[SEP]") -> str:
    """Make structured description text explicit before embedding."""
    text = normalize_spaces(text)

    if column in {"anode_description", "cathode_description"}:
        parts = [normalize_spaces(part) for part in text.split("|")]
        if len(parts) == len(ELECTRODE_ASPECT_LABELS):
            labelled = [
                f"{label}: {part}"
                for label, part in zip(ELECTRODE_ASPECT_LABELS, parts, strict=True)
            ]
            return f" {sep_token} ".join(labelled)

    if column == "substrate_description":
        return f"Substrate characteristics: {text}"

    return text


def dataframe_text_hash(df: pd.DataFrame, model_name: str, pooling: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(pooling.encode("utf-8"))
    for column in DESCRIPTION_COLUMNS:
        hasher.update(column.encode("utf-8"))
        for value in df[column].fillna("").astype(str):
            hasher.update(format_for_embedding(value, column).encode("utf-8"))
            hasher.update(b"\n")
    return hasher.hexdigest()


def load_transformers():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "This script requires torch and transformers. Install them before "
            "running embedding extraction."
        ) from exc
    return torch, AutoModel, AutoTokenizer


def load_model(model_name: str):
    torch, AutoModel, AutoTokenizer = load_transformers()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return torch, tokenizer, model, device


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return masked.sum(dim=1) / denom


def embed_unique_texts(
    texts: list[str],
    *,
    tokenizer,
    model,
    torch,
    device,
    pooling: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    all_vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        if pooling == "cls":
            vectors = outputs.last_hidden_state[:, 0, :]
        elif pooling == "mean":
            vectors = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")

        all_vectors.append(vectors.cpu().numpy())

    return np.vstack(all_vectors)


def embed_column(
    series: pd.Series,
    column: str,
    *,
    tokenizer,
    model,
    torch,
    device,
    pooling: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    formatted = [
        format_for_embedding(value, column, sep_token=tokenizer.sep_token or "[SEP]")
        for value in series.fillna("").astype(str)
    ]
    unique_texts = list(dict.fromkeys(formatted))
    unique_vectors = embed_unique_texts(
        unique_texts,
        tokenizer=tokenizer,
        model=model,
        torch=torch,
        device=device,
        pooling=pooling,
        batch_size=batch_size,
        max_length=max_length,
    )
    vector_by_text = dict(zip(unique_texts, unique_vectors, strict=True))
    return np.vstack([vector_by_text[text] for text in formatted])


def parse_args() -> argparse.Namespace:
    base_dir = default_base_dir()
    parser = argparse.ArgumentParser(
        description="Extract and cache frozen PubMedBERT description embeddings."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=base_dir / "MFC_dataset_checked_description_cleaned.csv",
        help="Cleaned dataset containing description columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "embedding_cache",
        help="Directory for cached embedding arrays and metadata.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name.")
    parser.add_argument(
        "--pooling",
        choices=["cls", "mean"],
        default="cls",
        help="Pooling strategy for token embeddings. cls matches the previous pipeline.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--force", action="store_true", help="Overwrite a matching cache.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")

    missing_columns = [col for col in DESCRIPTION_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing description columns: {missing_columns}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "pubmedbert_description_embeddings.npz"
    metadata_path = args.output_dir / "pubmedbert_description_embeddings.metadata.json"

    input_hash = dataframe_text_hash(df, args.model, args.pooling)
    if cache_path.exists() and metadata_path.exists() and not args.force:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        if metadata.get("input_sha256") == input_hash:
            print(f"Cache already exists and matches input: {cache_path}")
            return

    torch, tokenizer, model, device = load_model(args.model)
    print(f"Loaded model: {args.model}")
    print(f"Device: {device}")

    arrays = {"row_index": np.arange(len(df), dtype=np.int64)}
    for column in DESCRIPTION_COLUMNS:
        print(f"Embedding {column} ...")
        vectors = embed_column(
            df[column],
            column,
            tokenizer=tokenizer,
            model=model,
            torch=torch,
            device=device,
            pooling=args.pooling,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        arrays[column] = vectors.astype(np.float32)
        print(f"  {column}: {vectors.shape}")

    np.savez_compressed(cache_path, **arrays)

    metadata = {
        "input_csv": str(args.input_csv),
        "input_rows": int(len(df)),
        "input_sha256": input_hash,
        "model": args.model,
        "pooling": args.pooling,
        "description_columns": DESCRIPTION_COLUMNS,
        "embedding_dim": int(arrays[DESCRIPTION_COLUMNS[0]].shape[1]),
        "format": "npz arrays keyed by description column, row-aligned to input CSV",
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved embeddings: {cache_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
