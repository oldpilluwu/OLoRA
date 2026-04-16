"""
assets.py -- One-time download of model weights and dataset subsets to local disk.

This is a setup/provisioning module. You run it once (via scripts/download_assets.py)
before any training. After that, all code loads from the local `artifacts/` folder
so that experiments don't require internet access or re-downloading.

Flow:
  1. download_model()   -> pulls distilgpt2 weights from HuggingFace Hub into artifacts/models/distilgpt2
  2. download_dataset() -> pulls a small slice of each dataset and saves it as a HuggingFace DatasetDict
  3. download_all()     -> convenience wrapper that does both for every registered dataset
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from datasets import DatasetDict, load_dataset
from huggingface_hub import snapshot_download

from olora.settings import ARTIFACTS_DIR, DATASETS_DIR, DATASET_SPECS, DEFAULT_MODEL_ID, MODEL_DIR


def ensure_directories() -> None:
    """Create the artifacts directory tree if it doesn't exist yet."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)


def download_model(model_id: str = DEFAULT_MODEL_ID, local_dir: Path = MODEL_DIR) -> Path:
    """
    Download model weights + tokenizer from HuggingFace Hub.

    Uses snapshot_download which pulls all files in the repo except formats
    we don't need (h5, msgpack, etc.). The result is a local directory that
    AutoModelForCausalLM.from_pretrained() can load directly.
    """
    ensure_directories()
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        # Skip non-PyTorch weight formats to save disk space.
        ignore_patterns=("*.h5", "*.msgpack", "*.ot", "*.tflite"),
    )
    return local_dir


def download_dataset(dataset_key: str, force: bool = False) -> Path:
    """
    Download a small slice of one dataset and save it locally.

    Args:
      dataset_key -- A key into DATASET_SPECS (e.g. "ag_news" or "emotion").
                     The spec tells us the HuggingFace dataset name, which splits
                     and how many rows to take (e.g. "train[:512]" = first 512 rows).
      force       -- If True, re-download even if the local copy already exists.

    The result is saved as a HuggingFace DatasetDict with "train" and "validation"
    splits, plus a metadata.json with provenance info.
    """
    ensure_directories()
    spec = DATASET_SPECS[dataset_key]
    target_dir = DATASETS_DIR / dataset_key
    metadata_path = target_dir / "metadata.json"

    # Skip if already downloaded (unless force=True).
    if target_dir.exists() and not force:
        return target_dir

    # Pull the specified slices from HuggingFace.
    train = load_dataset(spec.name, spec.config, split=spec.train_split)
    validation = load_dataset(spec.name, spec.config, split=spec.validation_split)
    dataset_dict = DatasetDict({"train": train, "validation": validation})

    # Clean up any previous partial download, then save.
    if target_dir.exists():
        shutil.rmtree(target_dir)

    dataset_dict.save_to_disk(target_dir)

    # Write a small metadata file so we can remember what we downloaded.
    metadata = {
        "source_name": spec.name,
        "config": spec.config,
        "train_split": spec.train_split,
        "validation_split": spec.validation_split,
        "rows": {
            "train": len(train),
            "validation": len(validation),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target_dir


def download_all(force: bool = False) -> dict[str, str]:
    """Download the model and all registered datasets. Returns a manifest dict."""
    model_dir = download_model()
    datasets = {key: str(download_dataset(key, force=force)) for key in DATASET_SPECS}
    return {
        "model_dir": str(model_dir),
        "datasets": datasets,
    }
