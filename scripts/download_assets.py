"""
download_assets.py -- One-time setup script to download the model and datasets.

Run this ONCE before any training:
  .\.venv\Scripts\python.exe scripts\download_assets.py

This downloads:
  - distilgpt2 model weights + tokenizer  ->  artifacts/models/distilgpt2/
  - ag_news dataset subset (512 train, 128 val)  ->  artifacts/datasets/ag_news/
  - emotion dataset subset (512 train, 128 val)  ->  artifacts/datasets/emotion/

Use --force to re-download datasets even if they already exist locally.
"""

from __future__ import annotations

import argparse
import json

from olora.assets import download_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the local model snapshot and dataset subsets.")
    parser.add_argument("--force", action="store_true", help="Re-download local dataset subsets.")
    args = parser.parse_args()

    # download_all() returns a manifest dict showing what was downloaded and where.
    manifest = download_all(force=args.force)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

