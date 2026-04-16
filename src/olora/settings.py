"""
settings.py -- Central configuration for the OLORA project.

This file defines:
  - Directory paths (where models, datasets, and run outputs live on disk)
  - Training hyperparameter defaults
  - Dataset specifications (which HuggingFace datasets to use and how to read them)

Nothing in this file does any computation -- it's purely constants and config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
# PROJECT_ROOT is the top-level repo folder (two parents up from this file,
# i.e.  src/olora/settings.py  ->  src/olora  ->  src  ->  PROJECT_ROOT).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Downloaded model weights and tokenizer files go here.
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models" / "distilgpt2"

# Downloaded dataset subsets (saved as HuggingFace DatasetDicts) go here.
DATASETS_DIR = ARTIFACTS_DIR / "datasets"

# JSON result files from baseline runs are written here.
RUNS_DIR = PROJECT_ROOT / "runs"

# ---------------------------------------------------------------------------
# Training hyperparameter defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "distilgpt2"       # HuggingFace model ID to download
DEFAULT_MAX_LENGTH = 128              # Max token sequence length per sample
DEFAULT_BATCH_SIZE = 2                # Samples per batch per adapter job
DEFAULT_TRAIN_STEPS = 16              # Number of measured training steps per job
DEFAULT_LR = 2e-4                     # AdamW learning rate for LoRA parameters
DEFAULT_SEED = 42                     # Random seed for reproducibility
DEFAULT_WARMUP_STEPS = 2              # Steps run before timing/recording starts


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """
    Describes one classification dataset and how to load/format it.

    Fields:
      name              -- HuggingFace dataset identifier (e.g. "ag_news" or "dair-ai/emotion")
      config            -- HuggingFace dataset config name, or None if the dataset has no configs
      train_split       -- Which split slice to use for training (e.g. "train[:512]" = first 512 rows)
      validation_split  -- Which split slice to use for evaluation
      text_field        -- Column name in the dataset that contains the input text
      label_field       -- Column name that contains the integer label
      instruction       -- A short instruction string prepended to each sample for prompting
      label_names       -- Human-readable label strings, indexed by the integer label
                           e.g. label_names[0] = "World" means label 0 maps to "World"
    """
    name: str
    config: str | None
    train_split: str
    validation_split: str
    text_field: str
    label_field: str
    instruction: str
    label_names: tuple[str, ...]


# ---------------------------------------------------------------------------
# Registry of available datasets
# ---------------------------------------------------------------------------
# Each key (e.g. "ag_news") is a "dataset_key" used throughout the codebase
# to refer to a particular dataset + its formatting rules.
DATASET_SPECS: dict[str, DatasetSpec] = {
    # AG News: 4-class news topic classification (World, Sports, Business, Sci/Tech)
    # We only use 512 training rows and 128 test rows to keep things small.
    "ag_news": DatasetSpec(
        name="ag_news",
        config=None,
        train_split="train[:512]",
        validation_split="test[:128]",
        text_field="text",
        label_field="label",
        instruction="Classify the news topic.",
        label_names=("World", "Sports", "Business", "Sci/Tech"),
    ),
    # Emotion: 6-class tweet emotion classification (sadness, joy, love, anger, fear, surprise)
    # Again using small slices for quick iteration.
    "emotion": DatasetSpec(
        name="dair-ai/emotion",
        config="split",
        train_split="train[:512]",
        validation_split="validation[:128]",
        text_field="text",
        label_field="label",
        instruction="Classify the emotion.",
        label_names=("sadness", "joy", "love", "anger", "fear", "surprise"),
    ),
}

