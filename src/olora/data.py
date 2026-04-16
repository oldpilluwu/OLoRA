"""
data.py -- Dataset loading, prompt formatting, tokenization, and batching.

This module turns raw HuggingFace dataset rows into tokenized batches ready
for causal language model training. The key idea:

  Raw row  -->  format_example()  -->  "Instruction: ...\nText: ...\nLabel: World<eos>"
                                          (prompt part)             (answer part)

  Then PromptDataset.__getitem__() tokenizes this and builds:
    - input_ids:      [prompt_tokens... answer_tokens...]
    - attention_mask:  [1, 1, 1, ..., 1]
    - labels:          [-100, -100, ..., answer_token_1, answer_token_2, ...]
                        ^^^^^^^^^^^^^^^^^
                        prompt tokens are masked with -100 so the loss
                        is only computed on the answer tokens.

  BatchCollator pads variable-length samples to the same length within a batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import DatasetDict, load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from olora.settings import DATASETS_DIR, DATASET_SPECS, DEFAULT_BATCH_SIZE, DEFAULT_MAX_LENGTH


def format_example(dataset_key: str, sample: dict[str, Any]) -> tuple[str, str]:
    """
    Convert a raw dataset row into a (prompt, answer) string pair.

    Example output for ag_news:
      prompt = "Instruction: Classify the news topic.\nText: Oil prices rise...\nLabel:"
      answer = "Business"
    """
    spec = DATASET_SPECS[dataset_key]
    text = str(sample[spec.text_field]).strip().replace("\r\n", "\n")
    # Map the integer label (e.g. 2) to the human-readable name (e.g. "Business").
    label_id = int(sample[spec.label_field])
    answer = spec.label_names[label_id]
    prompt = f"Instruction: {spec.instruction}\nText: {text}\nLabel:"
    return prompt, answer


class PromptDataset(Dataset):
    """
    A PyTorch Dataset that wraps raw dataset rows and returns tokenized
    training examples on the fly.

    Each __getitem__ call:
      1. Formats the row into a prompt + answer string
      2. Tokenizes both parts separately
      3. Concatenates them into one sequence [prompt_tokens, answer_tokens]
      4. Masks prompt tokens in labels with -100 (so the model only learns
         to predict the answer, not the prompt)
      5. Truncates to max_length if needed
    """
    def __init__(
        self,
        samples: list[dict[str, Any]],
        dataset_key: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.samples = samples            # Raw dataset rows (list of dicts)
        self.dataset_key = dataset_key    # Key into DATASET_SPECS (e.g. "ag_news")
        self.tokenizer = tokenizer        # The model's tokenizer
        self.max_length = max_length      # Hard cap on total token count per sample

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        prompt, answer = format_example(self.dataset_key, self.samples[index])

        # Tokenize the answer part first (e.g. " Business<eos>").
        # We tokenize it first so we know how many tokens it takes,
        # then give the remaining budget to the prompt.
        answer_ids = self.tokenizer(f" {answer}{self.tokenizer.eos_token}", add_special_tokens=False)["input_ids"]

        # The prompt gets whatever token budget remains after the answer.
        prompt_budget = max(1, self.max_length - len(answer_ids))
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"][:prompt_budget]

        # Concatenate prompt + answer into one sequence.
        input_ids = prompt_ids + answer_ids
        attention_mask = [1] * len(input_ids)  # All real tokens (no padding yet)

        # Labels = same as input_ids, but with prompt positions set to -100.
        # -100 tells PyTorch's cross_entropy to ignore those positions in the loss.
        # This way the model only gets gradients for predicting the answer tokens.
        labels = input_ids.copy()
        prompt_token_count = len(prompt_ids)
        labels[:prompt_token_count] = [-100] * prompt_token_count

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass(slots=True)
class BatchCollator:
    """
    Pads a list of variable-length samples to the same length so they can
    be stacked into a batch tensor.

    Padding strategy:
      - input_ids:      padded with the tokenizer's pad_token_id
      - attention_mask:  padded with 0 (so the model ignores padding positions)
      - labels:          padded with -100 (so padding doesn't contribute to the loss)
    """
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        # Find the longest sample in this batch.
        max_tokens = max(item["input_ids"].shape[0] for item in batch)
        input_ids = []
        attention_mask = []
        labels = []

        for item in batch:
            # How many padding tokens this sample needs to reach max_tokens.
            pad_len = max_tokens - item["input_ids"].shape[0]
            input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=pad_id))
            attention_mask.append(torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0))
            labels.append(torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100))

        # Stack into [batch_size, max_tokens] tensors.
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


def load_local_dataset_dict(dataset_key: str) -> DatasetDict:
    """Load a previously-downloaded dataset from the local artifacts directory."""
    dataset_dir = DATASETS_DIR / dataset_key
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Local dataset '{dataset_key}' was not found at {dataset_dir}. Run scripts/download_assets.py first."
        )
    return load_from_disk(str(dataset_dir))


def build_dataloader(
    dataset_key: str,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> DataLoader:
    """
    Build a ready-to-iterate DataLoader for one dataset split.

    Args:
      dataset_key  -- Which dataset (e.g. "ag_news")
      split        -- "train" or "validation"
      tokenizer    -- The model's tokenizer (needed for tokenization + padding)
      batch_size   -- Samples per batch
      max_length   -- Max tokens per sample

    Returns a DataLoader that yields dicts with keys:
      input_ids [batch_size, seq_len], attention_mask [batch_size, seq_len], labels [batch_size, seq_len]
    """
    dataset_dict = load_local_dataset_dict(dataset_key)
    hf_dataset = dataset_dict[split]
    prompt_dataset = PromptDataset(list(hf_dataset), dataset_key=dataset_key, tokenizer=tokenizer, max_length=max_length)
    return DataLoader(
        prompt_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),        # Shuffle training data, keep validation deterministic
        collate_fn=BatchCollator(tokenizer),  # Custom padding logic
    )
