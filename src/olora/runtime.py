"""
runtime.py -- Baseline training loops and the runner that orchestrates them.

This is the main "engine" of the project. It:
  1. Loads the base model and wraps it with routed LoRA adapters
  2. Creates one AdapterJob per dataset (each job has its own dataloader + optimizer)
  3. Runs one of three baseline training strategies
  4. Logs per-step metrics and writes results to a JSON file

The three baselines:
  - SEQUENTIAL:               Train job A to completion, then train job B.
  - TIME_SLICED:              Alternate steps between jobs (round-robin: A, B, A, B, ...).
  - FIXED_SET_SIMULTANEOUS:   Fuse samples from all jobs into ONE batch, do ONE forward
                              pass and ONE backward pass, then step each optimizer.

All three use the same underlying RoutedCausalLM model -- the difference is only
in how batches are fed and when optimizers step.
"""

from __future__ import annotations

import gc
import json
import logging
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from olora.data import build_dataloader
from olora.routed_lora import RoutedCausalLM, RoutedLoraConfig
from olora.settings import (
    DATASET_SPECS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_MAX_LENGTH,
    DEFAULT_SEED,
    DEFAULT_TRAIN_STEPS,
    DEFAULT_WARMUP_STEPS,
    MODEL_DIR,
    RUNS_DIR,
)


LOGGER = logging.getLogger("olora.runtime")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def _cuda_sync() -> None:
    """Block until all queued CUDA work has completed. No-op on CPU.

    Required before/after timing measurements: PyTorch CUDA kernels are
    asynchronous, so perf_counter() around forward/backward only measures
    kernel launch time unless we sync.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _release_torch_memory() -> None:
    """Best-effort release of Python refs and CUDA cached blocks."""
    gc.collect()
    if torch.cuda.is_available():
        _cuda_sync()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                # IPC collection can fail harmlessly if CUDA context state changed.
                pass


# ===========================================================================
# Utility functions
# ===========================================================================

def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds for Python, PyTorch CPU, and PyTorch CUDA for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    """Auto-detect: use GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_base_model(local_model_dir: Path = MODEL_DIR) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load the pre-trained model and tokenizer from the local artifacts directory.
    This expects you've already run scripts/download_assets.py.

    Returns (tokenizer, model) -- both loaded from the local snapshot, not from the internet.
    """
    if not local_model_dir.exists():
        raise FileNotFoundError(
            f"Local model directory was not found at {local_model_dir}. Run scripts/download_assets.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
    # GPT-2 doesn't have a pad token by default; reuse EOS as the pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(local_model_dir, local_files_only=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def prepare_multi_adapter_model(
    model: AutoModelForCausalLM,
    adapter_names: list[str],
    lora_config: RoutedLoraConfig | None = None,
) -> RoutedCausalLM:
    """
    Wrap the base HuggingFace model with routed LoRA adapters.

    This replaces the GPT-2 layers named "c_attn", "c_proj", "c_fc" in every
    transformer block with RoutedLoRAConv1D layers. Each replaced layer gets
    one (A, B) parameter pair per adapter.

    Args:
      model         -- The base HuggingFace CausalLM (will be modified in-place)
      adapter_names -- List of adapter names (e.g. ["ag_news", "emotion"])

    Returns a RoutedCausalLM that accepts adapter_ids in its forward() call.
    """
    return RoutedCausalLM(
        base_model=model,
        adapter_names=adapter_names,
        target_modules=["c_attn", "c_proj", "c_fc"],  # GPT-2 specific layer names
        config=lora_config or RoutedLoraConfig(rank=8, alpha=16.0, dropout=0.05),
    )


def count_labeled_tokens(batch: dict[str, torch.Tensor]) -> int:
    """
    Count how many tokens in the batch will actually contribute to the loss.
    These are the tokens where labels != -100 (i.e. the answer tokens, not prompt or padding).
    """
    return int(batch["labels"].ne(-100).sum().item())


def build_adapter_ids(batch: dict[str, torch.Tensor], adapter_index: int) -> torch.Tensor:
    """
    Create an adapter_ids tensor for a batch where ALL samples belong to one adapter.

    E.g. for a batch of 2 samples and adapter_index=1:
      returns tensor([1, 1])

    This tells the routed LoRA layers "every sample in this batch uses adapter 1".
    """
    return torch.full((batch["input_ids"].shape[0],), adapter_index, dtype=torch.long)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move all tensors in a batch dict to the specified device (CPU or GPU)."""
    return {key: value.to(device) for key, value in batch.items()}


def gpu_snapshot() -> dict[str, float] | None:
    """
    Query nvidia-smi for current GPU utilization and memory usage.

    Returns a dict like:
      {"gpu_utilization_percent": 50.0, "gpu_memory_used_mb": 1992.0, "gpu_memory_total_mb": 6144.0}
    or None if CUDA is unavailable or nvidia-smi fails.
    """
    if not torch.cuda.is_available():
        return None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None

    # Parse the CSV output: "50, 1992, 6144"
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) < 3:
        return None

    try:
        return {
            "gpu_utilization_percent": float(parts[0]),
            "gpu_memory_used_mb": float(parts[1]),
            "gpu_memory_total_mb": float(parts[2]),
        }
    except ValueError:
        return None


# ===========================================================================
# Loss decomposition for fused batches
# ===========================================================================

def per_adapter_loss_tensors(
    logits: torch.Tensor,
    labels: torch.Tensor,
    adapter_ids: torch.Tensor,
    adapter_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """
    Compute per-adapter mean cross-entropy losses as differentiable tensors.

    This is the primitive used both for training (sum these and backward) and
    for logging (call .item() on each tensor).

    The summed-mean formulation is what makes fused multi-adapter training
    gradient-equivalent to standalone training: since adapter i's LoRA
    parameters only receive gradients from adapter i's samples, normalizing
    each per-adapter term by its own token count (T_i) gives each adapter the
    same effective gradient magnitude it would see training alone -- instead
    of the T_i / T_total dilution you get from a single mean-over-all-tokens
    loss.

    Returns:
      Dict mapping adapter_name -> scalar tensor (mean CE over that adapter's
      valid tokens). Adapters absent from the batch, or with zero valid tokens,
      are omitted.
    """
    # For causal LM loss, we shift: predict token t+1 from position t.
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()

    # Per-token CE loss (no reduction -- we need individual token losses).
    token_losses = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shifted_labels.shape)

    valid_mask = shifted_labels.ne(-100)

    losses: dict[str, torch.Tensor] = {}
    for adapter_index, adapter_name in enumerate(adapter_names):
        sample_mask = adapter_ids == adapter_index
        if not bool(sample_mask.any()):
            continue
        adapter_valid = valid_mask[sample_mask]
        if int(adapter_valid.sum().item()) == 0:
            continue
        losses[adapter_name] = token_losses[sample_mask][adapter_valid].mean()
    return losses


def per_adapter_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    adapter_ids: torch.Tensor,
    adapter_names: tuple[str, ...],
) -> dict[str, float]:
    """
    Float version of per_adapter_loss_tensors for logging.

    Returns:
      Dict like {"ag_news": 5.38, "emotion": 7.74} with mean loss per adapter.
    """
    return {
        name: float(tensor.item())
        for name, tensor in per_adapter_loss_tensors(
            logits.detach(), labels, adapter_ids, adapter_names
        ).items()
    }


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_adapter(
    model: RoutedCausalLM,
    job: "AdapterJob",
    eval_loader,
    device: torch.device,
    limit_steps: int = 8,
) -> dict[str, float]:
    """
    Run a quick validation pass for one adapter job.

    Puts the model in eval mode, runs up to `limit_steps` batches from the
    validation set, and returns the average loss. Then restores training mode.

    Returns: {"validation_loss": <float>}
    """
    model.eval()  # Disable dropout, etc.
    losses = []
    with torch.no_grad():  # No gradient computation during evaluation
        for step, batch in enumerate(eval_loader):
            if step >= limit_steps:
                break
            # Tag all samples in this batch as belonging to this adapter.
            batch["adapter_ids"] = build_adapter_ids(batch, job.adapter_index)
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            losses.append(float(outputs.loss.item()))
    model.train()  # Restore training mode
    if not losses:
        return {"validation_loss": float("nan")}
    return {"validation_loss": sum(losses) / len(losses)}


# ===========================================================================
# AdapterJob -- represents one training job (one adapter + its data)
# ===========================================================================

@dataclass(slots=True)
class AdapterJob:
    """
    Bundles everything needed to train one LoRA adapter.

    Fields:
      name               -- Human-readable name for this job (e.g. "ag_news")
      dataset_key        -- Key into DATASET_SPECS (determines which data to load)
      adapter_index      -- Integer index of this adapter in the RoutedCausalLM
                            (e.g. 0 for the first adapter, 1 for the second)
      train_loader       -- PyTorch DataLoader for the training split
      eval_loader        -- PyTorch DataLoader for the validation split
      optimizer          -- AdamW optimizer that ONLY updates this adapter's LoRA params
      train_iterator     -- A Python iterator over train_loader (so we can call next()
                            to get the next batch; wraps around when exhausted)
      steps_completed    -- How many training steps this job has done so far
      tokens_seen        -- Total number of labeled tokens this job has trained on
      completion_time_sec -- Wall-clock time (since run start) when this job finished
                            all its assigned steps. Used to compute JCT (job completion time).
    """
    name: str
    dataset_key: str
    adapter_index: int
    train_loader: any
    eval_loader: any
    optimizer: AdamW
    train_iterator: any
    steps_completed: int = 0
    tokens_seen: int = 0
    completion_time_sec: float | None = None

    def next_batch(self) -> dict[str, torch.Tensor]:
        """
        Get the next batch from this job's training data.
        If the dataloader is exhausted, restart from the beginning (epoch wrap-around).
        """
        try:
            batch = next(self.train_iterator)
        except StopIteration:
            # Dataset exhausted -- start a new epoch.
            self.train_iterator = iter(self.train_loader)
            batch = next(self.train_iterator)
        return batch


# ===========================================================================
# JobSpec -- lightweight descriptor for defining jobs from the command line
# ===========================================================================

@dataclass(frozen=True, slots=True)
class JobSpec:
    """
    A lightweight description of an adapter job before it's built.

    name        -- The adapter's unique name (used as the key everywhere)
    dataset_key -- Which dataset to use (key into DATASET_SPECS)

    These can differ! E.g. you can have two jobs both using the "ag_news" dataset
    but with different adapter names ("ag_news_1", "ag_news_2") for scaling experiments.
    """
    name: str
    dataset_key: str


def parse_job_specs(job_entries: list[str]) -> list[JobSpec]:
    """
    Parse command-line job entries into JobSpec objects.

    Accepts two formats:
      "ag_news"              -> JobSpec(name="ag_news", dataset_key="ag_news")
      "my_job=ag_news"       -> JobSpec(name="my_job", dataset_key="ag_news")

    The second format lets you create multiple distinct adapter jobs that
    share the same underlying dataset (useful for scaling experiments).
    """
    parsed: list[JobSpec] = []
    seen_names: set[str] = set()
    for entry in job_entries:
        if "=" in entry:
            # Format: "job_name=dataset_key"
            name, dataset_key = entry.split("=", maxsplit=1)
        else:
            # Format: "dataset_key" (name defaults to the dataset key)
            name = entry
            dataset_key = entry

        name = name.strip()
        dataset_key = dataset_key.strip()
        if not name or not dataset_key:
            raise ValueError(f"Invalid job entry '{entry}'. Expected 'job_name=dataset_key' or 'dataset_key'.")
        if dataset_key not in DATASET_SPECS:
            raise ValueError(
                f"Unknown dataset key '{dataset_key}'. Available datasets: {', '.join(sorted(DATASET_SPECS))}."
            )
        if name in seen_names:
            raise ValueError(f"Duplicate job name '{name}' is not allowed.")

        seen_names.add(name)
        parsed.append(JobSpec(name=name, dataset_key=dataset_key))
    return parsed


# ===========================================================================
# BaselineRunner -- the main training orchestrator
# ===========================================================================

class BaselineRunner:
    """
    Orchestrates model setup and all three baseline training strategies.

    Lifecycle:
      1. __init__ loads the model, wraps it with LoRA, builds all AdapterJobs
      2. Call one of: run_sequential(), run_time_sliced(), run_fixed_set_simultaneous()
      3. The run method trains, evaluates, and writes results JSON to disk
    """
    def __init__(
        self,
        job_specs: list[JobSpec],
        train_steps: int = DEFAULT_TRAIN_STEPS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        learning_rate: float = DEFAULT_LR,
        seed: int = DEFAULT_SEED,
        warmup_steps: int = DEFAULT_WARMUP_STEPS,
        output_dir: Path = RUNS_DIR,
        run_name: str | None = None,
    ) -> None:
        self.job_specs = job_specs          # What jobs to train
        self.train_steps = train_steps      # Measured (recorded) steps per job
        self.batch_size = batch_size        # Samples per batch per job
        self.max_length = max_length        # Max tokens per sample
        self.learning_rate = learning_rate  # AdamW LR for LoRA params
        self.seed = seed
        # Steps run before the timer starts and before any record/counter update.
        # Pays one-time CUDA JIT/compile/cache costs so the first measured step
        # is steady-state instead of dominated by warmup.
        self.warmup_steps = max(0, warmup_steps)
        self.output_dir = output_dir        # Where to write the result JSON
        self.run_name = run_name            # Optional custom filename for the output

        # Set random seeds for reproducibility.
        set_seed(seed)

        # Pick CPU or GPU.
        self.device = pick_device()

        # Load the pre-trained model from local disk and wrap with LoRA.
        self.lora_config = RoutedLoraConfig(rank=8, alpha=16.0, dropout=0.05)
        self.tokenizer, base_model = load_base_model()
        self.model = prepare_multi_adapter_model(
            base_model,
            [spec.name for spec in job_specs],
            lora_config=self.lora_config,
        )
        self.model.to(self.device)
        self.model.train()  # Put in training mode (enables dropout, etc.)

        # Build one AdapterJob per job spec (each with its own dataloader + optimizer).
        self.jobs = self._build_jobs(job_specs)

        # Wall-clock timing for the current run (set when a run method starts).
        self.run_start_time = 0.0
        self.last_run_wall_time_sec = 0.0

    def release_resources(self) -> None:
        """Drop references so one baseline run does not pin GPU memory for the next."""
        for job in self.jobs.values():
            job.train_iterator = None
            job.train_loader = None
            job.eval_loader = None
            job.optimizer = None
        self.jobs.clear()
        self.model = None
        self.tokenizer = None
        _release_torch_memory()

    def _log_run_header(self, baseline_name: str) -> None:
        """Emit a concise configuration summary at the start of each measured run."""
        LOGGER.info(
            "Starting baseline=%s device=%s jobs=%d seed=%d train_steps=%d warmup_steps=%d "
            "batch_size=%d max_length=%d lr=%.6f lora_rank=%d lora_alpha=%.1f lora_dropout=%.2f",
            baseline_name,
            self.device,
            len(self.job_specs),
            self.seed,
            self.train_steps,
            self.warmup_steps,
            self.batch_size,
            self.max_length,
            self.learning_rate,
            self.lora_config.rank,
            self.lora_config.alpha,
            self.lora_config.dropout,
        )
        job_summary = ", ".join(f"{spec.name}={spec.dataset_key}" for spec in self.job_specs)
        LOGGER.info("Jobs: %s", job_summary)

    def _log_job_start(self, baseline_name: str, job: AdapterJob, index: int, total_jobs: int) -> None:
        """Emit a job-level progress line for baselines that process jobs individually."""
        LOGGER.info(
            "Running baseline=%s job=%d/%d name=%s dataset=%s steps=%d",
            baseline_name,
            index,
            total_jobs,
            job.name,
            job.dataset_key,
            self.train_steps,
        )

    def _log_run_footer(self, baseline_name: str, output_path: Path, summary: dict[str, object]) -> None:
        """Emit a short run summary after results have been written."""
        tokens_per_sec = summary.get("aggregate_tokens_per_sec")
        wall_time = summary.get("total_wall_time_sec")
        mean_jct = summary.get("mean_jct_sec")
        peak_gpu_memory = summary.get("peak_gpu_memory_mb")
        log_line = (
            f"Finished baseline={baseline_name} wall_time_sec={wall_time:.3f} "
            f"tokens_per_sec={tokens_per_sec:.3f} mean_jct_sec={mean_jct:.3f}"
        )
        if peak_gpu_memory is not None:
            log_line += f" peak_gpu_memory_mb={peak_gpu_memory:.1f}"
        LOGGER.info("%s output=%s", log_line, output_path)

    def _build_jobs(self, job_specs: list[JobSpec]) -> dict[str, AdapterJob]:
        """
        Create an AdapterJob for each JobSpec.

        Each job gets:
          - Its own train and validation DataLoaders
          - Its own AdamW optimizer (only updates that adapter's LoRA parameters)
          - An iterator over the training data
        """
        jobs = {}
        for offset, spec in enumerate(job_specs):
            # Per-job seed offset: every job gets a deterministic but distinct
            # shuffle order so that two jobs reading the same dataset don't
            # train on identical batches in the same step.
            train_loader = build_dataloader(
                spec.dataset_key,
                "train",
                tokenizer=self.tokenizer,
                batch_size=self.batch_size,
                max_length=self.max_length,
                seed=self.seed + offset,
            )
            eval_loader = build_dataloader(
                spec.dataset_key,
                "validation",
                tokenizer=self.tokenizer,
                batch_size=self.batch_size,
                max_length=self.max_length,
                seed=self.seed + offset,
            )
            # Look up this adapter's integer index in the RoutedCausalLM.
            adapter_index = self.model.adapter_index[spec.name]
            # Create an optimizer that ONLY touches this adapter's LoRA A and B matrices.
            optimizer = AdamW(self.model.adapter_parameters(spec.name), lr=self.learning_rate)
            jobs[spec.name] = AdapterJob(
                name=spec.name,
                dataset_key=spec.dataset_key,
                adapter_index=adapter_index,
                train_loader=train_loader,
                eval_loader=eval_loader,
                optimizer=optimizer,
                train_iterator=iter(train_loader),
            )
        return jobs

    # -----------------------------------------------------------------------
    # Timing and GPU helpers
    # -----------------------------------------------------------------------

    def _memory_snapshot_mb(self) -> float | None:
        """Return peak GPU memory allocated (in MB) since the last reset, or None on CPU."""
        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

    def _run_elapsed(self) -> float:
        """Seconds elapsed since this run started (from self.run_start_time)."""
        return time.perf_counter() - self.run_start_time

    def _warmup(self, baseline: str) -> None:
        """
        Run self.warmup_steps un-recorded steps to absorb one-time CUDA costs.

        Warmup steps DO update model weights and DO advance the dataloader
        (otherwise the very first measured step still pays the JIT cost).
        Counters that feed JCT/throughput summaries are saved and restored so
        that the measured run reports exactly self.train_steps per job.
        """
        if self.warmup_steps <= 0:
            return

        jobs = list(self.jobs.values())
        # Snapshot per-job counters so warmup is invisible to summary metrics.
        snapshot = [(job.steps_completed, job.tokens_seen, job.completion_time_sec) for job in jobs]

        if baseline == "sequential" or baseline == "time_sliced":
            # Round-robin warmup steps across jobs so every adapter's LoRA
            # path is exercised at least once before timing begins.
            for warmup_step in range(self.warmup_steps * len(jobs)):
                self._step_job(jobs[warmup_step % len(jobs)])
        elif baseline == "fixed_set_simultaneous":
            for _ in range(self.warmup_steps):
                self._fused_step(jobs)
        else:
            raise ValueError(f"Unknown baseline for warmup: {baseline}")

        # Restore counters so the measured run starts from a clean slate.
        for job, (steps, tokens, completion) in zip(jobs, snapshot):
            job.steps_completed = steps
            job.tokens_seen = tokens
            job.completion_time_sec = completion

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        _cuda_sync()

    def _mark_job_completion(self, job: AdapterJob) -> None:
        """
        Record the wall-clock time when a job finishes all its assigned steps.
        Only marks once (the first time steps_completed >= train_steps).
        This is used to compute JCT (Job Completion Time) in the summary.
        """
        if job.completion_time_sec is None and job.steps_completed >= self.train_steps:
            job.completion_time_sec = self._run_elapsed()

    # -----------------------------------------------------------------------
    # Summary / metrics aggregation
    # -----------------------------------------------------------------------

    def _summary_from_records(self, baseline_name: str, records: list[dict[str, float]]) -> dict[str, object]:
        """
        Aggregate per-step metrics into a run-level summary.

        Computes:
          - total_wall_time_sec:  Real wall-clock time for the entire run
          - total_step_time_sec:  Sum of individual step times (excludes overhead)
          - aggregate_tokens_per_sec:  Total tokens / wall time
          - aggregate_steps_per_sec:   Total steps / wall time
          - mean_jct_sec / p95_jct_sec:  Job Completion Time statistics
          - GPU utilization and memory stats (if available)
        """
        # Filter to only step records (exclude the final peak_gpu_memory record).
        step_records = [record for record in records if "step_time_sec" in record]
        total_wall_time_sec = self.last_run_wall_time_sec
        total_step_time_sec = sum(float(record["step_time_sec"]) for record in step_records)
        total_tokens = sum(float(record.get("tokens", 0.0)) for record in step_records)
        total_steps = len(step_records)
        aggregate_tokens_per_sec = total_tokens / total_wall_time_sec if total_wall_time_sec > 0 else 0.0
        aggregate_steps_per_sec = total_steps / total_wall_time_sec if total_wall_time_sec > 0 else 0.0

        # JCT (Job Completion Time): how long until each job finished its training.
        completion_times = {
            job.name: float(job.completion_time_sec) if job.completion_time_sec is not None else None for job in self.jobs.values()
        }
        valid_completion_times = [value for value in completion_times.values() if value is not None]
        mean_jct = sum(valid_completion_times) / len(valid_completion_times) if valid_completion_times else None
        # P95 JCT: with few jobs this is just the max, but named p95 for consistency
        # with how it would be reported at larger scale.
        p95_jct = max(valid_completion_times) if valid_completion_times else None

        # GPU stats (averaged / maxed across steps).
        gpu_utils = [float(record["gpu_utilization_percent"]) for record in step_records if "gpu_utilization_percent" in record]
        gpu_memory_used = [float(record["gpu_memory_used_mb"]) for record in step_records if "gpu_memory_used_mb" in record]

        summary: dict[str, object] = {
            "baseline": baseline_name,
            "total_wall_time_sec": total_wall_time_sec,
            "total_step_time_sec": total_step_time_sec,
            "aggregate_tokens": total_tokens,
            "aggregate_tokens_per_sec": aggregate_tokens_per_sec,
            "aggregate_steps": total_steps,
            "aggregate_steps_per_sec": aggregate_steps_per_sec,
            "job_completion_time_sec": completion_times,
            "mean_jct_sec": mean_jct,
            "p95_jct_sec": p95_jct,
        }

        if gpu_utils:
            summary["mean_gpu_utilization_percent"] = sum(gpu_utils) / len(gpu_utils)
            summary["max_gpu_utilization_percent"] = max(gpu_utils)
        if gpu_memory_used:
            summary["mean_gpu_memory_used_mb"] = sum(gpu_memory_used) / len(gpu_memory_used)
            summary["max_gpu_memory_used_mb"] = max(gpu_memory_used)

        peak_records = [float(record["peak_gpu_memory_mb"]) for record in records if "peak_gpu_memory_mb" in record]
        if peak_records:
            summary["peak_gpu_memory_mb"] = max(peak_records)

        summary["training_loss_per_job"] = self._training_loss_per_job(step_records)

        return summary

    def _training_loss_per_job(self, step_records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
        """Aggregate training losses per job so runs can be compared for convergence."""
        per_job_losses: dict[str, list[float]] = {spec.name: [] for spec in self.job_specs}

        for record in step_records:
            if "job" in record and "loss" in record:
                per_job_losses.setdefault(str(record["job"]), []).append(float(record["loss"]))
                continue
            for spec in self.job_specs:
                loss_key = f"loss_{spec.name}"
                if loss_key in record:
                    per_job_losses.setdefault(spec.name, []).append(float(record[loss_key]))

        summaries: dict[str, dict[str, float]] = {}
        dataset_by_name = {spec.name: spec.dataset_key for spec in self.job_specs}
        for job_name, losses in per_job_losses.items():
            if not losses:
                continue
            summaries[job_name] = {
                "dataset_key": dataset_by_name.get(job_name, ""),
                "num_records": len(losses),
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "mean_loss": sum(losses) / len(losses),
                "min_loss": min(losses),
                "max_loss": max(losses),
            }
        return summaries

    # -----------------------------------------------------------------------
    # Single training step (used by sequential and time_sliced baselines)
    # -----------------------------------------------------------------------

    def _step_job(self, job: AdapterJob, zero_grad: bool = True, step_optimizer: bool = True) -> dict[str, float]:
        """
        Execute one training step for a single adapter job.

        This is the building block for the sequential and time_sliced baselines.
        It does: zero_grad -> forward -> loss.backward() -> optimizer.step()
        for one job's batch.

        Args:
          job            -- Which adapter job to train
          zero_grad      -- Whether to zero gradients before the step (default True)
          step_optimizer -- Whether to call optimizer.step() after backward (default True)

        Returns a dict of metrics for this step:
          loss, tokens, step_time_sec, tokens_per_sec, and optional GPU stats.
        """
        if zero_grad:
            job.optimizer.zero_grad(set_to_none=True)

        # Get the next batch from this job's training data.
        batch = job.next_batch()
        tokens = count_labeled_tokens(batch)

        # Tag every sample in this batch with this job's adapter index.
        batch["adapter_ids"] = build_adapter_ids(batch, job.adapter_index)
        batch = move_batch_to_device(batch, self.device)

        # Forward + backward. Sync first so step_time excludes work queued
        # by previous steps; sync after to include this step's GPU work.
        _cuda_sync()
        step_start = time.perf_counter()
        outputs = self.model(**batch)
        loss = outputs.loss
        loss.backward()

        if step_optimizer:
            job.optimizer.step()

        _cuda_sync()
        step_time = time.perf_counter() - step_start

        # Update job counters.
        job.steps_completed += 1
        job.tokens_seen += tokens
        self._mark_job_completion(job)  # Record JCT if this was the last step

        metrics = {
            "loss": float(loss.item()),
            "tokens": float(tokens),
            "step_time_sec": step_time,
            "tokens_per_sec": float(tokens / step_time) if step_time > 0 else 0.0,
        }
        # Optionally append GPU utilization snapshot.
        snapshot = gpu_snapshot()
        if snapshot is not None:
            metrics.update(snapshot)
        return metrics

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    def _evaluate_all(self) -> dict[str, dict[str, float]]:
        """Run validation for every adapter job and return their losses."""
        return {
            job.name: evaluate_adapter(self.model, job, job.eval_loader, self.device) for job in self.jobs.values()
        }

    # -----------------------------------------------------------------------
    # Fused batch construction (used by fixed_set_simultaneous)
    # -----------------------------------------------------------------------

    def _build_fused_batch(self, jobs: list[AdapterJob]) -> tuple[dict[str, torch.Tensor], dict[str, float], dict[str, int]]:
        """
        Build a single "fused" batch by concatenating one batch from each job.

        Example with 2 jobs, batch_size=2:
          Job A batch: input_ids shape [2, 30]
          Job B batch: input_ids shape [2, 25]
          -> Pad job B to length 30 (the max)
          -> Concatenate along dim 0 -> fused batch shape [4, 30]
          -> adapter_ids = [0, 0, 1, 1]  (first 2 samples are job A, last 2 are job B)

        Returns:
          fused_batch     -- Dict with input_ids, attention_mask, labels, adapter_ids
                            (all concatenated along the batch dimension)
          per_job_tokens  -- Dict mapping job name -> number of labeled tokens from that job
          per_job_samples -- Dict mapping job name -> number of samples from that job
        """
        # Pull one batch from each job.
        job_batches = [(job, job.next_batch()) for job in jobs]

        # Find the longest sequence across all jobs' batches.
        target_length = max(batch["input_ids"].shape[1] for _, batch in job_batches)

        merged: dict[str, list[torch.Tensor]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "adapter_ids": [],
        }
        per_job_tokens: dict[str, int] = {}
        per_job_samples: dict[str, int] = {}

        for job, batch in job_batches:
            # Pad this job's batch to the target length (right-padding).
            pad_len = target_length - batch["input_ids"].shape[1]
            merged["input_ids"].append(
                torch.nn.functional.pad(batch["input_ids"], (0, pad_len), value=self.tokenizer.pad_token_id)
            )
            merged["attention_mask"].append(torch.nn.functional.pad(batch["attention_mask"], (0, pad_len), value=0))
            merged["labels"].append(torch.nn.functional.pad(batch["labels"], (0, pad_len), value=-100))
            # Create adapter_ids: every sample in this chunk gets this job's adapter index.
            merged["adapter_ids"].append(build_adapter_ids(batch, job.adapter_index))
            per_job_tokens[job.name] = count_labeled_tokens(batch)
            per_job_samples[job.name] = int(batch["input_ids"].shape[0])

        # Concatenate all jobs' batches into one big batch along dim 0.
        fused_batch = {key: torch.cat(parts, dim=0) for key, parts in merged.items()}
        return fused_batch, {key: float(value) for key, value in per_job_tokens.items()}, per_job_samples

    # -----------------------------------------------------------------------
    # Results writing
    # -----------------------------------------------------------------------

    def _write_results(self, baseline_name: str, records: list[dict[str, float]]) -> Path:
        """
        Write the complete run results to a JSON file.

        The JSON contains:
          - Run config (baseline, device, dataset keys, hyperparams)
          - Per-step metrics (loss, tokens/sec, GPU stats, etc.)
          - Aggregated summary (wall time, throughput, JCT, GPU stats)
          - Evaluation results (validation loss per adapter)
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_stem = self.run_name or baseline_name
        output_path = self.output_dir / f"{output_stem}.json"
        summary = self._summary_from_records(baseline_name, records)
        payload = {
            "baseline": baseline_name,
            "device": str(self.device),
            "dataset_keys": [spec.dataset_key for spec in self.job_specs],
            "job_specs": [{"name": spec.name, "dataset_key": spec.dataset_key} for spec in self.job_specs],
            "train_steps": self.train_steps,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "warmup_steps": self.warmup_steps,
            "lora": {
                "rank": self.lora_config.rank,
                "alpha": self.lora_config.alpha,
                "dropout": self.lora_config.dropout,
            },
            "metrics": records,
            "summary": summary,
            "evaluation": self._evaluate_all(),
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._log_run_footer(baseline_name, output_path, summary)
        return output_path

    # ===================================================================
    # BASELINE 1: Sequential
    # ===================================================================

    def _fused_step(self, jobs: list[AdapterJob]) -> dict[str, object]:
        """
        Execute one fused multi-adapter training step.

        Returns a dict with the fields needed to build a step record. Used by
        both warmup (record discarded) and the measured loop.
        """
        for job in jobs:
            job.optimizer.zero_grad(set_to_none=True)

        fused_batch, per_job_tokens, per_job_samples = self._build_fused_batch(jobs)
        macro_tokens = sum(per_job_tokens.values())
        fused_batch = move_batch_to_device(fused_batch, self.device)

        _cuda_sync()
        macro_start = time.perf_counter()
        outputs = self.model(**fused_batch)

        # Use sum of per-adapter mean losses as the training objective, NOT
        # the HF mean-over-all-tokens loss. The single-mean form dilutes each
        # adapter's gradient by T_i / T_total relative to standalone training;
        # the sum-of-means form keeps each adapter's gradient magnitude equal
        # to what it would see training alone on its own sub-batch. Gradient
        # isolation (adapter i only gets grads from adapter i's samples) is
        # still handled by the routing masks in RoutedLoRAConv1D.
        adapter_loss_tensors = per_adapter_loss_tensors(
            outputs.logits,
            fused_batch["labels"],
            fused_batch["adapter_ids"],
            self.model.adapter_names,
        )
        training_loss = sum(adapter_loss_tensors.values())
        training_loss.backward()
        for job in jobs:
            job.optimizer.step()
        _cuda_sync()
        macro_time = time.perf_counter() - macro_start

        adapter_losses = {
            name: float(tensor.item()) for name, tensor in adapter_loss_tensors.items()
        }

        return {
            "loss": float(training_loss.item()),
            "macro_tokens": macro_tokens,
            "macro_time": macro_time,
            "per_job_tokens": per_job_tokens,
            "per_job_samples": per_job_samples,
            "adapter_losses": adapter_losses,
        }

    def run_sequential(self) -> Path:
        """
        Train each adapter job to completion, one after another.

        Schedule:
          Job A: step 1, step 2, ..., step N
          Job B: step 1, step 2, ..., step N
          (Job A is fully trained before Job B starts)

        This is the simplest baseline. Each job gets exclusive use of the GPU
        for its entire training. Good JCT for early jobs, bad for later ones.
        """
        records = []
        self._warmup("sequential")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        self._log_run_header("sequential")
        self.run_start_time = time.perf_counter()

        for index, job in enumerate(self.jobs.values(), start=1):
            self._log_job_start("sequential", job, index, len(self.jobs))
            for local_step in range(self.train_steps):
                metrics = self._step_job(job)
                records.append(
                    {
                        "global_step": len(records),
                        "job": job.name,
                        "job_step": local_step + 1,
                        "baseline": "sequential",
                        **metrics,
                    }
                )

        self.last_run_wall_time_sec = self._run_elapsed()
        peak_memory = self._memory_snapshot_mb()
        if peak_memory is not None:
            records.append({"baseline": "sequential", "peak_gpu_memory_mb": peak_memory})
        return self._write_results("sequential", records)

    # ===================================================================
    # BASELINE 2: Time-sliced (round-robin)
    # ===================================================================

    def run_time_sliced(self) -> Path:
        """
        Alternate training steps between jobs in a round-robin fashion.

        Schedule (with 2 jobs, 2 steps each = 4 total steps):
          Step 1: Job A step 1
          Step 2: Job B step 1
          Step 3: Job A step 2
          Step 4: Job B step 2

        All jobs make progress concurrently (interleaved). Fairer JCT than
        sequential, but each job takes longer to finish than if it ran alone.
        Only one job's batch is on the GPU at any time.
        """
        records = []
        jobs = list(self.jobs.values())
        # Total steps = steps_per_job * number_of_jobs.
        total_steps = self.train_steps * len(jobs)
        self._warmup("time_sliced")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        self._log_run_header("time_sliced")
        self.run_start_time = time.perf_counter()

        for global_step in range(total_steps):
            # Round-robin: cycle through jobs.
            job = jobs[global_step % len(jobs)]
            metrics = self._step_job(job)
            records.append(
                {
                    "global_step": global_step + 1,
                    "job": job.name,
                    "job_step": job.steps_completed,
                    "baseline": "time_sliced",
                    **metrics,
                }
            )

        self.last_run_wall_time_sec = self._run_elapsed()
        peak_memory = self._memory_snapshot_mb()
        if peak_memory is not None:
            records.append({"baseline": "time_sliced", "peak_gpu_memory_mb": peak_memory})
        return self._write_results("time_sliced", records)

    # ===================================================================
    # BASELINE 3: Fixed-set simultaneous (fused multi-adapter)
    # ===================================================================

    def run_fixed_set_simultaneous(self) -> Path:
        """
        Fuse samples from ALL jobs into one batch and train with one forward +
        one backward pass per step.

        Schedule (with 2 jobs, N steps):
          Step 1: Fused batch [A samples, B samples] -> 1 forward -> 1 backward -> step both optimizers
          Step 2: Fused batch [A samples, B samples] -> 1 forward -> 1 backward -> step both optimizers
          ...

        This is the key baseline for the paper. It tests whether we can train
        multiple adapters simultaneously with better GPU utilization than
        sequential/time-sliced approaches.

        How it works each step:
          1. Zero gradients for ALL jobs' optimizers
          2. Build a fused batch (concatenate one batch from each job)
          3. ONE forward pass through the model (RoutedLoRAConv1D routes each
             sample to its adapter's LoRA weights)
          4. Compute a per-adapter mean loss for each adapter, and sum them
             into the training loss. This is crucial for correctness: a single
             mean-over-all-tokens loss would dilute each adapter's gradient by
             T_i / T_total. Summing per-adapter means gives each adapter the
             same effective gradient as standalone training.
          5. ONE backward pass (gradients flow to each adapter's A/B matrices
             only from their own samples, thanks to the routing masks)
          6. Step each job's optimizer (applies the accumulated gradients)
        """
        records = []
        jobs = list(self.jobs.values())
        self._warmup("fixed_set_simultaneous")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        self._log_run_header("fixed_set_simultaneous")
        self.run_start_time = time.perf_counter()

        for fused_step in range(self.train_steps):
            step = self._fused_step(jobs)
            macro_tokens = step["macro_tokens"]
            macro_time = step["macro_time"]
            per_job_tokens = step["per_job_tokens"]
            per_job_samples = step["per_job_samples"]
            adapter_losses = step["adapter_losses"]

            for job in jobs:
                job.steps_completed += 1
                job.tokens_seen += int(per_job_tokens[job.name])
                self._mark_job_completion(job)

            record = {
                "global_step": fused_step + 1,
                "baseline": "fixed_set_simultaneous",
                "active_jobs": len(jobs),
                "mean_loss": step["loss"],
                "tokens": macro_tokens,
                "step_time_sec": macro_time,
                "tokens_per_sec": float(macro_tokens / macro_time) if macro_time > 0 else 0.0,
            }
            for job_name, token_count in per_job_tokens.items():
                record[f"tokens_{job_name}"] = token_count
            for job_name, sample_count in per_job_samples.items():
                record[f"samples_{job_name}"] = sample_count
            for job_name, adapter_loss in adapter_losses.items():
                record[f"loss_{job_name}"] = adapter_loss
            snapshot = gpu_snapshot()
            if snapshot is not None:
                record.update(snapshot)
            records.append(record)

        self.last_run_wall_time_sec = self._run_elapsed()
        peak_memory = self._memory_snapshot_mb()
        if peak_memory is not None:
            records.append({"baseline": "fixed_set_simultaneous", "peak_gpu_memory_mb": peak_memory})
        return self._write_results("fixed_set_simultaneous", records)


# ===========================================================================
# Public entry point
# ===========================================================================

def run_baseline(
    baseline: str,
    job_entries: list[str],
    train_steps: int = DEFAULT_TRAIN_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: int = DEFAULT_SEED,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    output_dir: Path = RUNS_DIR,
    run_name: str | None = None,
) -> Path:
    """
    Top-level entry point: parse job entries, build the runner, and execute a baseline.

    Called by scripts/run_baseline.py and scripts/run_experiment_family2.py.

    Args:
      baseline    -- One of "sequential", "time_sliced", "fixed_set_simultaneous"
      job_entries -- List of job strings, e.g. ["ag_news", "emotion"] or ["job1=ag_news", "job2=emotion"]
      train_steps -- Number of training steps per job
      batch_size  -- Samples per batch per job
      max_length  -- Max tokens per sample
      output_dir  -- Directory to write the result JSON
      run_name    -- Optional custom name for the output file (defaults to baseline name)

    Returns the path to the written JSON results file.
    """
    job_specs = parse_job_specs(job_entries)
    runner = BaselineRunner(
        job_specs=job_specs,
        train_steps=train_steps,
        batch_size=batch_size,
        max_length=max_length,
        seed=seed,
        warmup_steps=warmup_steps,
        output_dir=output_dir,
        run_name=run_name,
    )
    try:
        if baseline == "sequential":
            return runner.run_sequential()
        if baseline == "time_sliced":
            return runner.run_time_sliced()
        if baseline == "fixed_set_simultaneous":
            return runner.run_fixed_set_simultaneous()
        raise ValueError(f"Unsupported baseline '{baseline}'.")
    finally:
        runner.release_resources()
