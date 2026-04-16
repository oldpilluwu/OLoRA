"""
run_baseline.py -- Run a single baseline training loop and write results to JSON.

Usage examples:
  # Run the sequential baseline with default settings (16 steps, batch 2, max_length 128):
  python scripts/run_baseline.py --baseline sequential

  # Run the fused baseline with custom settings:
  python scripts/run_baseline.py --baseline fixed_set_simultaneous --train-steps 200 --batch-size 2 --max-length 128

  # Run with custom adapter job names (useful for scaling experiments):
  python scripts/run_baseline.py --baseline time_sliced --jobs ag1=ag_news ag2=ag_news emotion

Results are written to runs/<baseline>.json (or runs/<run-name>.json if --run-name is set).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from olora.runtime import run_baseline
from olora.settings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_SEED,
    DEFAULT_TRAIN_STEPS,
    DEFAULT_WARMUP_STEPS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one of the OLORA baseline training loops.")
    parser.add_argument(
        "--baseline",
        required=True,
        choices=["sequential", "time_sliced", "fixed_set_simultaneous"],
        help="Which baseline to run.",
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        default=["ag_news", "emotion"],
        help="Adapter jobs to include. Use 'dataset_key' or 'job_name=dataset_key'.",
    )
    parser.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    output_path = run_baseline(
        baseline=args.baseline,
        job_entries=args.jobs,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    print(output_path)


if __name__ == "__main__":
    main()
