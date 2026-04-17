"""
run_baseline.py -- Run a single baseline training loop and write results to JSON.

Usage examples:
  # Run the sequential baseline with default settings (16 steps, batch 2, max_length 128):
  python scripts/run_baseline.py --baseline sequential

  # Run the fused baseline with custom settings:
  python scripts/run_baseline.py --baseline fixed_set_simultaneous --train-steps 200 --batch-size 2 --max-length 128

  # Run the online insertion baseline: start one job, insert the second at fused step 4:
  python scripts/run_baseline.py --baseline online_insertion --jobs ag_news emotion --initial-job-count 1 --arrival-steps 4

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
        choices=["sequential", "time_sliced", "fixed_set_simultaneous", "online_insertion"],
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
    parser.add_argument(
        "--initial-job-count",
        type=int,
        default=1,
        help="For online_insertion, how many jobs start active before any arrivals.",
    )
    parser.add_argument(
        "--arrival-steps",
        nargs="*",
        type=int,
        default=None,
        help="For online_insertion, 1-based fused step indices when each pending job arrives.",
    )
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
        initial_job_count=args.initial_job_count,
        arrival_steps=args.arrival_steps,
    )
    print(output_path)


if __name__ == "__main__":
    main()
