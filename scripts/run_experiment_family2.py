"""
run_experiment_family2.py -- Sweep across job counts to measure scaling behavior.

This script implements "Experiment Family 2" from the plan: it runs all three
baselines at multiple job counts (e.g. 2, 4, 8 simultaneous adapter jobs) and
collects the results into summary files.

The key question this experiment answers:
  "How does throughput, JCT, and GPU utilization change as we add more
   concurrent adapter jobs?"

Since we only have 2 real datasets (ag_news, emotion), this script creates
MULTIPLE adapter jobs that REUSE the same datasets under different names.
E.g. with job_count=4 and dataset_pool=[ag_news, emotion]:
  -> job entries: ["ag_news_1=ag_news", "emotion_1=emotion", "ag_news_2=ag_news", "emotion_2=emotion"]
Each job gets its own independent LoRA adapter, even though two of them train
on ag_news data and two on emotion data.

Usage:
  python scripts/run_experiment_family2.py --job-counts 2 4 8 --train-steps 4 --batch-size 1 --max-length 64

Output structure:
  runs/family2/
    2_jobs/
      job_specs.json              # What jobs were used
      sequential.json             # Baseline results
      time_sliced.json
      fixed_set_simultaneous.json
    4_jobs/
      ...
    8_jobs/
      ...
    summary.csv                   # Combined results across all job counts
    summary.md                    # Markdown table for quick viewing
    summary.json                  # Machine-readable summary
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from olora.runtime import run_baseline


DEFAULT_BASELINES = ["sequential", "time_sliced", "fixed_set_simultaneous"]
DEFAULT_JOB_COUNTS = [2, 4, 8]
DEFAULT_DATASET_POOL = ["ag_news", "emotion"]


def make_job_entries(job_count: int, dataset_pool: list[str]) -> list[str]:
    """
    Generate job entry strings for a given job count by cycling through the dataset pool.

    Example with job_count=4, dataset_pool=["ag_news", "emotion"]:
      -> ["ag_news_1=ag_news", "emotion_1=emotion", "ag_news_2=ag_news", "emotion_2=emotion"]

    The "name=dataset_key" format creates distinct adapter names that map back
    to the same underlying datasets. This lets us simulate many concurrent jobs
    even with only 2 real datasets.
    """
    entries = []
    # Track how many times each dataset has been used (for naming: ag_news_1, ag_news_2, etc.)
    per_dataset_index = {dataset_key: 0 for dataset_key in dataset_pool}
    for job_index in range(job_count):
        # Cycle through datasets: ag_news, emotion, ag_news, emotion, ...
        dataset_key = dataset_pool[job_index % len(dataset_pool)]
        per_dataset_index[dataset_key] += 1
        # E.g. "ag_news_2=ag_news" means adapter named "ag_news_2" using the ag_news dataset.
        entries.append(f"{dataset_key}_{per_dataset_index[dataset_key]}={dataset_key}")
    return entries


def load_summary(path: Path) -> dict:
    """
    Load a run result JSON and extract its summary block into a flat dict
    that's easy to tabulate and compare across runs.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {}).copy()
    # Add top-level fields that aren't in the summary block.
    summary["baseline"] = payload.get("baseline")
    summary["device"] = payload.get("device")
    summary["job_count"] = len(payload.get("job_specs", payload.get("dataset_keys", [])))
    summary["train_steps"] = payload.get("train_steps")
    summary["batch_size"] = payload.get("batch_size")
    summary["run_file"] = str(path)
    summary["job_specs"] = payload.get("job_specs", [])
    return summary


def write_csv(rows: list[dict], output_path: Path) -> None:
    """Write the summary rows to a CSV file with a fixed set of columns."""
    fieldnames = [
        "baseline",
        "device",
        "job_count",
        "train_steps",
        "batch_size",
        "total_wall_time_sec",
        "total_step_time_sec",
        "aggregate_tokens",
        "aggregate_tokens_per_sec",
        "aggregate_steps",
        "aggregate_steps_per_sec",
        "mean_jct_sec",
        "p95_jct_sec",
        "mean_gpu_utilization_percent",
        "max_gpu_utilization_percent",
        "mean_gpu_memory_used_mb",
        "max_gpu_memory_used_mb",
        "run_file",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def format_value(value: object) -> str:
    """Format a value for display: floats get 3 decimal places, None becomes '-'."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(rows: list[dict]) -> str:
    """Render the summary rows as a Markdown table for quick viewing."""
    columns = [
        ("job_count", "Jobs"),
        ("baseline", "Baseline"),
        ("total_wall_time_sec", "Wall(s)"),
        ("aggregate_tokens_per_sec", "Tok/s"),
        ("aggregate_steps_per_sec", "Steps/s"),
        ("mean_jct_sec", "Mean JCT"),
        ("p95_jct_sec", "P95 JCT"),
        ("mean_gpu_utilization_percent", "GPU Util"),
        ("max_gpu_utilization_percent", "GPU Util Max"),
        ("mean_gpu_memory_used_mb", "GPU Mem"),
    ]
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for row in rows:
        line = "| " + " | ".join(format_value(row.get(key)) for key, _ in columns) + " |"
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment Family 2 fixed-set baseline sweeps.")
    parser.add_argument("--job-counts", nargs="+", type=int, default=DEFAULT_JOB_COUNTS)
    parser.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--dataset-pool", nargs="+", default=DEFAULT_DATASET_POOL)
    parser.add_argument("--train-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "family2")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []

    # Outer loop: iterate over job counts (e.g. 2, 4, 8).
    for job_count in args.job_counts:
        # Generate the job entries for this count (e.g. 4 jobs cycling through 2 datasets).
        job_entries = make_job_entries(job_count, args.dataset_pool)

        # Create a subdirectory for this job count.
        job_dir = args.output_dir / f"{job_count}_jobs"
        job_dir.mkdir(parents=True, exist_ok=True)
        # Save the job specs so we can see what was used.
        (job_dir / "job_specs.json").write_text(json.dumps(job_entries, indent=2), encoding="utf-8")

        # Inner loop: run each baseline with this job count.
        for baseline in args.baselines:
            run_name = baseline
            output_path = run_baseline(
                baseline=baseline,
                job_entries=job_entries,
                train_steps=args.train_steps,
                batch_size=args.batch_size,
                max_length=args.max_length,
                output_dir=job_dir,
                run_name=run_name,
            )
            summaries.append(load_summary(output_path))

    # Sort by (job_count, baseline) for a clean presentation.
    summaries.sort(key=lambda row: (int(row["job_count"]), str(row["baseline"])))

    # Write combined summary files.
    csv_path = args.output_dir / "summary.csv"
    markdown_path = args.output_dir / "summary.md"
    json_path = args.output_dir / "summary.json"

    write_csv(summaries, csv_path)
    markdown_path.write_text(render_markdown(summaries) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    # Print the markdown table to stdout for quick review.
    print(render_markdown(summaries))
    print()
    print(f"CSV: {csv_path}")
    print(f"Markdown: {markdown_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
