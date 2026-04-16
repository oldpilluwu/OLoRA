"""
compare_runs.py -- Print a side-by-side comparison table of baseline run results.

Usage:
  # Compare the three default baselines:
  python scripts/compare_runs.py

  # Compare specific run files:
  python scripts/compare_runs.py runs/family2/2_jobs/sequential.json runs/family2/2_jobs/time_sliced.json

Reads the JSON result files written by run_baseline.py and prints:
  1. A formatted table with key metrics (wall time, throughput, JCT, GPU stats)
  2. Per-job completion times for each baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Default files to compare (the three main baselines in the runs/ folder).
DEFAULT_RUN_FILES = [
    "runs/sequential.json",
    "runs/time_sliced.json",
    "runs/fixed_set_simultaneous.json",
]


def format_value(value: object) -> str:
    """Format a value for display: floats get 3 decimal places, None becomes '-'."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def load_payload(path: Path) -> dict:
    """Load a run result JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows(payloads: list[dict]) -> list[dict[str, object]]:
    """
    Extract the key comparison metrics from each run's payload into a flat row dict.
    Each row represents one baseline run.
    """
    rows = []
    for payload in payloads:
        summary = payload.get("summary", {})
        job_specs = payload.get("job_specs", [])
        rows.append(
            {
                "baseline": payload.get("baseline"),
                "device": payload.get("device"),
                "jobs": len(job_specs) if job_specs else len(payload.get("dataset_keys", [])),
                "train_steps": payload.get("train_steps"),
                "batch_size": payload.get("batch_size"),
                "wall_time_s": summary.get("total_wall_time_sec"),
                "agg_tokens_s": summary.get("aggregate_tokens_per_sec"),
                "agg_steps_s": summary.get("aggregate_steps_per_sec"),
                "mean_jct_s": summary.get("mean_jct_sec"),
                "p95_jct_s": summary.get("p95_jct_sec"),
                "mean_gpu_util_pct": summary.get("mean_gpu_utilization_percent"),
                "max_gpu_util_pct": summary.get("max_gpu_utilization_percent"),
                "mean_gpu_mem_mb": summary.get("mean_gpu_memory_used_mb"),
                "max_gpu_mem_mb": summary.get("max_gpu_memory_used_mb"),
            }
        )
    return rows


def render_table(rows: list[dict[str, object]]) -> str:
    """
    Render a list of row dicts as a fixed-width ASCII table.

    Columns are defined here -- each is a (dict_key, display_header) pair.
    Column widths auto-adjust to fit the data.
    """
    columns = [
        ("baseline", "Baseline"),
        ("device", "Device"),
        ("jobs", "Jobs"),
        ("train_steps", "Steps"),
        ("batch_size", "Batch"),
        ("wall_time_s", "Wall(s)"),
        ("agg_tokens_s", "Tok/s"),
        ("agg_steps_s", "Steps/s"),
        ("mean_jct_s", "Mean JCT"),
        ("p95_jct_s", "P95 JCT"),
        ("mean_gpu_util_pct", "GPU Util"),
        ("max_gpu_util_pct", "GPU Util Max"),
        ("mean_gpu_mem_mb", "GPU Mem"),
        ("max_gpu_mem_mb", "GPU Mem Max"),
    ]

    # Calculate column widths (max of header and all values).
    widths = {}
    for key, label in columns:
        values = [format_value(row.get(key)) for row in rows]
        widths[key] = max(len(label), *(len(value) for value in values))

    header = " | ".join(label.ljust(widths[key]) for key, label in columns)
    separator = "-+-".join("-" * widths[key] for key, _ in columns)
    lines = [header, separator]

    for row in rows:
        line = " | ".join(format_value(row.get(key)).ljust(widths[key]) for key, _ in columns)
        lines.append(line)
    return "\n".join(lines)


def render_job_completion(payloads: list[dict]) -> str:
    """Render per-job completion times for each baseline (the JCT breakdown)."""
    lines = ["", "Per-job completion times (seconds):"]
    for payload in payloads:
        summary = payload.get("summary", {})
        completion = summary.get("job_completion_time_sec", {})
        formatted = ", ".join(f"{name}={format_value(value)}" for name, value in completion.items())
        lines.append(f"- {payload.get('baseline')}: {formatted}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OLORA baseline run summaries.")
    parser.add_argument(
        "run_files",
        nargs="*",
        default=DEFAULT_RUN_FILES,
        help="Run JSON files to compare. Defaults to the three baseline outputs in runs/.",
    )
    args = parser.parse_args()

    paths = [Path(path) for path in args.run_files]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(f"Missing run files: {missing_text}")

    payloads = [load_payload(path) for path in paths]
    rows = build_rows(payloads)
    print(render_table(rows))
    print(render_job_completion(payloads))


if __name__ == "__main__":
    main()
