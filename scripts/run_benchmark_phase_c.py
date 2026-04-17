"""
run_benchmark_phase_c.py -- Benchmark harness for Phase C online-insertion runs.

This script is meant for a stronger GPU box such as an RTX A6000. It runs a
repeatable sweep across job-count scenarios and compares:

  - sequential
  - time_sliced
  - fixed_set_simultaneous
  - online_insertion

For each scenario x baseline x seed it writes one JSON run file, then produces:

  - raw per-run CSV / JSON
  - aggregated mean/std tables across seeds
  - a markdown report with throughput, JCT, and insertion metrics
  - per-scenario trace metadata, including the online arrival schedule

Example:
    python scripts/run_benchmark_phase_c.py --job-counts 2 4 8 12 --train-steps 128
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import socket
import subprocess
from pathlib import Path

from olora.runtime import run_baseline
from olora.settings import DEFAULT_WARMUP_STEPS


DEFAULT_BASELINES = [
    "sequential",
    "time_sliced",
    "fixed_set_simultaneous",
    "online_insertion",
]
DEFAULT_JOB_COUNTS = [2, 4, 8, 12]
DEFAULT_DATASET_POOL = ["ag_news", "emotion"]
DEFAULT_SEEDS = [42, 43, 44]

# A6000-friendly defaults. These are intentionally larger than the laptop-scale
# sweep but still conservative enough to be easy to start with.
DEFAULT_TRAIN_STEPS = 128
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_LENGTH = 128
DEFAULT_WARMUP_STEPS_A6000 = max(DEFAULT_WARMUP_STEPS, 4)
DEFAULT_INITIAL_ACTIVE_FRACTION = 0.5
DEFAULT_ARRIVAL_START_FRACTION = 0.20
DEFAULT_ARRIVAL_STOP_FRACTION = 0.80


RAW_FIELDS = [
    "scenario",
    "job_count",
    "baseline",
    "seed",
    "device",
    "train_steps",
    "batch_size",
    "max_length",
    "warmup_steps",
    "initial_job_count",
    "arrival_steps",
    "total_wall_time_sec",
    "total_step_time_sec",
    "aggregate_tokens",
    "aggregate_tokens_per_sec",
    "aggregate_steps",
    "aggregate_steps_per_sec",
    "mean_jct_sec",
    "p95_jct_sec",
    "mean_completion_since_arrival_sec",
    "p95_completion_since_arrival_sec",
    "mean_insertion_latency_sec",
    "max_insertion_latency_sec",
    "mean_insertion_slowdown_ratio",
    "max_insertion_slowdown_ratio",
    "mean_gpu_utilization_percent",
    "max_gpu_utilization_percent",
    "mean_gpu_memory_used_mb",
    "max_gpu_memory_used_mb",
    "peak_gpu_memory_mb",
    "run_file",
]


AGG_METRICS = [
    "total_wall_time_sec",
    "aggregate_tokens_per_sec",
    "aggregate_steps_per_sec",
    "mean_jct_sec",
    "p95_jct_sec",
    "mean_completion_since_arrival_sec",
    "p95_completion_since_arrival_sec",
    "mean_insertion_latency_sec",
    "max_insertion_latency_sec",
    "mean_insertion_slowdown_ratio",
    "max_insertion_slowdown_ratio",
    "mean_gpu_utilization_percent",
    "mean_gpu_memory_used_mb",
    "peak_gpu_memory_mb",
]


def make_job_entries(job_count: int, dataset_pool: list[str]) -> list[str]:
    """
    Generate reproducible job-entry strings by cycling through the dataset pool.

    Example with pool [ag_news, emotion] and job_count=4:
      ag_news_1=ag_news
      emotion_1=emotion
      ag_news_2=ag_news
      emotion_2=emotion
    """
    entries: list[str] = []
    per_dataset_index = {dataset_key: 0 for dataset_key in dataset_pool}
    for job_index in range(job_count):
        dataset_key = dataset_pool[job_index % len(dataset_pool)]
        per_dataset_index[dataset_key] += 1
        entries.append(f"{dataset_key}_{per_dataset_index[dataset_key]}={dataset_key}")
    return entries


def choose_initial_job_count(job_count: int, initial_job_count: int | None, initial_active_fraction: float) -> int:
    """Choose how many jobs start active before online insertions begin."""
    if initial_job_count is not None:
        if initial_job_count < 1 or initial_job_count > job_count:
            raise ValueError(
                f"initial_job_count must be between 1 and job_count={job_count}, got {initial_job_count}."
            )
        return initial_job_count
    computed = max(1, int(round(job_count * initial_active_fraction)))
    return min(job_count, computed)


def build_arrival_steps(
    job_count: int,
    initial_job_count: int,
    train_steps: int,
    start_fraction: float,
    stop_fraction: float,
) -> list[int]:
    """
    Spread pending online arrivals across the measured run window.

    If there are N pending jobs, we place them between start_fraction and
    stop_fraction of the train_steps horizon. Duplicate steps are allowed when
    the number of arrivals exceeds the number of distinct integer step slots.
    """
    pending_count = max(0, job_count - initial_job_count)
    if pending_count == 0:
        return []

    if not (0.0 <= start_fraction <= 1.0 and 0.0 <= stop_fraction <= 1.0):
        raise ValueError("arrival fractions must be in [0, 1].")
    if stop_fraction < start_fraction:
        raise ValueError("arrival_stop_fraction must be >= arrival_start_fraction.")

    if pending_count == 1:
        return [max(1, int(round(train_steps * start_fraction)))]

    start_step = max(1, int(round(train_steps * start_fraction)))
    stop_step = max(start_step, int(round(train_steps * stop_fraction)))
    if stop_step < start_step:
        stop_step = start_step

    positions: list[int] = []
    span = stop_step - start_step
    for index in range(pending_count):
        fraction = index / (pending_count - 1)
        positions.append(start_step + int(round(span * fraction)))
    return positions


def detect_gpu() -> dict[str, str]:
    """Best-effort GPU metadata for the benchmark manifest."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return {}

    name, driver_version, memory_total_mb, *_ = [part.strip() for part in lines[0].split(",")]
    return {
        "gpu_name": name,
        "gpu_driver_version": driver_version,
        "gpu_memory_total_mb": memory_total_mb,
    }


def load_summary(path: Path) -> dict[str, object]:
    """Flatten one benchmark JSON run into a single raw summary row."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {}).copy()
    summary["baseline"] = payload.get("baseline")
    summary["device"] = payload.get("device")
    summary["job_count"] = len(payload.get("job_specs", payload.get("dataset_keys", [])))
    summary["train_steps"] = payload.get("train_steps")
    summary["batch_size"] = payload.get("batch_size")
    summary["max_length"] = payload.get("max_length")
    summary["seed"] = payload.get("seed")
    summary["warmup_steps"] = payload.get("warmup_steps")
    summary["initial_job_count"] = len(payload.get("initial_job_specs", []))
    insertion_events = summary.get("insertion_events") or []
    summary["arrival_steps"] = ",".join(
        str(event["requested_step"])
        for event in insertion_events
        if isinstance(event, dict) and "requested_step" in event
    )
    summary["run_file"] = str(path)
    return summary


def write_csv(rows: list[dict[str, object]], output_path: Path, fieldnames: list[str]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _stats(values: list[object]) -> tuple[float | None, float | None, int]:
    """Return (mean, sample_std, n) while skipping None/NaN values."""
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(clean)
    if n == 0:
        return None, None, 0
    mean = sum(clean) / n
    if n < 2:
        return mean, None, n
    variance = sum((x - mean) ** 2 for x in clean) / (n - 1)
    return mean, math.sqrt(variance), n


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate raw rows by (scenario, baseline) with mean/std/n across seeds."""
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["scenario"]), str(row["baseline"]))
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict[str, object]] = []
    for (scenario, baseline), group in sorted(grouped.items()):
        exemplar = group[0]
        out: dict[str, object] = {
            "scenario": scenario,
            "job_count": exemplar.get("job_count"),
            "baseline": baseline,
            "n_seeds": len(group),
            "seeds": sorted(int(r["seed"]) for r in group if r.get("seed") is not None),
            "initial_job_count": exemplar.get("initial_job_count"),
            "arrival_steps": exemplar.get("arrival_steps"),
        }
        for metric in AGG_METRICS:
            mean, std, n = _stats([r.get(metric) for r in group])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_n"] = n
        aggregated.append(out)
    return aggregated


def _fmt_mean_std(mean: float | None, std: float | None, decimals: int = 3) -> str:
    if mean is None:
        return "-"
    if std is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def render_markdown_agg(rows: list[dict[str, object]]) -> str:
    """Render the aggregated benchmark summary as markdown."""
    columns = [
        ("Scenario", lambda r: str(r["scenario"])),
        ("Jobs", lambda r: str(r["job_count"])),
        ("Baseline", lambda r: str(r["baseline"])),
        ("Seeds", lambda r: str(r["n_seeds"])),
        ("Tok/s", lambda r: _fmt_mean_std(r["aggregate_tokens_per_sec_mean"], r["aggregate_tokens_per_sec_std"])),
        ("Steps/s", lambda r: _fmt_mean_std(r["aggregate_steps_per_sec_mean"], r["aggregate_steps_per_sec_std"])),
        ("Mean JCT", lambda r: _fmt_mean_std(r["mean_jct_sec_mean"], r["mean_jct_sec_std"])),
        ("P95 JCT", lambda r: _fmt_mean_std(r["p95_jct_sec_mean"], r["p95_jct_sec_std"])),
        (
            "Insert Lat(s)",
            lambda r: _fmt_mean_std(r["mean_insertion_latency_sec_mean"], r["mean_insertion_latency_sec_std"]),
        ),
        (
            "Slowdown",
            lambda r: _fmt_mean_std(
                r["mean_insertion_slowdown_ratio_mean"],
                r["mean_insertion_slowdown_ratio_std"],
            ),
        ),
        (
            "GPU Util%",
            lambda r: _fmt_mean_std(
                r["mean_gpu_utilization_percent_mean"],
                r["mean_gpu_utilization_percent_std"],
                decimals=1,
            ),
        ),
        (
            "Peak Mem MB",
            lambda r: _fmt_mean_std(
                r["peak_gpu_memory_mb_mean"],
                r["peak_gpu_memory_mb_std"],
                decimals=0,
            ),
        ),
    ]
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(getter(row) for _, getter in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Phase C benchmark sweep, tuned for larger GPUs such as an A6000.")
    parser.add_argument("--job-counts", nargs="+", type=int, default=DEFAULT_JOB_COUNTS)
    parser.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--dataset-pool", nargs="+", default=DEFAULT_DATASET_POOL)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS_A6000)
    parser.add_argument(
        "--initial-job-count",
        type=int,
        default=None,
        help="Override the number of jobs that start active for online_insertion. Defaults to a fraction of job_count.",
    )
    parser.add_argument(
        "--initial-active-fraction",
        type=float,
        default=DEFAULT_INITIAL_ACTIVE_FRACTION,
        help="Used when --initial-job-count is omitted. Example: 0.5 means half the jobs start active.",
    )
    parser.add_argument(
        "--arrival-start-fraction",
        type=float,
        default=DEFAULT_ARRIVAL_START_FRACTION,
        help="First online arrival step as a fraction of train_steps.",
    )
    parser.add_argument(
        "--arrival-stop-fraction",
        type=float,
        default=DEFAULT_ARRIVAL_STOP_FRACTION,
        help="Last online arrival step as a fraction of train_steps.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run if the target JSON already exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "phase_c_benchmark_a6000",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": detect_gpu(),
        "config": {
            "job_counts": args.job_counts,
            "baselines": args.baselines,
            "dataset_pool": args.dataset_pool,
            "seeds": args.seeds,
            "train_steps": args.train_steps,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "warmup_steps": args.warmup_steps,
            "initial_job_count": args.initial_job_count,
            "initial_active_fraction": args.initial_active_fraction,
            "arrival_start_fraction": args.arrival_start_fraction,
            "arrival_stop_fraction": args.arrival_stop_fraction,
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    raw_rows: list[dict[str, object]] = []

    for job_count in args.job_counts:
        scenario = f"{job_count}_jobs"
        scenario_dir = args.output_dir / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)

        job_entries = make_job_entries(job_count, args.dataset_pool)
        initial_job_count = choose_initial_job_count(
            job_count=job_count,
            initial_job_count=args.initial_job_count,
            initial_active_fraction=args.initial_active_fraction,
        )
        arrival_steps = build_arrival_steps(
            job_count=job_count,
            initial_job_count=initial_job_count,
            train_steps=args.train_steps,
            start_fraction=args.arrival_start_fraction,
            stop_fraction=args.arrival_stop_fraction,
        )

        scenario_config = {
            "scenario": scenario,
            "job_entries": job_entries,
            "initial_job_count": initial_job_count,
            "arrival_steps": arrival_steps,
        }
        (scenario_dir / "scenario.json").write_text(json.dumps(scenario_config, indent=2), encoding="utf-8")

        for seed in args.seeds:
            seed_dir = scenario_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            for baseline in args.baselines:
                run_name = baseline
                output_path = seed_dir / f"{run_name}.json"
                if args.skip_existing and output_path.exists():
                    row = load_summary(output_path)
                    row["scenario"] = scenario
                    raw_rows.append(row)
                    continue

                baseline_kwargs = {
                    "baseline": baseline,
                    "job_entries": job_entries,
                    "train_steps": args.train_steps,
                    "batch_size": args.batch_size,
                    "max_length": args.max_length,
                    "seed": seed,
                    "warmup_steps": args.warmup_steps,
                    "output_dir": seed_dir,
                    "run_name": run_name,
                }
                if baseline == "online_insertion":
                    baseline_kwargs["initial_job_count"] = initial_job_count
                    baseline_kwargs["arrival_steps"] = arrival_steps

                written_path = run_baseline(**baseline_kwargs)
                row = load_summary(written_path)
                row["scenario"] = scenario
                raw_rows.append(row)

    raw_rows.sort(
        key=lambda row: (
            str(row["scenario"]),
            str(row["baseline"]),
            int(row["seed"]),
        )
    )
    aggregated = aggregate(raw_rows)

    raw_csv = args.output_dir / "summary_raw.csv"
    raw_json = args.output_dir / "summary_raw.json"
    agg_csv = args.output_dir / "summary_agg.csv"
    agg_json = args.output_dir / "summary_agg.json"
    agg_md = args.output_dir / "summary_agg.md"

    write_csv(raw_rows, raw_csv, RAW_FIELDS)
    raw_json.write_text(json.dumps(raw_rows, indent=2), encoding="utf-8")

    agg_fields = ["scenario", "job_count", "baseline", "n_seeds", "seeds", "initial_job_count", "arrival_steps"]
    for metric in AGG_METRICS:
        agg_fields += [f"{metric}_mean", f"{metric}_std", f"{metric}_n"]
    write_csv(aggregated, agg_csv, agg_fields)
    agg_json.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
    agg_md.write_text(render_markdown_agg(aggregated) + "\n", encoding="utf-8")

    print(render_markdown_agg(aggregated))
    print()
    print(f"Manifest: {args.output_dir / 'manifest.json'}")
    print(f"Raw CSV : {raw_csv}")
    print(f"Agg CSV : {agg_csv}")
    print(f"Agg MD  : {agg_md}")


if __name__ == "__main__":
    main()
