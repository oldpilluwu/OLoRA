"""
run_experiment_family2.py -- Hardened sweep across job counts and seeds.

For each job_count x baseline x seed, run one measured training run and
collect its summary. Then aggregate across seeds (mean / std / N) per
(job_count, baseline) so headline numbers come with error bars rather
than as one-shot point estimates.

Output structure:
  runs/family2/
    <count>_jobs/
      job_specs.json
      seed_<n>/
        sequential.json
        time_sliced.json
        fixed_set_simultaneous.json
    summary_raw.csv         # one row per seed
    summary_raw.json
    summary_agg.csv         # one row per (job_count, baseline) with mean/std
    summary_agg.md          # human-readable aggregated table
    summary_agg.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from olora.runtime import run_baseline
from olora.settings import DEFAULT_WARMUP_STEPS


DEFAULT_BASELINES = ["sequential", "time_sliced", "fixed_set_simultaneous"]
DEFAULT_JOB_COUNTS = [2, 4, 8]
DEFAULT_DATASET_POOL = ["ag_news", "emotion"]
DEFAULT_SEEDS = [42, 43, 44]
# Hardened defaults: large enough that warmup-corrected timings are
# above the per-step noise floor on a 6 GB consumer GPU.
DEFAULT_TRAIN_STEPS = 32
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_LENGTH = 128


def make_job_entries(job_count: int, dataset_pool: list[str]) -> list[str]:
    """
    Generate job entry strings by cycling through the dataset pool with
    distinct adapter names per repeat (e.g. ag_news_1, ag_news_2, ...).
    """
    entries = []
    per_dataset_index = {dataset_key: 0 for dataset_key in dataset_pool}
    for job_index in range(job_count):
        dataset_key = dataset_pool[job_index % len(dataset_pool)]
        per_dataset_index[dataset_key] += 1
        entries.append(f"{dataset_key}_{per_dataset_index[dataset_key]}={dataset_key}")
    return entries


def load_summary(path: Path) -> dict:
    """Read one run JSON and flatten its summary into a single dict row."""
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
    summary["run_file"] = str(path)
    return summary


RAW_FIELDS = [
    "job_count",
    "baseline",
    "seed",
    "device",
    "train_steps",
    "batch_size",
    "max_length",
    "warmup_steps",
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

# Metrics aggregated across seeds (mean / std / n).
AGG_METRICS = [
    "total_wall_time_sec",
    "aggregate_tokens_per_sec",
    "aggregate_steps_per_sec",
    "mean_jct_sec",
    "p95_jct_sec",
    "mean_gpu_utilization_percent",
    "mean_gpu_memory_used_mb",
    "max_gpu_memory_used_mb",
]


def write_csv(rows: list[dict], output_path: Path, fieldnames: list[str]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _stats(values: list[float]) -> tuple[float | None, float | None, int]:
    """Return (mean, sample_std, n) skipping None/NaN. Std is None if n < 2."""
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(clean)
    if n == 0:
        return None, None, 0
    mean = sum(clean) / n
    if n < 2:
        return mean, None, n
    variance = sum((x - mean) ** 2 for x in clean) / (n - 1)
    return mean, math.sqrt(variance), n


def aggregate(rows: list[dict]) -> list[dict]:
    """Group raw per-seed rows by (job_count, baseline) and compute mean/std/n."""
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        key = (int(row["job_count"]), str(row["baseline"]))
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict] = []
    for (job_count, baseline), group in sorted(grouped.items()):
        out: dict = {
            "job_count": job_count,
            "baseline": baseline,
            "n_seeds": len(group),
            "seeds": sorted(int(r["seed"]) for r in group if r.get("seed") is not None),
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
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def render_markdown_agg(rows: list[dict]) -> str:
    columns = [
        ("Jobs", lambda r: str(r["job_count"])),
        ("Baseline", lambda r: r["baseline"]),
        ("Seeds", lambda r: str(r["n_seeds"])),
        ("Wall(s)", lambda r: _fmt_mean_std(r["total_wall_time_sec_mean"], r["total_wall_time_sec_std"])),
        ("Tok/s", lambda r: _fmt_mean_std(r["aggregate_tokens_per_sec_mean"], r["aggregate_tokens_per_sec_std"])),
        ("Steps/s", lambda r: _fmt_mean_std(r["aggregate_steps_per_sec_mean"], r["aggregate_steps_per_sec_std"])),
        ("Mean JCT", lambda r: _fmt_mean_std(r["mean_jct_sec_mean"], r["mean_jct_sec_std"])),
        ("P95 JCT", lambda r: _fmt_mean_std(r["p95_jct_sec_mean"], r["p95_jct_sec_std"])),
        ("GPU Util%", lambda r: _fmt_mean_std(r["mean_gpu_utilization_percent_mean"], r["mean_gpu_utilization_percent_std"], decimals=1)),
        ("GPU Mem MB", lambda r: _fmt_mean_std(r["mean_gpu_memory_used_mb_mean"], r["mean_gpu_memory_used_mb_std"], decimals=0)),
    ]
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(getter(row) for _, getter in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hardened Family 2 sweep across job counts and seeds.")
    parser.add_argument("--job-counts", nargs="+", type=int, default=DEFAULT_JOB_COUNTS)
    parser.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--dataset-pool", nargs="+", default=DEFAULT_DATASET_POOL)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "family2_parity")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict] = []

    for job_count in args.job_counts:
        job_entries = make_job_entries(job_count, args.dataset_pool)
        job_dir = args.output_dir / f"{job_count}_jobs"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job_specs.json").write_text(json.dumps(job_entries, indent=2), encoding="utf-8")

        for seed in args.seeds:
            seed_dir = job_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            for baseline in args.baselines:
                output_path = run_baseline(
                    baseline=baseline,
                    job_entries=job_entries,
                    train_steps=args.train_steps,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    seed=seed,
                    warmup_steps=args.warmup_steps,
                    output_dir=seed_dir,
                    run_name=baseline,
                )
                raw_rows.append(load_summary(output_path))

    raw_rows.sort(key=lambda row: (int(row["job_count"]), str(row["baseline"]), int(row["seed"])))
    aggregated = aggregate(raw_rows)

    raw_csv = args.output_dir / "summary_raw.csv"
    raw_json = args.output_dir / "summary_raw.json"
    agg_csv = args.output_dir / "summary_agg.csv"
    agg_json = args.output_dir / "summary_agg.json"
    agg_md = args.output_dir / "summary_agg.md"

    write_csv(raw_rows, raw_csv, RAW_FIELDS)
    raw_json.write_text(json.dumps(raw_rows, indent=2), encoding="utf-8")

    agg_fields = ["job_count", "baseline", "n_seeds", "seeds"]
    for metric in AGG_METRICS:
        agg_fields += [f"{metric}_mean", f"{metric}_std", f"{metric}_n"]
    write_csv(aggregated, agg_csv, agg_fields)
    agg_json.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
    agg_md.write_text(render_markdown_agg(aggregated) + "\n", encoding="utf-8")

    print(render_markdown_agg(aggregated))
    print()
    print(f"Raw CSV : {raw_csv}")
    print(f"Agg CSV : {agg_csv}")
    print(f"Agg MD  : {agg_md}")


if __name__ == "__main__":
    main()
