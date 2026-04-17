"""Quality parity check across baselines.

Walks a runs root, groups JSONs by scenario directory (the parent folder
holding sibling baseline files), extracts comparable per-adapter metrics from
each run, and reports the spread across baselines per adapter.

Important for ``fixed_set_simultaneous``:
    Its training objective is the sum of per-adapter mean losses inside the
    fused batch. That summed objective is correct for optimization, but it is
    not directly comparable to the raw single-job batch loss logged by
    ``sequential`` or ``time_sliced``. For parity checks, compare per-adapter
    training summaries and validation loss instead of the fused run's raw
    ``metrics[].mean_loss`` field.

Usage:
    python scripts/parity_check.py runs/family2_hardened
    python scripts/parity_check.py runs/family2_hardened --metrics validation_loss
    python scripts/parity_check.py runs/family2_hardened --tol 5e-3
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Final

BASELINES = ("sequential", "time_sliced", "fixed_set_simultaneous")
METRIC_LABELS: Final[dict[str, str]] = {
    "validation_loss": "validation_loss",
    "final_training_loss": "final_training_loss",
    "mean_training_loss": "mean_training_loss",
}


def load_run_metrics(path: Path) -> dict[str, dict[str, float]]:
    data = json.loads(path.read_text())
    evaluation = data.get("evaluation") or {}
    training = (data.get("summary") or {}).get("training_loss_per_job") or {}

    metrics: dict[str, dict[str, float]] = {
        "validation_loss": {},
        "final_training_loss": {},
        "mean_training_loss": {},
    }

    metrics["validation_loss"] = {
        adapter: float(payload["validation_loss"])
        for adapter, payload in evaluation.items()
        if isinstance(payload, dict) and "validation_loss" in payload
    }
    metrics["final_training_loss"] = {
        adapter: float(payload["final_loss"])
        for adapter, payload in training.items()
        if isinstance(payload, dict) and "final_loss" in payload
    }
    metrics["mean_training_loss"] = {
        adapter: float(payload["mean_loss"])
        for adapter, payload in training.items()
        if isinstance(payload, dict) and "mean_loss" in payload
    }
    return metrics


def discover_groups(root: Path) -> dict[Path, dict[str, Path]]:
    groups: dict[Path, dict[str, Path]] = defaultdict(dict)
    for baseline in BASELINES:
        for path in root.rglob(f"{baseline}.json"):
            groups[path.parent][baseline] = path
    return {k: v for k, v in groups.items() if len(v) >= 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="runs root (e.g. runs/family2_hardened)")
    ap.add_argument(
        "--metrics",
        nargs="+",
        choices=tuple(METRIC_LABELS),
        default=("validation_loss", "final_training_loss"),
        help=(
            "comparable per-adapter metrics to check "
            "(default: validation_loss final_training_loss)"
        ),
    )
    ap.add_argument("--tol", type=float, default=1e-2,
                    help="max allowed spread across baselines (default 1e-2)")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"error: {args.root} does not exist", file=sys.stderr)
        return 2

    groups = discover_groups(args.root)
    if not groups:
        print(f"no baseline JSON groups found under {args.root}", file=sys.stderr)
        return 2

    worst = 0.0
    failures: list[str] = []
    for metric_name in args.metrics:
        print()
        print(f"[metric] {METRIC_LABELS[metric_name]}")
        print(
            f"{'scenario':<48} {'adapter':<18} "
            + " ".join(f"{b[:14]:>14}" for b in BASELINES)
            + f" {'spread':>10} {'ok':>4}"
        )
        print("-" * 120)

        for scenario in sorted(groups):
            files = groups[scenario]
            run_metrics = {b: load_run_metrics(p) for b, p in files.items()}
            adapter_metrics = {
                b: run_metrics[b].get(metric_name, {})
                for b in files
            }
            adapters = sorted({adapter for values in adapter_metrics.values() for adapter in values})
            rel = scenario.relative_to(args.root).as_posix() or "."
            for adapter in adapters:
                vals = {
                    baseline: adapter_metrics.get(baseline, {}).get(adapter)
                    for baseline in BASELINES
                }
                present = [v for v in vals.values() if v is not None]
                spread = max(present) - min(present) if len(present) >= 2 else float("nan")
                worst = max(worst, spread if spread == spread else 0.0)
                cells = " ".join(
                    f"{vals[baseline]:>14.6f}" if vals[baseline] is not None else f"{'--':>14}"
                    for baseline in BASELINES
                )
                ok = spread <= args.tol
                mark = "OK" if ok else "FAIL"
                print(f"{rel:<48} {adapter:<18} {cells} {spread:>10.3e} {mark:>4}")
                if not ok:
                    failures.append(
                        f"{METRIC_LABELS[metric_name]}::{rel}::{adapter} spread={spread:.3e}"
                    )

        print("-" * 120)

    print(f"groups: {len(groups)}   worst spread: {worst:.3e}   tol: {args.tol:.3e}")
    if failures:
        print(f"PARITY FAIL ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PARITY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
