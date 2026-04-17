# OLORA Prototype

This repository now contains the first implementation pass for the scheduler-paper prototype described in [PLAN.md](C:\Users\fawwa\projects\OLORA\PLAN.md):

- a local Python virtual environment target
- a small-model setup sized for a Windows laptop GPU with 6 GB VRAM
- local download scripts for the model and dataset subsets
- three initial baselines:
  - `sequential`
  - `time_sliced`
  - `fixed_set_simultaneous`
  - `online_insertion`

## Current prototype choices

- Base model: `distilgpt2`
- Adapter jobs: `ag_news` and `emotion`
- Training style: LoRA adapters on a frozen shared base model
- Fixed-set fused training: one mixed batch, one routed forward/backward, per-sample adapter IDs
- Online insertion: dynamically add a new adapter job to the active fused set without rebuilding the model
- Device policy: use CUDA automatically when available, otherwise CPU

`distilgpt2` is intentionally conservative for the first pass so the shared-base multi-adapter runtime stays usable on a 6 GB Windows GPU.

## One-time setup

Create the virtual environment:

```powershell
python -m venv .venv
```

Install the package:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

If you want the environment to use an NVIDIA GPU on Windows, install a CUDA wheel from the official PyTorch index that matches your driver support. On the machine used for this setup, the CUDA-enabled install was:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
```

## Download local assets

```powershell
.\.venv\Scripts\python.exe scripts\download_assets.py
```

This stores everything under:

- `artifacts/models/distilgpt2`
- `artifacts/datasets/ag_news`
- `artifacts/datasets/emotion`

## Run baselines

Example smoke tests:

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline.py --baseline sequential --train-steps 1 --batch-size 1 --max-length 64
.\.venv\Scripts\python.exe scripts\run_baseline.py --baseline time_sliced --train-steps 1 --batch-size 1 --max-length 64
.\.venv\Scripts\python.exe scripts\run_baseline.py --baseline fixed_set_simultaneous --train-steps 1 --batch-size 1 --max-length 64
.\.venv\Scripts\python.exe scripts\run_baseline.py --baseline online_insertion --jobs ag_news emotion --initial-job-count 1 --arrival-steps 4 --train-steps 4 --batch-size 1 --max-length 64
```

Each run writes a JSON result file into `runs/`.

Compare the latest baseline runs:

```powershell
.\.venv\Scripts\python.exe scripts\compare_runs.py
```

Run the Experiment Family 2 fixed-set sweep:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment_family2.py --job-counts 2 4 8 --train-steps 4 --batch-size 1 --max-length 64
.\.venv\Scripts\python.exe scripts\run_experiment_family2.py --output-dir runs\family2_hardened
```

This writes per-run JSON files plus combined `summary.csv`, `summary.md`, and `summary.json` files under `runs/family2/`.

Run the larger Phase C benchmark sweep for a remote GPU server such as an A6000:

```powershell
.\.venv\Scripts\python.exe scripts\run_benchmark_phase_c.py --job-counts 2 4 8 12 --train-steps 128 --batch-size 4 --max-length 128
```

This writes scenario folders plus aggregated benchmark reports under `runs/phase_c_benchmark_a6000/`.

## Notes

- The `fixed_set_simultaneous` baseline now performs true routed fused-set training: samples from different jobs are concatenated into one batch and routed to their adapter-specific LoRA weights inside the same model forward.
- The `online_insertion` baseline starts with a smaller active set, creates a new adapter on arrival, warms its dataloader/optimizer, and admits it into the next fused step without restarting the executor.
- The current metrics include step time, tokens/sec, loss, validation loss, and peak GPU memory when CUDA is available.
- Online insertion runs also log per-job arrival metadata plus insertion latency and simple disruption metrics in `summary.insertion_events`.
- The Family 2 sweep can scale beyond the two local datasets by creating multiple distinct adapter jobs that reuse the same downloaded dataset under different adapter names, which is useful for throughput and JCT stress tests on one machine.
- The next logical step after Phase C is a richer trace-driven experiment driver for staggered arrivals and planner ablations.
