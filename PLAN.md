Start with a **scheduler paper prototype, not a kernel paper prototype**. Your shortest path to a publishable result is to prove that **online insertion of newly arriving LoRA jobs into an already-running fused training group** improves throughput/JCT/fairness with low disruption. That is the gap you need to isolate, because standard PEFT does not support mixed-adapter batch **training** as an official path, while mLoRA, LoRAFusion, and tLoRA already cover broader concurrent multi-LoRA execution and scheduling ideas. ([Hugging Face][1])

Here is the plan I would use.

## 1) Fix the target before writing code

Your first milestone should be:

**“A single-node runtime that trains multiple LoRA adapters simultaneously on one frozen base model, supports online arrival of a new adapter without draining the system, and measures the cost of incremental insertion.”**

Do **not** begin with:

* custom CUDA kernels
* multi-node pipeline parallelism
* full Megatron integration

mLoRA already has concurrent multi-adapter training plus pipeline parallelism and public training entry points, while LoRAFusion already adds kernel fusion and adaptive batching on top of Megatron-LM. If you start there, you will burn time reproducing their infrastructure instead of isolating your novelty. ([GitHub][2])

## 2) Use a staged implementation path

### Phase A: Reproduce baselines first

Spend the first stretch just getting three baselines working:

1. **Sequential baseline**
   One adapter job at a time on a frozen base.

2. **Time-sliced multi-adapter baseline**
   Same base, multiple adapters loaded, alternating steps/jobs.

3. **Fixed-set simultaneous baseline**
   A fused-batch prototype where the set of active adapters is fixed at launch.

This gives you the control group you need before tackling online arrivals. Since PEFT’s mixed-adapter path is documented for inference rather than training, use PEFT mainly for adapter definitions/checkpoint format and implement the mixed-training runtime yourself. ([Hugging Face][1])

### Phase B: Build the minimal execution engine

Implement five runtime objects:

* `BaseRuntime`: one frozen model copy in GPU memory
* `AdapterJob`: adapter weights, optimizer, LR scheduler, dataloader, progress state
* `ActiveSet`: adapters currently participating in fused training
* `BatchPlanner`: constructs the next fused batch from active jobs
* `Executor`: runs one joint forward/backward and dispatches gradients/updates

The first version should be **single process, single node**, with no custom kernels. Keep the LoRA math in plain PyTorch first. The point is to validate correctness and scheduler behavior, not peak speed.

### Phase C: Add online insertion

Only after Phase B works, implement the feature that matters:

* new request arrives
* create adapter + optimizer state
* warm-start its dataloader
* add it to the active set
* include it in the next or next-few fused steps
* measure disruption to existing jobs

This is the heart of your idea. If the system has to stop, rebuild everything, and restart, you have not really solved the problem.

### Phase D: Add incremental planning

Once hot-plug works, improve the planner:

* naive equal-share packing
* weighted packing by priority
* token-budget-aware packing
* length-bucketed packing
* incremental regrouping with a disruption budget

LoRAFusion already uses grouping + bin packing for efficient microbatch construction, and tLoRA already studies online residual-capacity-aware grouping with per-job progress concerns. Your planner should therefore focus on **incremental update cost** and **minimal disturbance**, not just raw packing quality. ([arXiv][3])

## 3) Pick the right implementation starting point

For code, I would do this in order:

**First base:** a lightweight Hugging Face / PyTorch prototype
Reason: fastest path to proving the scheduler idea, even though PEFT itself does not give you the training primitive you need out of the box. ([Hugging Face][1])

**Second base:** mLoRA reproduction
Reason: it already exposes public training scripts like `mlora_train.py` and `mlora_pp_train.py`, supports concurrent LoRA fine-tuning, and gives you a real systems baseline. ([GitHub][2])

**Third base:** LoRAFusion-style scaling path
Reason: only after your scheduler works should you move into Megatron-style execution and fusion work. LoRAFusion already targets that regime and has public code. ([GitHub][4])

## 4) What to implement in the prototype

Your first prototype only needs four nontrivial mechanisms.

### A. Per-sample adapter routing

Each training sample needs an `adapter_id`. The fused batch planner concatenates subbatches from multiple jobs and carries an index map so the runtime knows which LoRA branch applies to which sample.

### B. Per-adapter optimizer isolation

Each adapter keeps its own:

* LoRA parameters
* optimizer state
* LR schedule
* checkpoint history

Only the base model is shared.

### C. Incremental active-set updates

Support:

* add adapter
* pause adapter
* remove finished adapter
* rebalance adapter shares

This should not require a full executor restart.

### D. Instrumentation from day one

Log:

* step time
* tokens/sec
* per-job progress
* insertion latency
* slowdown of existing jobs after insertion
* GPU memory headroom
* GPU utilization

Without this, you will not know whether the idea is helping.

## 5) Design the experiments around your actual claim

Your experiments should answer one question:

**Is online incremental insertion better than either keeping jobs isolated or fully rebuilding groups whenever a new job arrives?**

So run these experiment families.

### Family 1: Correctness / quality parity

Show that simultaneous training does not materially hurt final adapter quality compared with isolated LoRA fine-tuning for the same budget.

Measure:

* validation loss
* task metric for each adapter
* convergence curves

Loss interpretation note for `fixed_set_simultaneous`:

* For `sequential` and `time_sliced`, the logged training loss is the usual per-batch mean cross-entropy over valid labeled tokens for one job at a time.
* For `fixed_set_simultaneous`, the training loss should be computed as the **sum of per-adapter mean losses** within the fused batch, not as one mean over all fused tokens.
* This matters because a single mean over all tokens would dilute adapter `i`'s gradient by `T_i / T_total` relative to standalone training. Summing per-adapter means preserves the gradient scale each adapter would see if trained alone.
* Because of that definition, the fused run's raw training loss is **not directly comparable** to the raw training loss from `sequential` or `time_sliced`, and it is also not directly comparable across different active-set sizes such as 2 vs 4 vs 8 jobs.
* When comparing `fixed_set_simultaneous` against the other two baselines, use per-adapter views instead: per-job final training loss, per-job validation loss, task metrics, and convergence curves normalized per adapter.
* If a single scalar must be reported for readability, prefer average per-adapter loss across the active set or a token-weighted mean loss over all valid tokens in the fused batch, but keep the summed per-adapter objective as the optimization loss used for backpropagation.

### Family 2: Fixed-set throughput

Start with 2, then 4, then 8 concurrent adapters, all known in advance.

Measure:

* aggregate tokens/sec
* aggregate steps/sec
* GPU utilization
* mean job completion time

This anchors you against mLoRA/LoRAFusion-style “fixed active set” thinking. ([VLDB][5])

### Family 3: Online arrival workload

This is your main experiment.

Generate traces where:

* two adapters start
* a third arrives later
* more arrive at random times
* some jobs are short, some long
* ranks and sequence lengths differ

Compare:

* isolated execution
* time-sliced admission
* full regroup/restart
* your incremental insertion scheduler

Primary metrics:

* mean and p95 JCT
* existing-job slowdown after insertion
* insertion-to-first-update latency
* aggregate throughput
* fairness / starvation

tLoRA explicitly frames online grouping and per-job progress guarantees as important, so you should report those metrics clearly. ([arXiv][6])

### Family 4: Heterogeneity stress test

Vary:

* LoRA rank
* batch size
* sequence length distribution
* dataset size
* number of active jobs

This matters because both LoRAFusion and tLoRA treat heterogeneity as a core challenge. ([arXiv][3])

### Family 5: Planner overhead

If your scheduler wins throughput but spends too much time replanning, that weakens the story.

Measure:

* time spent in scheduling/planning
* number of jobs moved or reshaped after each arrival
* number of steps disrupted
* warm-up delay for the new adapter

This is likely where your cleanest novelty can emerge.

## 6) What would count as a publishable result

You do not need to beat LoRAFusion on absolute peak throughput right away. A strong first paper result would be:

* similar quality to isolated LoRA training
* clear throughput/JCT win over isolated and time-sliced baselines
* much lower insertion disruption than full regroup/rebuild
* robustness under heterogeneous online arrivals
* ablation showing the gain comes from incremental admission, not just “more batching”

That would give you a paper story centered on **online incremental fusion**, which is more defensible than just “multi-LoRA training.” tLoRA already occupies the broader “online residual-capacity-aware” space, so your distinct angle should be **low-disruption insertion into already-running fused groups**. ([arXiv][6])

## 7) A concrete 4-week starting schedule

### Week 1

* Reproduce single-adapter LoRA training.
* Reproduce a sequential multi-job baseline.
* Build logging for throughput, JCT, slowdown, memory.

### Week 2

* Implement fixed-set simultaneous fused-batch training.
* Verify gradient isolation and checkpoint correctness.
* Run 2-adapter and 4-adapter sanity tests.

### Week 3

* Add online arrival and active-set updates.
* Measure insertion latency and disruption.
* Implement simple planners: equal-share and weighted-share.

### Week 4

* Add heterogeneity-aware packing: token budget + length bucketing.
* Run trace-driven experiments with staggered arrivals.
* Write ablations: no insertion, insertion with rebuild, insertion with incremental planner.

## 8) The first deliverables I would insist on

By the end of the first implementation pass, you should have:

* a reproducible runtime that can train 2–4 adapters simultaneously
* a trace generator for online arrivals
* a result sheet with throughput, JCT, slowdown, quality
* one figure showing what happens when a new adapter is inserted mid-run
* one ablation proving whether “incremental” actually matters

That is the minimum that tells you whether this is a real paper or just an engineering curiosity.

The most important tactical advice: **prove the scheduler before optimizing the kernels**. mLoRA and LoRAFusion already show that high-performance multi-LoRA systems exist; your edge has to come from the online insertion behavior and the metrics around disruption, fairness, and progress under continuous arrivals. ([GitHub][2])

I can turn this into a concrete repo skeleton next, with module names, interfaces, and the exact first prototype loop.

[1]: https://huggingface.co/docs/peft/developer_guides/lora?utm_source=chatgpt.com "LoRA"
[2]: https://github.com/TUDB-Labs/mLoRA "GitHub - TUDB-Labs/mLoRA: An Efficient \"Factory\" to Build Multiple LoRA Adapters · GitHub"
[3]: https://arxiv.org/pdf/2510.00206?utm_source=chatgpt.com "LoRAFusion: Efficient LoRA Fine-Tuning for LLMs"
[4]: https://github.com/CentML/lorafusion "GitHub - CentML/lorafusion: LoRAFusion: Efficient LoRA Fine-Tuning for LLMs · GitHub"
[5]: https://www.vldb.org/pvldb/vol18/p1948-tang.pdf?utm_source=chatgpt.com "mLoRA: Fine-Tuning LoRA Adapters via Highly-Efficient ..."
[6]: https://arxiv.org/pdf/2602.07263 "tLoRA: Efficient Multi-LoRA Training with Elastic Shared Super-Models"
