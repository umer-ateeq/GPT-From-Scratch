# Run logs

Every `train.py` run writes a folder here. The point is that a reported number
and the log that produced it travel together, so a figure in the documentation
can always be traced back to the run that generated it.

The original training run recorded none of this, and three separate bugs went
unnoticed for months as a direct result. That is audit bug 4 in
[../docs/AUDIT.md](../docs/AUDIT.md), and this directory is the fix.

## What each run folder contains

| File | Contents |
|---|---|
| `config.json` | every hyperparameter, the model config, the git commit, the seed, Python and torch versions, the GPU name, and the exact command line |
| `metrics.jsonl` | one JSON line per evaluation: step, tokens seen, learning rate, train and val loss, perplexity, tokens/sec, achieved TFLOP/s, MFU, peak GPU memory |
| `summary.json` | final losses, total tokens, optimizer steps, steady-state and end-to-end throughput, MFU, peak memory, wall time |
| `loss_curve.png` | train and validation loss against tokens seen |
| `ckpt_last.pt`, `ckpt_best.pt` | weights and optimizer state. **Gitignored**, since they are hundreds of MB |

`tokens_seen` is accumulated from `x.numel()` on the tensors that actually pass
through the model, never computed from the config. That distinction is the whole
reason the original token count was wrong.

## What is here

**`p100_throughput/`** is the run behind the throughput and memory figures in
[../docs/RESULTS.md](../docs/RESULTS.md): 20M tokens on a Kaggle P100 at the
released checkpoint's batch shape, purely to measure speed. Only `summary.json`
was retrieved from that session. Its `final_val_perplexity` of 708 is not a
quality result and is not reported as one; it is a freshly initialized model 152
steps into training.

**`cpu_example/`** is a complete run of the full artifact set, produced on CPU
with a tiny model so it is small enough to commit. It exists so the format above
can be inspected without running anything. It is a smoke run and its numbers mean
nothing.

Reproduce the P100 measurement with [../docs/MEASURE.md](../docs/MEASURE.md).
