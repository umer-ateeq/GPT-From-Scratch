# Measuring throughput and peak GPU memory

How the throughput and peak-memory figures in [RESULTS.md](RESULTS.md) were
produced, and how to reproduce them.

They were missing for a long time because the original training run recorded
nothing (audit bug 4), and an early CV draft quoted a throughput figure with no
instrumented run behind it. Rather than keep a number nobody could check, it was
removed and then earned properly.

**Result of running this: 10,200 tokens/sec steady state, 6.12 GB peak, on a
Kaggle P100 at batch 32 x 128, fp16.** The procedure below is confirmed working,
including the PyTorch fix in the next section.

**Caveat on the evidence.** Only `summary.json` was retrieved from that session;
`config.json` and `metrics.jsonl` were lost when the Kaggle output archive proved
too large to download. So the per-step throughput series quoted in
[RESULTS.md](RESULTS.md) comes from the console output of that run rather than
from a committed log, and `summary.json` predates the MFU logging, so its MFU was
computed afterwards. That is weaker evidence than this project demands of itself,
and it is flagged rather than smoothed over. Re-running this notebook with the
current `train.py` and committing the whole `runs/` folder is the fix.

**Run this on the same hardware the model was trained on: a Kaggle notebook with
a P100.** Throughput is a property of code *and* hardware together, so a figure
measured on a different GPU would not describe this project.

## The short way

[../notebooks/measure_throughput_kaggle_p100.ipynb](../notebooks/measure_throughput_kaggle_p100.ipynb)
does everything below. It is **self-contained**: it carries the project's source
files inline and writes them to disk, so it needs no clone, no dataset upload and
no setup.

1. Kaggle → **Create → New Notebook → File → Import Notebook**, upload it
2. **Session options → Accelerator → GPU P100**
3. **Session options → Internet → On**
4. **Run All**, about 20 minutes

### Kaggle's PyTorch no longer runs on the P100

This bites immediately and is worth stating plainly, because the failure looks
like a code bug and is not one.

Kaggle's preinstalled PyTorch is compiled for compute capability **7.0 and
above**. The Tesla P100 is **6.0**. The stock build therefore reports
`torch.cuda.is_available() == True`, moves the model to the GPU without
complaint, and then dies on the first kernel launch:

```
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the
current PyTorch installation. The current PyTorch install supports CUDA
capabilities sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120.

torch.AcceleratorError: CUDA error: no kernel image is available for execution
on the device
```

The notebook handles this in two steps:

- **It installs `torch==2.5.1+cu121` when it detects a pre-Volta GPU**, since that
  wheel still ships Pascal kernels. No kernel restart is needed, because training
  runs as a subprocess (`!python train.py`) which starts a fresh interpreter.
- **It then launches a real matmul and a real backward pass before doing anything
  else.** `is_available()` says nothing about whether the binary has kernels for
  the chip; the only way to know is to launch one. Without this check the notebook
  spends five minutes tokenizing before discovering it cannot train.

If the fix does not take, switch **Accelerator → T4 x2** and re-run from the top.
A T4 measurement is perfectly valid, it simply has to be reported as a T4 rather
than a P100, and it will not describe the hardware the released checkpoint was
trained on.

Its final cell prints SHA-256 hashes of the files it wrote. **These are the
hashes of the copies embedded in the notebook at the time it was generated, not of
the current repository files**, which have since been edited (the MFU fix, the
Windows encoding fix, and the bug-5 correction all landed afterwards). Regenerate
the notebook before relying on them as an integrity check:

| File | sha256 (first 16) |
|---|---|
| `config.py` | `6c002573ff215678` |
| `model.py` | `477240a17f61c1cf` |
| `data.py` | `83f7917ae1e528ac` |
| `tokenize_data.py` | `08e80dd9461d3be9` |
| `train.py` | `ce3278d6b2811bfe` |

That is what makes the notebook's numbers attributable to this repository's code
rather than to something pasted into a cell. If a hash differs, the file was
edited after the notebook was generated; regenerate it before trusting the run.

## The long way, cell by cell

## Setup

In a Kaggle notebook: **Settings > Accelerator > GPU P100**, and
**Settings > Internet > On**.

```python
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

Confirm it prints `Tesla P100-PCIE-16GB`. If Kaggle gave you a T4 or a different
accelerator, stop and switch, or the number will describe the wrong machine.

```python
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>
!pip install -q tiktoken datasets
```

`torch`, `numpy`, `tqdm` and `matplotlib` are already installed on Kaggle.

## Build a small corpus

Throughput depends on tensor shapes and hardware, not on what the tokens say, so
a small corpus measures the same number as the full 8B one. 50M tokens is plenty
and takes a few minutes.

```python
!python tokenize_data.py --out train.bin      --max-tokens 50e6 --dump CC-MAIN-2024-10
!python tokenize_data.py --out validation.bin --max-tokens 2e6  --dump CC-MAIN-2024-18
```

## Run the measurement

The flags below reproduce the **exact shape of the original run**: 32 sequences
of 128 tokens per optimizer step, learning rate 4e-4. Accumulation is 1 because
that is what actually ran (AUDIT.md bug 5). Throughput at a different batch shape
is a different number, so this matters.

```python
!python train.py \
    --train-bin train.bin --val-bin validation.bin \
    --batch-size 32 --context 128 --grad-accum 1 \
    --lr 4e-4 --dtype float16 \
    --train-tokens 20e6 --eval-freq 10 --run-name p100_throughput
```

20M tokens is roughly 150 optimizer steps, which is well past the point where
throughput stabilizes. Ignore the first reported figure: it includes CUDA context
setup and the first kernel autotune.

P100 notes: it supports fp16 arithmetic at full rate, so `--dtype float16` is
correct. It does **not** support bfloat16, and `torch.compile` needs Triton with
compute capability 7.0 or newer while the P100 is 6.0, so neither is used here.

## Read the numbers

```python
import json
run = "runs/p100_throughput"
summary = json.load(open(f"{run}/summary.json"))
config  = json.load(open(f"{run}/config.json"))

print("GPU             :", config["device_name"])
print("torch           :", config["torch_version"])
print("dtype           :", config["args"]["dtype"])
print("batch x context :", config["args"]["batch_size"], "x", config["args"]["context"])
print("grad accum      :", config["args"]["grad_accum"])
print("tokens/step     :", f"{config['tokens_per_optimizer_step']:,}")
print()
print("AVG TOKENS/SEC  :", f"{summary['avg_tokens_per_sec']:,}")
print("PEAK GPU MEMORY :", summary["peak_gpu_mem_gb"], "GB")
print("tokens seen     :", f"{summary['tokens_seen']:,}")
print("optimizer steps :", summary["optimizer_steps"])
print("wall time       :", summary["total_wall_time_s"], "s")
```

Per-evaluation throughput, if you want to see it settle:

```python
import json
for line in open(f"{run}/metrics.jsonl"):
    m = json.loads(line)
    print(f"step {m['step']:4d}  {m['tokens_per_sec']:>7,} tok/s  "
          f"{m['peak_gpu_mem_gb']:.2f} GB  loss {m['train_loss']:.3f}")
```

## Report it correctly

**Always attach the hardware, dtype and batch shape.** The same code on the same
GPU differs by more than 2x between `--batch-size 8 --dtype float32` and
`--batch-size 64 --dtype float16`. A bare "75K tokens/sec" is not a claim anyone
can check, which is exactly why that figure was removed from the CV.

Write it as:

> N tokens/sec on a single P100 at batch 32 x 128, fp16, gradient accumulation 32

Then update, in this order, so no number exists in two places with two values:

1. The "Throughput and memory" table in [RESULTS.md](RESULTS.md), keeping the
   hardware, dtype and batch shape attached to the figure
2. The `Throughput` and `Peak GPU memory` rows in [../README.md](../README.md)
3. `HARDWARE` in [../config.py](../config.py), so the number lives in code too
4. The CV bullet, last, because it is the first thing anyone checks

Commit the run folder too. `runs/**/ckpt_*.pt` is gitignored, so only the logs and
the loss curve go in, which is the intent:

```python
!git add runs/p100_throughput && git status --short
```

That way the published number and the log that produced it travel together, which
is the whole point of the run logging.

## Peak memory, and what it tells you

`peak_gpu_mem_gb` comes from `torch.cuda.max_memory_allocated()`, which counts
tensors PyTorch allocated. It excludes the CUDA context and allocator caching, so
it reads lower than `nvidia-smi`. Both are legitimate; say which one you mean.

The interesting comparison is against the 16 GB the P100 has. Gradient
accumulation is what keeps this run inside that budget: 32 micro-batches of 32
sequences give the gradient of a 1024-sequence batch while only one micro-batch
of activations is ever resident. Running the same effective batch without
accumulation would need roughly 32x the activation memory and would not fit.
