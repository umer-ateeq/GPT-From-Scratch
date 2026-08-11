# Results

Every number this project reports about the **model and the training run** lives
here, next to the command that produces it and the hardware it ran on. The
interpretability results have their own file,
[INDUCTION_HEADS.md](INDUCTION_HEADS.md), on the same terms.

Nothing here is estimated or extrapolated. Where something has not been measured,
or was measured with weaker evidence than this project would like, this file says
so rather than filling in a plausible figure.

---

## The released checkpoint

`weights8b_300epoch.pth`, 538 MB, a bare `state_dict` of 117 tensors.

| Property | Value | Command |
|---|---|---|
| Trainable parameters | 134,077,440 | `python audit_checkpoint.py --ckpt weights8b_300epoch.pth` |
| Causal-mask buffers | 524,288 | same |
| Total tensor elements | 134,601,728 | same |
| Layers / heads / width | 8 / 12 / 768 | same, recovered from tensor shapes |
| Configured context | 256 | same |
| **Effective trained context** | **128** | same, from positional embedding norms |

## Training scale

| Quantity | Value |
|---|---|
| Hardware | one NVIDIA Tesla P100, 16 GB, free Kaggle session |
| Corpus written to disk | **8B tokens**, FineWeb-Edu `CC-MAIN-2024-10`, GPT-2 BPE, uint16 memmap |
| Tokens consumed by the run | **~1.0-1.2B** (300 cycles x 1000 steps x 4,096 tokens), about 15% of the corpus |
| Learning rate | 4e-4, **fixed** (no schedule; see [AUDIT.md](AUDIT.md)) |
| Batch shape | 32 sequences x 128 tokens = 4,096 tokens per optimizer step (accumulation never ran, AUDIT.md bug 5) |
| Tokens per parameter, at the floor | 9.17 |
| Chinchilla-optimal budget for 134M params | ~2.68B tokens (20 per parameter) |

**Corpus size and tokens consumed are different numbers, and conflating them is
the easiest way to overstate a pretraining run by several multiples.** 8B is how
much text the tokenizer wrote; ~1B is how much passed through the model,
sampled as random windows from it. That conflation is exactly what happened here
before the audit.

**Two independent methods agree on roughly 1B tokens.** One reads the weights,
one reads the notebook source, and neither depends on the other.

| Source | Implies |
|---|---|
| The filename's 300 cycles at 32 x 128 | 1.23B tokens, 9.2 per parameter |
| The weight-decay clock (see below) | 234,478 successful optimizer steps, so ~0.96B tokens at the real 4,096 tokens/step |
| The model's actual quality | 3.1x worse than GPT-2-small, exactly what ~9 tokens/parameter predicts |

### The weight-decay clock

AdamW applies two updates per step: a multiplicative decay `p ← p(1 − lr·wd)`,
and a gradient update. Positional rows 128-255 never received a gradient, because
no batch was ever long enough to reach them, so **only the decay applied**. Their
shrinkage therefore integrates the learning rate over every step the checkpoint
ever took:

```
final = initial x (1 - lr x wd)^N
0.002346 / 27.7765 = 8.448e-5
ln(8.448e-5) / ln(1 - 4e-4 x 0.1) = 234,000 steps
```

The method depends on those rows being untouched initialization rather than
trained-then-decayed. Two checks confirm it:

| | Norm spread (std/mean) | Adjacent-row cosine |
|---|---|---|
| Fresh initialization | 2.75% | 0.026 |
| **Checkpoint rows 128-255** | **2.72%** | **0.028** |
| Checkpoint rows 0-127 (trained) | 23.10% | 0.822 |

The dead rows kept their random directions and their uniform norms. They were
only scaled.

---

## Perplexity

Perplexity is `exp(average cross-entropy per token)`: on average the model was as
uncertain as if choosing uniformly among this many tokens. Lower is better.

Two protocols appear below, and mixing them up produces numbers that compare to
nothing:

- **Sequential non-overlapping windows** (`--mode bin`). Every token scored
  exactly once, each window independent.
- **Strided sliding window** (`--mode wikitext`). Each token gets up to
  `max_length - 1` tokens of left context, and overlap tokens are masked out of
  the loss so nothing is counted twice. **Note that the numbers below were run at
  `--stride 128`, equal to `--max-length`**, which means the windows do not
  overlap and the average token sees about 64 tokens of context, not 127. At that
  setting this reduces to non-overlapping chunking. It is applied identically to
  both models, so the 3.10x ratio is unaffected, but it is not the strided
  protocol from the GPT-2 paper and should not be described as one. Running with
  `--stride 64` or lower would give both models more context and lower both
  numbers.

### In domain: FineWeb-Edu held out

Held-out data is `CC-MAIN-2024-18`, a **different Common Crawl snapshot** from
the training data (`CC-MAIN-2024-10`): distribution-matched but fully disjoint.

| Context | Loss | Perplexity | Tokens | Command |
|---|---|---|---|---|
| **128** | 3.661 | **38.89** | 409,600 | `python evaluate.py --ckpt weights8b_300epoch.pth --mode bin --data-bin validation.bin --context 128` |
| 256 | 4.313 | 74.70 | 409,600 | same with `--context 256` |

The 256 row is the model scored outside its trained regime. It is 1.92x worse
for a reason unrelated to model quality: half of every window sits on positional
rows that never received gradient. `evaluate.py` prints a warning when asked to
do this.

### Out of domain: TinyStories

| Context | Loss | Perplexity | Tokens |
|---|---|---|---|
| 128 | 3.567 | 35.41 | 204,800 |

Slightly better than in-domain FineWeb-Edu, which is expected: TinyStories uses
a deliberately small vocabulary and simple syntax, so it is easier to predict
than educational web text even though it is a different distribution.

### Out of domain: WikiText-2, against GPT-2-small on the identical harness

A perplexity number means nothing without a reference measured the same way, so
`evaluate.py --model gpt2` pushes HuggingFace's GPT-2-small through the same
`strided_perplexity` function, the same tokenizer, the same test set and the same
window settings. **The only thing that differs between these two rows is the
model.**

Full WikiText-2 raw test set, 287,644 tokens read, 285,396 scored, context 128,
stride 128, CPU:

| Model | Params | NLL | Perplexity | Command |
|---|---|---|---|---|
| **This model** | 134.08M | 5.2201 | **184.96** | `python evaluate.py --ckpt weights8b_300epoch.pth --mode wikitext --max-length 128` |
| GPT-2-small | 124.44M | 4.0891 | **59.69** | `python evaluate.py --model gpt2 --mode wikitext --max-length 128` |

**This model is 3.10x worse than GPT-2-small on WikiText-2.** That is the honest
headline. It is also the expected one: GPT-2-small saw roughly 8B tokens of
WebText against this model's ~1B of filtered educational web text, and
WikiText-2 is encyclopedic prose far from FineWeb-Edu.

**The baseline also validates the harness.** GPT-2-small's published WikiText-2
perplexity is about 29.4 at context 1024. Cutting the context to 128 gives each
token roughly an eighth of the conditioning information, and this harness
measures 59.69, a degradation in the expected direction and of a plausible size.
So 184.96 is a real property of this model rather than an artefact of the
evaluation code.

The gap between 38.89 in domain and 184.96 on WikiText-2 is 4.8x, which is what
heavy domain specialization on ~1B tokens looks like.

### On the perplexity figure this project used to report

An earlier version of my CV reported **WikiText-2 perplexity 31.23**. That number
was `exp(val_loss)` computed on the in-distribution validation split *during
training*; it never touched WikiText-2. The real figure is 184.96.

31.23 was also never reachable at this scale, being below GPT-2-small's own
published score. A 134M model trained on ~1B tokens does not beat GPT-2-small,
and a pipeline that reports it doing so has a measurement bug rather than a
result. Reporting a benchmark name for a number measured on something else is the
bug; `evaluate.py` printing its dataset, protocol, window settings and token
count on every run is the fix.

---

## Generation

[../SAMPLES.md](../SAMPLES.md) holds six unedited completions from one seeded run.

```bash
python generate.py --ckpt weights8b_300epoch.pth --out SAMPLES.md
```

They are locally fluent, factually unreliable, and visibly repetitive. That is
the behaviour 184.96 out-of-domain perplexity predicts, and it is shown rather
than hidden.

---

## Tests

```bash
python -m pytest tests/ -v
```

23 passed: 9 on the model and training loop, 14 on the analysis code. The analysis tests matter more, because a bug there produces a confident, plausible, wrong scientific claim rather than a loss that will not go down. Each of these maps to a property that was wrong at some point:

| Test | Guards |
|---|---|
| `test_parameter_count_is_exactly_134M` | the headline number drifting |
| `test_untied_output_head_is_a_separate_matrix` | silently dropping 38.6M parameters by tying |
| `test_forward_returns_one_logit_per_vocabulary_entry` | basic wiring |
| `test_attention_cannot_see_the_future` | future-token leakage, which would invalidate every loss number |
| `test_every_parameter_receives_gradient` | a silently dead subnetwork |
| `test_untrained_loss_is_close_to_random_guessing` | the loss being computed over the wrong axis |
| `test_batch_shape_follows_arguments_not_globals` | **audit bug 1** |
| `test_lr_warms_up_then_decays_and_never_climbs` | **audit bug 3** |
| `test_training_run_leaves_a_complete_record` | **audit bug 4**, and `tokens_seen` disagreeing with what reached the model |

---

## Throughput and memory

Measured on the training hardware, at the exact batch shape the released
checkpoint was trained with. Reproduce with [MEASURE.md](MEASURE.md).

| Metric | Value |
|---|---|
| **Training throughput** | **10,200 tokens/sec** steady state, 9,621 end to end |
| **Model FLOPs Utilization** | **31.7%** of the P100's 18.7 TFLOP/s fp16 peak |
| **Peak GPU memory** | **6.12 GB** of the 16 GB available |
| GPU | NVIDIA Tesla P100-PCIE-16GB, Kaggle |
| Precision | fp16 mixed precision |
| Batch shape | 32 sequences x 128 tokens |
| Gradient accumulation | 32 micro-batches, 131,072 tokens per optimizer step |
| Model size in this run | 133,979,136 parameters |
| PyTorch | 2.5.1+cu121 |

Throughput here is the windowed figure `train.py` logs at every evaluation,
covering the preceding 10 optimizer steps. It was flat across the run: 10.1K at
step 10, then 10.2K at steps 20, 30 and 40, while peak memory never moved off
6.1 GB. At 131,072 tokens per step that is 12.85 seconds per optimizer step.

The end-to-end `avg_tokens_per_sec` is **9,621**, 6% lower, because it includes
CUDA context creation and the time spent inside evaluation passes. The
steady-state figure describes training speed; the end-to-end figure describes
wall-clock cost. Both are in the run log.

Verbatim from `runs/p100_throughput/summary.json`:

```json
{
  "final_val_loss": 6.5627,
  "final_val_perplexity": 708.16,
  "tokens_seen": 19922944,
  "optimizer_steps": 152,
  "avg_tokens_per_sec": 9621,
  "peak_gpu_mem_gb": 6.12,
  "n_params": 133979136,
  "total_wall_time_s": 2075.9
}
```

The perplexity of 708 is not a quality result and is not reported as one: this
was a **fresh, randomly initialized model** trained for 20M tokens purely to
measure speed. It is the loss you would expect 152 steps into training. The
released checkpoint's real perplexity numbers are in the tables above.

**Model FLOPs Utilization.** Throughput is only comparable across hardware once
converted to a fraction of the GPU's peak:

```
N_matmul        = 134,077,440 - 38,597,376 (tok_emb) - 196,608 (pos_emb)
                = 95,283,456
FLOPs per token = 6 x 95,283,456 + 12(8)(128)(768) = 581,137,920
achieved        = 581,137,920 x 10,200 = 5.93 TFLOP/s
P100 fp16 peak  = 18.7 TFLOP/s
MFU             = 31.7%
```

**The token embedding is excluded from N on purpose.** Reading a row of `tok_emb`
is a gather, not a matrix multiply, so it costs no FLOPs. nanoGPT counts its
embedding table in N only because its output head is *tied* to it: one parameter
block, counted once, doing the work once. This model's head is **untied**, so
`out_head` is a genuine 768 x 50257 matmul that belongs in N while `tok_emb` is a
separate table of identical size that does not. An earlier version of this file
counted both and reported **44.4%**, inflating the figure by about 40% relative.

No nanoGPT comparison is offered here. Their published MFU excludes embeddings
under a tied head, and the P100's compute-to-bandwidth ratio (18.7 TFLOP/s over
732 GB/s = 25.6 FLOP/byte) is roughly 8x lower than an A100's, which makes a high
MFU *easier* to reach on this card, not harder. Quoting an A100 number beside a
P100 number would flatter this one in two directions at once.

**The parameter count in this run is 133,979,136, not 134,077,440.** That is not
a discrepancy. Running at context 128 allocates 128 positional embedding rows
rather than 256, so the model is exactly `128 x 768 = 98,304` parameters smaller.
It is a useful confirmation that the measurement really did run at the trained
context.

### Two things this measurement shows

**The old 75K tokens/sec claim was about 7x too high.** An early version of my CV
carried it with no instrumented run behind it. The real figure on this hardware
at this batch shape is 10.2K. It had already been removed for lack of evidence;
this is the run that would have contradicted it.

**The run used 38% of the available GPU memory**, 6.1 GB of 16, leaving 9.9 GB
idle. The batch could have been considerably larger at no cost. Together with the
audit finding that training silently ran at 32 x 128 instead of the configured
64 x 256, the same conclusion arrives twice by different routes: this run left
hardware on the table, and not by choice. Logging peak memory next to throughput
is what makes that visible at all.

### A note on running this on a P100 today

Kaggle's preinstalled PyTorch is built for compute capability 7.0 and above,
while the P100 is 6.0. The stock build reports `cuda.is_available() == True`,
moves the model to the GPU, then fails on the first kernel launch with
`no kernel image is available for execution on the device`. Installing
`torch==2.5.1+cu121`, which still ships Pascal kernels, fixes it. Details and the
verbatim error are in [MEASURE.md](MEASURE.md).

## Not swept, by design

The released checkpoint was trained at a **fixed learning rate of 4e-4 and a
fixed batch shape of 32 x 128**. No learning-rate schedule and no hyperparameter
search was part of this project, so there is no sweep to report and none is
claimed anywhere. `train.py` implements warmup and cosine decay because the
notebook intended to and the wiring bug in [AUDIT.md](AUDIT.md) prevented it, not
because a schedule was tuned here.

---

## Reproduction environment

The perplexity and generation numbers above were produced on CPU
(torch 2.12.0+cpu, Python 3.12, Windows). WikiText-2 at context 128 over the full
test set takes about 15 minutes per model on CPU and is near instant on a GPU.
Perplexity is deterministic, so a GPU run reproduces these figures up to
floating-point reduction-order noise. `generate.py` is seeded (`--seed 123`).
