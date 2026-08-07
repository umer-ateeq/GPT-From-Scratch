# Results

Every number this project reports lives here, next to the command that produces
it and the hardware it ran on. If a number is not in this file, it is not
claimed anywhere.

Nothing here is estimated or extrapolated. Where something has not been
measured, this file says so rather than filling in a plausible figure.

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
| Corpus written to disk | **8B tokens**, FineWeb-Edu `CC-MAIN-2024-10`, GPT-2 BPE, uint16 memmap |
| Tokens consumed by the run | **~1.23B** (300 x 1000 x 32 x 128), about 15% of the corpus |
| Tokens per parameter | 9.17 |
| Chinchilla-optimal budget for 134M params | ~2.68B tokens (20 per parameter) |
| Fraction of compute-optimal | **~46%** |

**Corpus size and tokens consumed are different numbers, and conflating them is
the easiest way to overstate a pretraining run by several multiples.** 8B is how
much text the tokenizer wrote; 1.23B is how much passed through the model,
sampled as random windows from it. That conflation is exactly what happened here
before the audit.

At 9.17 tokens per parameter this model is undertrained by roughly half against
the Chinchilla rule of thumb. That is the expected outcome of a fixed free-tier
GPU budget, not a defect.

---

## Perplexity

Perplexity is `exp(average cross-entropy per token)`: on average the model was as
uncertain as if choosing uniformly among this many tokens. Lower is better.

Two protocols appear below, and mixing them up produces numbers that compare to
nothing:

- **Sequential non-overlapping windows** (`--mode bin`). Every token scored
  exactly once, each window independent.
- **Strided sliding window** (`--mode wikitext`). The GPT-2 paper protocol: each
  token gets up to `max_length - 1` tokens of left context, and overlap tokens
  are masked out of the loss so nothing is counted twice.

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
WebText against this model's 1.23B of filtered educational web text, and
WikiText-2 is encyclopedic prose far from FineWeb-Edu.

**The baseline also validates the harness.** GPT-2-small's published WikiText-2
perplexity is about 29.4 at context 1024. Cutting the context to 128 gives each
token roughly an eighth of the conditioning information, and this harness
measures 59.69, a degradation in the expected direction and of a plausible size.
So 184.96 is a real property of this model rather than an artefact of the
evaluation code.

The gap between 38.89 in domain and 184.96 on WikiText-2 is 4.8x, which is what
heavy domain specialization on 1.23B tokens looks like.

### On the perplexity figure this project used to report

An earlier version of my CV reported **WikiText-2 perplexity 31.23**. That number
was `exp(val_loss)` computed on the in-distribution validation split *during
training*; it never touched WikiText-2. The real figure is 184.96.

31.23 was also never reachable at this scale, being below GPT-2-small's own
published score. A 134M model trained on 1.23B tokens does not beat GPT-2-small,
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

9 passed. Each maps to a property that was wrong at some point:

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

## Not measured

Stated explicitly so that an absence reads as a known gap rather than an
oversight.

| Metric | Status | What it needs |
|---|---|---|
| **Training throughput (tokens/sec)** | **Never measured on GPU.** An early CV draft claimed 75K tok/s with no instrumented run behind it | one `train.py` run on a named GPU; it logs `tokens_per_sec` per eval and `avg_tokens_per_sec` in `summary.json` |
| **Peak GPU memory** | Not measured | the same run; `train.py` logs `peak_gpu_mem_gb` |
| **Learning-rate sweep** | Not run | three runs at a fixed token budget, varying only `--lr` |
| **Batch-size sweep** | Not run | three runs holding `batch_size x grad_accum` constant |
| **RoPE / RMSNorm / SwiGLU / GQA** | Not implemented | deliberately absent from this baseline |
| **Exact GPU model** | Uncertain | a free-tier 16 GB notebook GPU. No run log survives to name it and the notebook artefacts are all Colab, so this says "free-tier 16 GB" rather than guessing a model number |

---

## Reproduction environment

The perplexity and generation numbers above were produced on CPU
(torch 2.12.0+cpu, Python 3.12, Windows). WikiText-2 at context 128 over the full
test set takes about 15 minutes per model on CPU and is near instant on a GPU.
Perplexity is deterministic, so a GPU run reproduces these figures up to
floating-point reduction-order noise. `generate.py` is seeded (`--seed 123`).
