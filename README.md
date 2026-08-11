# GPT-134M: a transformer built from scratch, and a look inside it

[![tests](https://github.com/umer-ateeq/GPT-From-Scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/umer-ateeq/GPT-From-Scratch/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)

I wrote a 134M-parameter decoder transformer in PyTorch with no transformer
library, built the data pipeline for it, and pretrained it on FineWeb-Edu using a
single free Kaggle P100. Attention, the causal mask, layer normalization and the
training loop are all written out directly.

Then I wanted to know two things about the thing I had made.

**Did it train the way I thought it did?** It did not. Auditing the checkpoint
against its own configuration turned up six bugs, one of which had been quietly
halving the context length for the entire run.

**Are the structures I had been reading about actually in there?** Yes. Two of
its 96 attention heads form an induction circuit, the mechanism Anthropic's
interpretability work identifies behind in-context learning, fed by a
previous-token head one layer below. Ablating them removes 85% of the model's
ability to copy from context.

This repository is those three things in order: how it was built, whether it did
what I thought, and what turned out to be inside it.

---

## 1. What I built

| | |
|---|---|
| Type | decoder-only transformer, GPT-2 style |
| Parameters | **134,077,440** trainable |
| Layers / heads / width | 8 / 12 / 768, head dimension 64 |
| Feed-forward | 4x expansion, ReLU |
| Normalization | custom LayerNorm, learned scale and shift, **pre-norm** |
| Positional encoding | learned absolute embeddings |
| Vocabulary | 50257, GPT-2 BPE via `tiktoken` |
| Output head | **untied** from the input embedding |
| Context | 256 allocated, **128 actually trained** (see section 2) |

Every row was recovered from the checkpoint's tensor shapes rather than copied
from a config, and `tests/test_model.py` asserts the parameter count exactly.
Component-by-component reasoning is in
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**; the code is
**[model.py](model.py)**, about 130 lines.

### The data pipeline

FineWeb-Edu does not fit on a free notebook's disk, so it is **streamed** rather
than downloaded, tokenized in chunks, and written straight into a pre-allocated
memory-mapped `uint16` file. Peak RAM stays at roughly one chunk regardless of
corpus size. `uint16` is deliberate: GPT-2's vocabulary tops out at 50256, which
fits in 16 bits, so storing IDs as `int64` would have made a 16 GB file 64 GB for
nothing.

That produced an **8B-token** corpus from `CC-MAIN-2024-10`. Validation comes from
`CC-MAIN-2024-18`, a **different Common Crawl snapshot**, so it is
distribution-matched but genuinely disjoint.

```bash
python tokenize_data.py --out train.bin      --max-tokens 8e9 --dump CC-MAIN-2024-10
python tokenize_data.py --out validation.bin --max-tokens 5e6 --dump CC-MAIN-2024-18
```

### The training decisions, and why

Three of them exist because of one constraint, 16 GB of P100:

**Memory-mapped batching.** The corpus is a flat array on disk. `np.memmap` pages
in only the windows actually sampled, so training reads from an 8B-token file
without loading it. Batches are random windows, and the memmap is reopened each
call, because holding one across thousands of iterations leaks the pages it
touches.

**Mixed precision (fp16).** Halves activation memory and uses the P100's
full-rate fp16 path. A `GradScaler` multiplies the loss before backward so small
gradients do not underflow to zero, then unscales before the optimizer step. The
unscaling has to happen *before* gradient clipping, or the clip normalizes
scaled gradients, which is bug 6 below.

**Pinned asynchronous transfers.** Host-to-device copies overlap with compute
rather than blocking on it.

**Gradient accumulation** is implemented in `train.py` and available via
`--grad-accum`, but be aware it did **not** run for the released checkpoint. That
is bug 5. Activation memory is set by the micro-batch either way, which is why the
run still fit in 6.12 GB.

```bash
python train.py --train-bin train.bin --val-bin validation.bin \
    --train-tokens 1e9 --lr 4e-4 --batch-size 32 --run-name baseline
```

Every run writes its full configuration, git commit, seed, per-eval metrics,
throughput, MFU, peak memory and loss curve to `runs/`. Examples are committed
there. That infrastructure exists because the original run recorded none of it,
which is how the bugs in section 2 survived.

## 2. What it actually does

| Metric | Value |
|---|---|
| Tokens consumed | **~1.0-1.2B**, about 15% of the corpus, agreed by two independent methods |
| Tokens per parameter | ~9, roughly half the Chinchilla-optimal budget |
| Held-out perplexity | **38.89** at context 128, on the disjoint snapshot |
| TinyStories perplexity | 35.41 |
| WikiText-2 perplexity | **184.96**, against **59.69** for GPT-2-small |
| Throughput | **10,200 tokens/sec**, **31.7% MFU** at batch 32 x 128, fp16 |
| Peak GPU memory | **6.12 GB** of 16 GB |

**This model loses to GPT-2-small by 3.1x on WikiText-2**, and that is the honest
headline. A perplexity number without a reference measured the same way is not a
claim anyone can check, so `evaluate.py --model gpt2` runs GPT-2-small through the
identical scoring function, tokenizer, test set and window. The only thing that
differs between those two rows is the model.

The comparison also validates the harness: GPT-2-small's published ~29.4 at
context 1024 degrading to 59.69 at context 128 is the right direction and
magnitude, so 184.96 is a property of this model rather than of my evaluation
code. Losing is the expected outcome at ~9 tokens per parameter.

[SAMPLES.md](SAMPLES.md) has six unedited completions from one seeded run, with
the prompts fixed in source so they could not be chosen after the fact. They are
locally fluent and factually wrong, which is what 184.96 predicts.

Full detail, both protocols, and what remains weakly evidenced:
**[docs/RESULTS.md](docs/RESULTS.md)**.

## 3. Did it train the way I thought?

No. The configuration and the weights disagreed, and the weights were right.

`pos_emb.weight` holds one row per position. A row only receives gradient if some
batch was long enough to reach it, and AdamW's weight decay shrinks whatever it
touches, so an untrained position decays toward zero and leaves a permanent mark.

![Positional embedding row norms, a 93x cliff exactly at position 128](docs/images/pos_emb_norms.png)

A **93x cliff exactly at position 128**. The run had been training at context 128
while every summary said 256, because a cell eighteen cells after `get_batch` was
defined rebound the globals it read.

**Six bugs in total**, each proven from the checkpoint or the notebook source:

| | Bug | Effect |
|---|---|---|
| 1 | Batch-shape globals rebound after `get_batch` closed over them | Context silently halved to 128 |
| 2 | LR scheduler bound to an optimizer the training cell then replaced | No warmup, no decay, flat 4e-4 |
| 3 | Cosine floor set 5x **above** the peak | Inert, only because of bug 2 |
| 4 | No run recorded a config, seed, or metric | Bugs 1-3 invisible for months |
| 5 | `gradient_accumulation_steps` never passed to `train_model` | Accumulation never ran |
| 6 | `clip_grad_norm_` applied to **scaled** gradients | Effective step size suppressed |

**Bugs 5 and 6 were found by an adversarial review of this repository, after it
was written, and bug 5 invalidated a conclusion I had already published here.**
Both are recorded in full rather than quietly corrected, because a repo whose
argument is "check your claims against the artifact" has to survive that being
done to it.

### The weights can also count the optimizer steps

The untrained rows are a clock. They received decay and no gradient, so their
total shrinkage integrates the learning rate across the checkpoint's whole life:

```
final = initial x (1 - lr x weight_decay)^N
27.7765 -> 0.002346   shrinkage 8.448e-5,  sum of lr = 93.8,  N ~ 234,000 steps
```

Independently, the notebook's loop counts give 300 cycles x 1000 batches =
**300,000** attempted steps. The two agree to within 22%, and the gap is
`GradScaler` skipping fp16 overflow steps, which apply no decay either. A third
clock, 517 never-sampled `tok_emb` rows, gives 227,000-236,600.

Three routes, **~1B tokens**, matching the model's quality. Full derivation and
the controls that validate it: **[docs/AUDIT.md](docs/AUDIT.md)**.

## 4. What is inside it

Having built the model and established what it really is, the question I actually
wanted to answer: **are the structures described in the interpretability
literature present in a model I trained myself, on a free GPU, badly?**

An induction head implements one rule: *"I have seen this token before. What came
next last time? Attend to that."* It is the leading mechanistic account of
in-context learning (Olsson et al., 2022). Feed the model a random token sequence
repeated twice and measure where each of the 96 heads looks. Random tokens matter:
the model cannot fall back on memorized English, so any copying comes from context.

![Induction score by head](docs/images/induction_heads.png)

| Head | Attention on the induction target | std | vs uniform |
|---|---|---|---|
| **L6.H9** | 0.4188 | 0.0397 | **29.5x** |
| **L7.H8** | 0.2738 | 0.0341 | **19.3x** |
| everything else | ~0.014 | | ~1x |

`L6.H9` puts 42% of its attention mass on one position out of fifty-plus.

**It is induction, not duplicate detection.** A head attending to the *same*
earlier token would be a duplicate-token head, which notices repetition without
predicting. L6.H9 puts 0.4188 on the next token against 0.0136 on the same token,
a factor of **31**. Varying the repeat period across 32/48/56 moves the score only
**8%**, so it is matching on content rather than counting to a fixed offset.

### The heads are causally necessary

| Intervention | 2nd-copy loss change | 95% CI | Copying destroyed |
|---|---|---|---|
| **Mean ablation** (field standard) | **+3.3473** | [+3.2306, +3.4697] | **85.4%** |
| Zero ablation | +3.9089 | [+3.7656, +4.0438] | 99.7% |

**Ablating 2 of 96 heads removes 85.4% of the model's repeated-sequence copying**,
against a size-matched null of random head *pairs* at +0.0624 ± 0.1074:
**30.6 standard deviations**. First-copy loss barely moves, so this is not general
damage. Mean ablation is quoted because zeroing pushes the residual stream
off-distribution and overstates (Zhang and Nanda, 2024).

The denominator is corrected too: positions 48-95 have more context than 0-47
whether or not anything repeats, and on non-repeated sequences that is worth
**+0.51 nats**, so the real copying benefit is 3.92 rather than 4.43.

### And it is a circuit, not two correlated heads

An induction head cannot work alone. To find "the position after the previous B",
something must first tag each position with what preceded it. That is a
**previous-token head**, and it must run earlier.

`L5.H11` does it, at 6.2x uniform, in layer 5, immediately below. Ablating it
costs 1.25 nats, 28% of copying, while first-copy loss *improves*.

The test that earns the word "circuit": ablate L5.H11 and re-measure the induction
heads' **attention pattern**, not the loss.

| Head | Induction score, intact | With L5.H11 ablated | Fall |
|---|---|---|---|
| L6.H9 | 0.4188 | 0.2504 | **40.2%** |
| L7.H8 | 0.2738 | 0.1671 | **39.0%** |

The upstream head is writing the tag the downstream heads match on. That is
K-composition, and it is what makes this a mechanism with parts rather than two
interesting heads.

**The asymmetry is the interesting part.** The previous-token role is redundant
(L2.H2, L2.H10, L3.H0 all do it partially), so removing L5.H11 leaves weaker
copies behind. The induction role is not. Two heads do it, and removing both
removes the capability.

Method, controls, limitations, and 13 references:
**[docs/INDUCTION_HEADS.md](docs/INDUCTION_HEADS.md)**.

```bash
python induction_heads.py       --ckpt weights8b_300epoch.pth   # which heads
python ablate_heads.py          --ckpt weights8b_300epoch.pth   # do they matter
python previous_token_heads.py  --ckpt weights8b_300epoch.pth   # the upstream head
python circuit_controls.py      --ckpt weights8b_300epoch.pth   # the three controls
```

## 5. Where this goes

The two halves sit oddly together, and that is the part worth pursuing. The
training run was degraded in four separate ways and the model still built a clean,
causally necessary induction circuit. As a claim that could be wrong:

> **Induction-circuit formation is robust to substantial degradation of the
> training setup.** Halving the context, removing the learning-rate schedule, and
> training at roughly half the compute-optimal budget did not prevent the circuit
> from forming or from carrying almost all of the model's copying ability.

**What would falsify it:** train the same architecture at context 32, or stop at
100M tokens, and find no induction heads.

**What is already known:** Olsson et al. found induction heads across every
architecture and scale they examined, so the optimizer-axis version of this is
probably not surprising. Chan et al. (2022) found the opposite along a different
axis: in-context learning fails to emerge when burstiness and Zipfian marginals
are removed from the data. So the open version of the question is about the
**data distribution**, not the optimizer.

**Why it matters:** if a capability like in-context learning emerges reliably even
from badly configured training, you cannot suppress it by training carelessly. You
have to be able to detect it afterwards. That is the argument for interpretability,
and it is why this repo spends as much effort looking inside the model as building
it.

**The experiment I would run next** is the one this checkpoint cannot answer:
Olsson et al. showed induction heads appear abruptly, in a narrow band of
training. Seeing *when* the circuit formed needs checkpoints saved during
training, which this run never saved. `train.py` now saves them.

## Verify any of it

```bash
pip install -r requirements.txt
```

**Needs nothing but this repo:**

```bash
python -m pytest tests/ -v                                        # 23 tests
python evaluate.py --model gpt2 --mode wikitext --max-length 128  # the GPT-2 baseline
```

The 23 tests include the analysis code, not just the model: that the captured
attention exactly reproduces what `MultiHeadAttention` computes, that an untrained
model shows no induction (so the probe is not measuring itself), that the ablation
hook zeroes one head and no other, and that the audit recovers a planted context
cliff. A bug in the analysis produces a confident, plausible, wrong claim, which is
much harder to notice than a loss that will not go down.

**Needs the checkpoint.** It is 538 MB, past GitHub's 100 MB file limit, and **is
not published yet**. `upload_to_hf.py` will put it on the Hugging Face Hub; until
that runs, the commands in sections 3 and 4 cannot be reproduced by anyone but me,
and this README says so rather than linking a page with no file behind it.

Every analysis command runs on CPU in under two minutes. No GPU is needed to check
any claim here.

## The code

```
model.py             LayerNorm, FeedForward, MultiHeadAttention, TransformerBlock,
                     GPTModel. Written out, no transformer library
config.py            every hyperparameter in one place, which is the single change
                     that would have prevented the worst bug
data.py              memory-mapped batch sampling from a uint16 token file
tokenize_data.py     stream a HuggingFace dataset into that token file
train.py             mixed precision, gradient accumulation, warmup + cosine decay,
                     clipping, resumable, MFU logging, full run records
generate.py          greedy and temperature/top-k sampling
evaluate.py          perplexity, two protocols, GPT-2-small as an in-harness baseline

audit_checkpoint.py     recover architecture and true training context from raw weights
induction_heads.py      probe every head for induction behaviour
ablate_heads.py         zero or mean ablation, size-matched controls, bootstrap CIs
previous_token_heads.py find the other half of the circuit
circuit_controls.py     K-composition, varied repeat period, positional baseline

tests/               23 tests: 9 on the model and training loop, 14 on the analysis
runs/                per-run logs; the provenance the original run never kept
notebooks/           the original Colab notebook, unedited, cited by the audit
```

[docs/CODE_MAP.md](docs/CODE_MAP.md) says which code is verbatim from the original
training notebook, which was restructured and why, and which was added afterwards.

## What this is not

- **Not an assistant.** No instruction tuning, no RLHF, no safety tuning.
- **Not competitive.** It loses to GPT-2-small, a 2019 model, by 3.1x.
- **Not a training trajectory.** The interpretability results come from one final
  checkpoint. When the circuit formed is the obvious next question and this run
  cannot answer it.
- **Not fully reproducible by a stranger yet**, until the checkpoint is published.

## Documentation

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | every component, why it is there, parameter budget |
| [docs/RESULTS.md](docs/RESULTS.md) | every measurement with the command that produces it |
| [docs/AUDIT.md](docs/AUDIT.md) | six bugs, the proof for each, and what prevents them now |
| [docs/INDUCTION_HEADS.md](docs/INDUCTION_HEADS.md) | the circuit: method, controls, limitations, references |
| [docs/CODE_MAP.md](docs/CODE_MAP.md) | what each file is, and which code is verbatim |
| [docs/CHANGES_FROM_NOTEBOOK.md](docs/CHANGES_FROM_NOTEBOOK.md) | line-level diff from the original notebook |
| [docs/MEASURE.md](docs/MEASURE.md) | reproducing throughput and memory on a P100 |

## Attribution

The model implementation follows Sebastian Raschka's
[Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch).
The memory-mapped batch sampler and parts of the training loop follow Andrej
Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). Training data is
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). The
interpretability analysis follows Elhage et al.,
[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
and Olsson et al.,
[In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).

MIT licensed, see [LICENSE](LICENSE).
