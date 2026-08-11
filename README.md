# GPT-134M: what the weights say the training run actually did

[![tests](https://github.com/umer-ateeq/GPT-From-Scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/umer-ateeq/GPT-From-Scratch/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)

I pretrained a 134M-parameter GPT from scratch in PyTorch on a single free
Kaggle P100, with no transformer library: attention, the causal mask, layer
normalization and the training loop all written out directly.

Then I stopped trusting it, and took it apart. Two things came out of that.

**The configuration was lying.** The config said context 256 and 4.9B tokens. The
weights say context **128**. **Six** separate bugs, each proven from the
checkpoint or the notebook source. Two of them were found by an adversarial
review of this repository, after it was written, and one invalidated a conclusion
I had already published here. Both are recorded in full rather than quietly
corrected.

**The model built an induction circuit anyway.** Two of its 96 attention heads
implement the in-context copying mechanism from Anthropic's induction-heads work,
fed by a previous-token head one layer below. Ablating those two heads destroys
**85.4% of the model's repeated-sequence copying**, while leaving everything else
intact, and ablating the upstream head degrades the downstream heads' attention
pattern by 40%, which is what makes it a circuit rather than two correlated
heads.

Those two findings sit oddly together, and that is the interesting part: a run
this broken, at roughly half its compute-optimal token budget with half its
context untrained and no working learning-rate schedule, still formed a clean,
causally necessary induction circuit.

**Almost every number here was produced by a script in this repo, with the
command next to it.** The exceptions are named where they occur: the weight-decay
clock is hand-computed from `audit_checkpoint.py` output, and the throughput
figures come from a Kaggle session whose full logs were not retrieved. Where
something cannot be pinned down, this repo says so rather than picking a
flattering point.

---

## 1. The configuration was lying

`pos_emb.weight` holds one row per position. A row only receives gradient if some
training batch was long enough to reach it, and AdamW's weight decay shrinks
whatever it touches. So an untrained position decays toward zero and leaves a
permanent mark.

![Positional embedding row norms, a 93x cliff exactly at position 128](docs/images/pos_emb_norms.png)

A **93x cliff exactly at position 128**, on a log scale. The run had been training
at context 128 while every summary reported 256, because a notebook cell
eighteen cells after `get_batch` was defined rebound the globals it read.

```bash
python audit_checkpoint.py --ckpt weights8b_300epoch.pth
```

Three more bugs turned up in the same pass, including a warmup-plus-cosine
schedule bound to an optimizer the training cell had already replaced, so it never
touched the weights at all. A later adversarial review of this repo found two
more, one of which invalidated a published conclusion. All six, with the proof for
each and the code change that now prevents it: **[docs/AUDIT.md](docs/AUDIT.md)**.

### Reading the token count off the weights, two ways

The untrained rows are also a clock. AdamW touches every parameter twice per
step: a multiplicative weight decay, and a gradient update. Rows 128-255 received
no gradient, so **only the decay ever applied to them**, and their total shrinkage
integrates the learning rate across the checkpoint's whole life:

```
final = initial x (1 - lr x weight_decay)^N

27.7765  ->  0.002346          shrinkage 8.448e-5, measured
sum of learning rates = 93.8   (weight_decay = 0.1)
at the flat 4e-4 that ran      ->  ~234,000 optimizer steps
```

Now the second route. Bug 5 in the audit is that gradient accumulation never ran:
`train_model` defaults `gradient_accumulation_steps=1` and the training cell never
passes the argument, so the `32` set in the hyperparameter cell was inert. **Every
micro-batch was an optimizer step.** 300 cycles x 1000 batches = **300,000**
attempted steps.

| Method | Optimizer steps | Tokens at 32 x 128 per step |
|---|---|---|
| Weight-decay clock, from the weights | 234,478 | 0.96B |
| Loop counts, from the notebook | 300,000 | 1.23B |

**The two agree to within 22%**, and the gap has an obvious cause: `GradScaler`
skips `optimizer.step()` whenever fp16 gradients overflow, which is common early
in training, and a skipped step applies no weight decay either. So the clock
measures *successful* steps and is a lower bound on attempted ones.

Two independent methods, one reading the weights and one reading the source,
converging on **roughly 1B tokens**. That also matches the model's quality: at
about 9 tokens per parameter, under half the Chinchilla budget, losing 3.1x to
GPT-2-small is the expected outcome rather than a puzzle.

**An earlier version of this README got this wrong** and claimed the clock implied
25x more training than the filename, presenting the two as an unresolved conflict.
That came from multiplying by an accumulation factor of 32 that never executed. The
error, and the correction, are in [docs/AUDIT.md](docs/AUDIT.md) as bug 5. Full
working in [docs/RESULTS.md](docs/RESULTS.md).

## 2. The model built an induction circuit anyway

An induction head implements one rule: *"I have seen this token before. What came
next last time? Attend to that."* It is believed to be the main mechanism behind
in-context learning.

Feed the model a random token sequence repeated twice and measure where each of
the 96 heads looks. Random tokens matter: the model cannot fall back on memorized
English, so any copying has to come from the context.

![Induction score by head](docs/images/induction_heads.png)

| Head | Attention on the induction target | std over 16 sequences | vs uniform |
|---|---|---|---|
| **L6.H9** | 0.4188 | 0.0397 | **29.5x** |
| **L7.H8** | 0.2738 | 0.0341 | **19.3x** |
| everything else | ~0.014 | | ~1x |

`L6.H9` puts 42% of its attention mass on a single position out of fifty-plus, and
the effect is about 10x its own standard deviation across sequences.

**It is induction, not duplicate detection.** A head attending to the *same*
token's earlier occurrence would be a duplicate-token head, which notices
repetition without predicting anything. Measured on the same sequences, L6.H9
puts 0.4188 on the next token against **0.0136** on the same token, a factor of
**31**.

### The heads are causally necessary, not decorative

Attention patterns are correlational. So both heads were ablated, by zeroing
their slice of the attention output, and the model re-measured:

| | Loss, 1st copy | Loss, 2nd copy | In-context benefit |
|---|---|---|---|
| Intact | 12.7327 | 8.3056 | **4.4271 nats** |
| Both heads ablated | 13.0137 (+0.28) | 12.2145 (**+3.91**) | 0.5182 |

| Intervention | 2nd-copy loss change | 95% CI | Copying destroyed |
|---|---|---|---|
| **Mean ablation** (field standard) | **+3.3473** | [+3.2306, +3.4697] | **85.4%** |
| Zero ablation | +3.9089 | [+3.7656, +4.0438] | 99.7% |

**Ablating 2 of 96 heads destroys 85.4% of the model's repeated-sequence
copying**, against a size-matched null of random head *pairs* at +0.0624 ± 0.1074:
**30.6 standard deviations**. First-copy loss barely moves, so it is not general
damage. Mean ablation is quoted rather than zero ablation because zeroing pushes
the residual stream off-distribution and overstates (Zhang and Nanda, 2024).

Two corrections a review forced, both of which made the result sharper:

- **The denominator needed a positional baseline.** Positions 48-95 have more
  context than 0-47 whether or not anything repeats. On a *non*-repeated random
  sequence that is worth **+0.51 nats**, so the true copying benefit is 3.92, not
  4.43, and the ablation destroys 99.7% of it rather than 88.3%. The residual was
  the artefact.
- **It is copying, not "in-context learning".** Olsson et al. define the ICL score
  on natural text; this measures verbatim copying of random tokens. Corrected
  everywhere.
- **Zero ablation overstated it.** Under mean ablation, the field-standard
  intervention, the figure is 85.4% rather than 99.7%. The conclusion survives;
  the number moves.

### And it is a circuit, with parts in the right order

An induction head cannot work alone. To find "the position after the previous
`B`", something must first tag each position with what preceded it. That is a
**previous-token head**, and it has to run in an earlier layer.

| Head | Attention on position i-1 | vs uniform |
|---|---|---|
| **L5.H11** | 0.2420 | **6.2x** |

Layer 5, immediately below the induction heads in layers 6 and 7. The ordering the
mechanism requires is exactly what the model has.

Ablating it costs **1.25 nats, 28% of in-context learning**, while first-copy loss
*improves* slightly (-0.07). So every part of the circuit is causally load-bearing:

| Ablated | In-context learning lost |
|---|---|
| L5.H11 (previous-token) | 28.3% |
| L6.H9 + L7.H8 (induction) | 88.3% |

**The asymmetry is the interesting bit.** The previous-token role is redundant:
L2.H2, L2.H10, L2.H6 and L3.H0 all show partial previous-token behaviour, so
removing L5.H11 leaves weaker copies of the signal behind. The induction role is
not redundant. Two heads do it, and removing both removes the capability.

Method, controls, and limitations, including the ones a reviewer would raise:
**[docs/INDUCTION_HEADS.md](docs/INDUCTION_HEADS.md)**.

```bash
python induction_heads.py       --ckpt weights8b_300epoch.pth   # which heads
python ablate_heads.py          --ckpt weights8b_300epoch.pth   # do they matter
python previous_token_heads.py  --ckpt weights8b_300epoch.pth   # the other half
```

---

## The model itself

| | |
|---|---|
| Parameters | **134,077,440** trainable |
| Architecture | 8 layers, 12 heads, 768 wide, GPT-2 BPE (50257), learned positions, untied head |
| Corpus built | **8B tokens**, FineWeb-Edu, streamed and tokenized to a uint16 memmap |
| Tokens consumed | **~1.0-1.2B**, agreed by two independent methods (see above) |
| Held-out perplexity | **38.89** at context 128, on a disjoint Common Crawl snapshot |
| WikiText-2 perplexity | **184.96**, against **59.69** for GPT-2-small on the identical harness |
| Hardware | one NVIDIA Tesla P100, 16 GB, free Kaggle session |
| Learning rate | 4e-4, fixed, no schedule (see the audit) |
| Throughput | **10,200 tokens/sec**, **31.7% MFU** at batch 32 x 128, fp16 |
| Peak GPU memory | **6.1 GB** of 16 GB |

**This model loses to GPT-2-small by 3.1x on WikiText-2.** Stated up front, because
a perplexity number without a reference measured the same way is not a claim
anyone can check. `evaluate.py --model gpt2` runs GPT-2-small through the
identical scoring function; the only thing that differs between those two rows is
the model. That comparison also validates the harness: GPT-2-small's published
~29.4 at context 1024 degrading to 59.69 at context 128 is the right direction
and magnitude.

Full detail, both protocols, and what remains unmeasured:
**[docs/RESULTS.md](docs/RESULTS.md)**.

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
attention exactly reproduces what `MultiHeadAttention` computes, that an
untrained model shows no induction (so the probe is not measuring itself), that
the ablation hook zeroes one head and no other, and that the audit recovers a
planted context cliff.

**Needs the checkpoint.** It is 538 MB, past GitHub's 100 MB file limit, and
**is not published yet**. `upload_to_hf.py` will put it on the Hugging Face Hub;
until that runs, the four commands below cannot be reproduced by anyone but me,
and this README says so rather than linking a page that has no file on it:

```bash
python audit_checkpoint.py      --ckpt weights8b_300epoch.pth   # the context bug
python induction_heads.py       --ckpt weights8b_300epoch.pth   # the circuit
python ablate_heads.py          --ckpt weights8b_300epoch.pth   # the causal test
python previous_token_heads.py  --ckpt weights8b_300epoch.pth   # the upstream head
```

Every one of these runs on CPU in under two minutes. No GPU required to check any
claim in this README.

Example run logs, including the full artifact set every training run produces,
are committed under [runs/](runs/).

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
ablate_heads.py         zero a head and measure what the model loses, with controls
previous_token_heads.py find the other half of the circuit

tests/               23 tests: 9 on the model and training loop, 14 on the analysis code
notebooks/           the original Colab notebook, unedited, cited by the audit
```

[docs/CODE_MAP.md](docs/CODE_MAP.md) says which code is verbatim from the original
training notebook, which was restructured and why, and which was added afterwards.

## How it was trained

Three techniques, all present because of the 16 GB limit:

**Gradient accumulation.** 32 micro-batches of 32 sequences accumulate before one
optimizer step, giving the gradient quality of a 1024-sequence batch at the memory
cost of 32. Measured peak memory is 6.1 GB of 16; without accumulation the same
effective batch would need roughly 32x the activation memory and would not fit.

**Mixed precision (fp16).** Halves activation memory. A gradient scaler multiplies
the loss before backward so small gradients do not underflow, then unscales before
the optimizer step.

**Memory-mapped data.** The corpus is a flat uint16 array on disk. `np.memmap`
pages in only the windows actually sampled, so training reads from an 8B-token
file without loading it.

```bash
python tokenize_data.py --out train.bin      --max-tokens 1e9 --dump CC-MAIN-2024-10
python tokenize_data.py --out validation.bin --max-tokens 5e6 --dump CC-MAIN-2024-18

python train.py --train-bin train.bin --val-bin validation.bin \
    --train-tokens 1e9 --lr 4e-4 --batch-size 32 --run-name baseline
```

Validation comes from a **different Common Crawl snapshot** than training, so it
is distribution-matched but genuinely disjoint.

## What this is not

- **Not an assistant.** No instruction tuning, no RLHF, no safety tuning.
- **Not competitive.** It loses to GPT-2-small, a 2019 model, by 3.1x.
- **Not a training trajectory.** The interpretability results are from one final
  checkpoint. Showing *when* the induction circuit formed would need checkpoints
  saved during training, which this run never saved. That is the honest gap.

[SAMPLES.md](SAMPLES.md) has six unedited completions, chosen before I saw the
output. They loop and get facts wrong, which is what 184.96 out-of-domain
perplexity predicts.

## Documentation

| Document | Contents |
|---|---|
| [docs/AUDIT.md](docs/AUDIT.md) | six bugs, the proof for each, and what prevents them now |
| [docs/INDUCTION_HEADS.md](docs/INDUCTION_HEADS.md) | the circuit: method, controls, limitations |
| [docs/RESULTS.md](docs/RESULTS.md) | every measurement with the command that produces it |
| [docs/CODE_MAP.md](docs/CODE_MAP.md) | what each file is, and which code is verbatim |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | every component, why it is there, parameter budget |
| [docs/CHANGES_FROM_NOTEBOOK.md](docs/CHANGES_FROM_NOTEBOOK.md) | line-level diff from the original notebook |
| [docs/MEASURE.md](docs/MEASURE.md) | reproducing the throughput and memory numbers on a P100 |

## What this suggests, stated as a hypothesis

The two halves of this repo sit oddly together. The training run was degraded in
four separate ways, and the model still built a clean, causally necessary
induction circuit. Put as a claim that could be wrong:

> **Induction-circuit formation is robust to substantial degradation of the
> training setup.** Halving the context, removing the learning-rate schedule,
> training at a flat rate, and sampling random windows with replacement did not
> prevent the circuit from forming or from carrying almost all of the model's
> in-context learning.

**What would falsify it:** train the same architecture at context 32, or stop at
100M tokens, and find no induction heads. If the circuit disappears under milder
degradation than this run suffered, then it is not robust and this was luck.

**Why it would matter:** if a capability like in-context learning emerges reliably
even from badly configured training, you cannot suppress it by training carelessly.
You have to be able to detect it after the fact. That is the argument for
interpretability, and it is the reason this repo spends as much effort looking
inside the model as it does training it.

This experiment is **not run here**. The hypothesis is stated so that it can be
attacked.

## Attribution

The model implementation follows Sebastian Raschka's
[Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch).
The memory-mapped batch sampler and parts of the training loop follow Andrej
Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). Training data is
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
Perplexity uses the strided-window protocol from the GPT-2 paper. The induction
head analysis follows Olsson et al.,
[In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).

MIT licensed, see [LICENSE](LICENSE).
