# GPT-134M: pretraining a transformer from scratch, and reading the circuits inside it

[![tests](https://github.com/umer-ateeq/GPT-From-Scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/umer-ateeq/GPT-From-Scratch/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)

I wrote a 134M-parameter decoder-only transformer in PyTorch with no transformer
library, built the streaming data pipeline that feeds it, and pretrained it on
FineWeb-Edu using a single free-tier Kaggle P100. Attention, the causal mask,
layer normalization, the sampler and the training loop are all written out
directly.

Then I did the two things that make a checkpoint worth keeping.

**I established what the finished model actually is, from its weights.** Not from
a config file, from the tensors: the trained context length, the number of
optimizer steps it survived, and the number of tokens it consumed, each recovered
from a different part of the parameter space and each agreeing with the others.

**I looked for the mechanisms.** Two of the model's 96 attention heads form an
**induction circuit**, the structure Anthropic's interpretability work identifies
behind in-context learning, driven by a previous-token head one layer below.
Ablating those two heads removes **85% of the model's ability to copy from
context**, against a size-matched null of random head pairs at 30.6 standard
deviations.

This repository is those things in order: what I built, what it does, how I
verified it, and what turned out to be inside it.

---

## 1. What I built

| | |
|---|---|
| Type | decoder-only transformer, GPT-2 style |
| Parameters | **134,077,440** trainable |
| Layers | 8 |
| Attention heads | 12, head dimension 64 |
| Model width | 768 |
| Feed-forward | 3072 hidden, 4x expansion, ReLU |
| Normalization | custom LayerNorm, learned scale and shift, **pre-norm** |
| Positional encoding | learned absolute embeddings |
| Vocabulary | 50257, GPT-2 BPE via `tiktoken` |
| Output head | 768 x 50257, no bias, **untied** from the input embedding |
| Dropout | 0.1 on embeddings, attention weights and residual branches |
| Positional rows allocated | 256 |
| **Trained context** | **128** (established in section 3) |

Every row was recovered from the checkpoint's own tensor shapes rather than
copied from a config file, and `tests/test_model.py` asserts the parameter count
exactly, so the headline number cannot drift from the code.

Component-by-component reasoning, including the parameter budget and why each
piece is there, is in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**. The model
is **[model.py](model.py)**, about 130 lines.

Three choices worth calling out:

**Pre-norm residuals.** Normalization runs before each sublayer rather than
after, so the residual path from input to output is never normalized. That is
what keeps gradients well behaved through 8 stacked blocks; the original
post-norm transformer needs careful warmup to train at all.

**Untied output head.** The input embedding and the output projection are
separate 50257 x 768 matrices. Tying them, as GPT-2 does, would save 38.6M
parameters and put this model near 96M. Keeping them separate lets the model
represent a token differently as an input than as a prediction target.

**Learned absolute positions.** Attention is permutation-invariant on its own,
so position has to be injected explicitly. One learned vector per position is
simple and is what GPT-2 did. It also leaves a per-position record in the
weights, which section 3 reads.

### The data pipeline

FineWeb-Edu does not fit on a free notebook's disk, so it is **streamed** rather
than downloaded, tokenized in chunks, and written straight into a pre-allocated
memory-mapped `uint16` file. Peak RAM stays at roughly one chunk regardless of
corpus size.

`uint16` is deliberate: GPT-2's vocabulary tops out at 50256, which fits in 16
bits, so storing IDs as `int64` would have turned a 16 GB file into 64 GB for
nothing.

That produced an **8B-token corpus** from `CC-MAIN-2024-10`. Validation comes
from `CC-MAIN-2024-18`, a **different Common Crawl snapshot**, so held-out data
is distribution-matched but genuinely disjoint.

```bash
python tokenize_data.py --out train.bin      --max-tokens 8e9 --dump CC-MAIN-2024-10
python tokenize_data.py --out validation.bin --max-tokens 5e6 --dump CC-MAIN-2024-18
```

### The training configuration

Everything the released checkpoint was trained with, in one place:

| | |
|---|---|
| Hardware | one NVIDIA Tesla P100, 16 GB, free-tier Kaggle session |
| Corpus | FineWeb-Edu `CC-MAIN-2024-10`, 8B tokens, uint16 memmap |
| Held-out set | FineWeb-Edu `CC-MAIN-2024-18`, disjoint snapshot |
| Optimizer | AdamW |
| Learning rate | 4e-4, **constant** |
| Weight decay | 0.1 |
| Gradient clipping | global norm 1.0 |
| Precision | fp16 mixed precision with `GradScaler` |
| Batch shape | 32 sequences x 128 tokens = **4,096 tokens per optimizer step** |
| Successful optimizer steps | **~234,000** (measured from the weights, section 3) |
| Tokens consumed | **~1.0-1.2B**, about 15% of the corpus |
| Tokens per parameter | ~9, roughly half the Chinchilla-optimal budget |
| Throughput | 10,200 tokens/sec, 31.7% MFU |
| Peak GPU memory | 6.12 GB of 16 GB |

**Corpus size and tokens consumed are different numbers, and conflating them
overstates a pretraining run by several multiples.** 8B is how much text the
tokenizer wrote to disk. ~1B is how much passed through the model, sampled as
random windows from it. This repository reports both separately, everywhere.

### The decisions the hardware forced

Three of them exist because of one constraint, 16 GB of P100:

**Memory-mapped batching.** The corpus is a flat array on disk. `np.memmap` pages
in only the windows actually sampled, so training reads from an 8B-token file
without loading it. Batches are random windows, and the memmap is reopened on
each call, because holding one open across thousands of iterations leaks the
pages it touches.

**Mixed precision.** fp16 halves activation memory and uses the P100's full-rate
fp16 path. A `GradScaler` multiplies the loss before backward so small gradients
do not underflow to zero, then unscales before the optimizer step. The unscaling
has to happen *before* gradient clipping, or the clip normalizes gradients that
are still in the scaled domain; `train.py` calls `scaler.unscale_(optimizer)`
first for exactly that reason.

**Pinned asynchronous transfers.** Host-to-device copies overlap with compute
rather than blocking on it.

```bash
python train.py --train-bin train.bin --val-bin validation.bin \
    --train-tokens 1e9 --lr 4e-4 --batch-size 32 --run-name baseline
```

### Every run leaves a record

`train.py` writes its full configuration, git commit, seed, library versions, GPU
name, launch command, per-eval metrics, throughput, MFU, peak memory and loss
curve to `runs/`, and checkpoints embed their own config. Examples are committed
there.

This is the part most from-scratch training repos skip, and it is the part that
makes every number below checkable by someone who is not me.

## 2. What it does

| Metric | Value |
|---|---|
| Held-out perplexity, FineWeb-Edu `CC-MAIN-2024-18` | **38.89** at context 128 |
| TinyStories perplexity | 35.41 |
| WikiText-2 perplexity | **184.96**, against **59.69** for GPT-2-small |
| Throughput | **10,200 tokens/sec**, **31.7% MFU** at 32 x 128, fp16 |
| Peak GPU memory | **6.12 GB** of 16 GB |

### The baseline is the point

A perplexity number without a reference measured the same way is not a claim
anyone can check. So `evaluate.py --model gpt2` pushes HuggingFace's GPT-2-small
through the **identical scoring function, tokenizer, test set and window
settings**. The only thing that differs between those two rows is the model.

This model's WikiText-2 perplexity is **3.10x higher** than GPT-2-small's, and
perplexity is lower-is-better, so GPT-2-small wins that comparison by 3.10x.
That is what ~9 tokens per parameter predicts: GPT-2-small saw roughly 8B tokens of WebText
against this model's ~1B of filtered educational web text, and WikiText-2 is
encyclopedic prose far from FineWeb-Edu. A 134M model at half the
compute-optimal budget does not beat a model trained on 8x the data, and a
pipeline that reports it doing so has a measurement problem rather than a result.

**The comparison also validates the harness.** GPT-2-small's published ~29.4 at
context 1024 degrading to 59.69 at context 128 is the right direction and the
right magnitude, so 184.96 is a property of this model rather than of my
evaluation code.

The 4.8x gap between 38.89 in domain and 184.96 on WikiText-2 is what heavy
domain specialization on ~1B tokens looks like.

### MFU, computed honestly

Throughput only becomes comparable across hardware once converted to a fraction
of the GPU's peak:

```
N_matmul        = 134,077,440 - 38,597,376 (tok_emb) - 196,608 (pos_emb) = 95,283,456
FLOPs per token = 6 x 95,283,456 + 12(8)(128)(768) = 581,137,920
achieved        = 581,137,920 x 10,200 = 5.93 TFLOP/s
P100 fp16 peak  = 18.7 TFLOP/s
MFU             = 31.7%
```

**The token embedding is excluded from N on purpose.** Reading a row of `tok_emb`
is a gather, not a matrix multiply, so it costs no FLOPs. nanoGPT counts its
embedding table only because its output head is *tied* to it: one parameter
block, doing the work once. This model's head is untied, so `out_head` is a
genuine 768 x 50257 matmul that belongs in N while `tok_emb` is a separate table
of identical size that does not. Counting both would inflate the figure by about
40% relative.

No nanoGPT comparison is offered, because their published MFU excludes embeddings
under a tied head and the P100's compute-to-bandwidth ratio is roughly 8x lower
than an A100's, which makes a high MFU *easier* to reach on this card. Quoting an
A100 number beside a P100 number would flatter this one in two directions at
once.

[SAMPLES.md](SAMPLES.md) has six unedited completions from one seeded run, with
the prompts fixed in source so they could not be chosen after the fact. They are
locally fluent and factually unreliable, which is what 184.96 predicts.

Full detail, both evaluation protocols, and what remains weakly evidenced:
**[docs/RESULTS.md](docs/RESULTS.md)**.

## 3. Reading the training run out of the weights

A checkpoint's configuration file is a claim. Its weights are evidence. This
checkpoint arrived as a bare `state_dict`, 117 tensors with no config, no
optimizer state and no metadata, so before reporting a single number from it I
reconstructed the run it came from using nothing but the tensors.

```bash
python audit_checkpoint.py --ckpt weights8b_300epoch.pth
```

### The positional table records the trained context

`pos_emb.weight` holds one row per position. A row only receives gradient if some
batch was long enough to reach it, and AdamW applies weight decay to every
parameter on every step whether or not a gradient arrives. So an untrained
position decays geometrically toward zero while a trained one holds its norm,
and the boundary between them is legible directly in the weights:

```
       positions   mean norm       min       max
            0-31      0.2132    0.1696    0.5629
           32-63      0.1886    0.1811    0.2046
           64-95      0.2196    0.1984    0.2394
          96-127      0.2487    0.2364    0.2800
         128-159      0.0024    0.0022    0.0025
         160-191      0.0024    0.0022    0.0025
         192-223      0.0024    0.0022    0.0025
         224-255      0.0023    0.0022    0.0024
```

![Positional embedding row norms, a 93x cliff exactly at position 128](docs/images/pos_emb_norms.png)

A **93x cliff exactly at position 128**. The run trained at context 128 across
256 allocated rows, and that is why every perplexity number in this repository is
quoted at 128. Scoring at 256 puts half of every window on rows that never
received gradient and roughly doubles perplexity, 38.89 against 74.70 on
identical data, for reasons that have nothing to do with model quality.
`evaluate.py` detects this from the weights and warns before printing.

### The same rows count the optimizer steps

Those dead rows are a clock. They received decay and no gradient, so their total
shrinkage integrates the learning rate across the checkpoint's entire life:

```
final = initial x (1 - lr x weight_decay)^N

0.002346 / 27.7765 = 8.448e-5
ln(8.448e-5) / ln(1 - 4e-4 x 0.1) = 234,478 successful optimizer steps
```

At the run's real 4,096 tokens per step, that is **~0.96B tokens**.

### Three independent routes, one answer

| Route | Reads | Gives |
|---|---|---|
| Weight-decay clock on `pos_emb` rows 128-255 | the weights | 234,478 steps, 0.96B tokens |
| 517 never-sampled `tok_emb` rows at the same decay floor | a **disjoint** parameter subspace | 227,000 to 236,600 steps |
| The notebook's loop counts, 300 cycles x 1000 batches | the source | 300,000 attempted steps, 1.23B tokens |

The weight-based routes bracket each other, and both sit **22% below** the
attempted-step count. That residual is exactly what `GradScaler` produces: a step
skipped on fp16 overflow applies neither the Adam update nor the decay, so the
clock counts *successful* steps and is a lower bound by construction.

Three routes, two of them touching no source code at all, converge on **~1B
tokens**, which is also what the model's measured quality predicts.

### The controls that make the clock valid

The method assumes those rows are untouched initialization that was only scaled,
rather than rows that trained and then decayed. Two properties distinguish those
cases, and both check out:

| | Norm spread (std/mean) | Adjacent-row cosine |
|---|---|---|
| Fresh initialization | 2.75% | 0.026 |
| **Checkpoint rows 128-255** | **2.72%** | **0.028** |
| Checkpoint rows 0-127 (trained) | 23.10% | 0.822 |

The dead rows kept their random directions and their uniform norms. They were
scaled, not trained. `tests/test_model.py` also plants a synthetic context cliff
in a fresh model and asserts the audit recovers it, so the detector is tested
against a known answer.

Full derivation, every control, and what the method cannot establish:
**[docs/AUDIT.md](docs/AUDIT.md)**.

## 4. What is inside it: an induction circuit

Having built the model and established what it is, the question I actually wanted
to answer: **are the structures described in the interpretability literature
present in a model I trained myself, at 134M parameters, on a free GPU?**

An induction head implements one rule: *"I have seen this token before. What came
next last time? Attend to that."* It is the leading mechanistic account of
in-context learning (Olsson et al., 2022).

The probe feeds the model a random token sequence repeated twice and measures
where each of the 96 heads looks. **Random tokens are the point**: the model
cannot fall back on memorized English, so any copying has to come from the
context. Scores are averaged over 16 sequences and reported against a uniform
baseline of 0.0142, the attention an indifferent causal head would put on any one
position.

![Induction score by head](docs/images/induction_heads.png)

| Head | Attention on the induction target | std | vs uniform |
|---|---|---|---|
| **L6.H9** | 0.4188 | 0.0397 | **29.5x** |
| **L7.H8** | 0.2738 | 0.0341 | **19.3x** |
| L6.H7 | 0.0864 | | 6.1x |
| everything else | ~0.014 | | ~1x |

`L6.H9` puts **42% of its total attention mass on one position** out of fifty
plus. That is a head doing one job.

### It is induction, not duplicate detection

A head attending to the *same* earlier token would be a duplicate-token head,
which notices repetition without predicting anything. L6.H9 puts **0.4188 on the
next token against 0.0136 on the same token, a factor of 31**.

It is also not positional counting. Elhage et al. note that induction can be
implemented by pointer arithmetic over positions rather than by matching on
content, and this model has learned absolute positional embeddings, so the
alternative is live. Varying the repeat period across 32, 48 and 56 moves the
score by only **8%**, which a fixed-offset mechanism could not survive.

### The heads are causally necessary

| Intervention | 2nd-copy loss change | 95% CI | Copying destroyed |
|---|---|---|---|
| **Mean ablation** (field standard) | **+3.3473** | [+3.2306, +3.4697] | **85.4%** |
| Zero ablation | +3.9089 | [+3.7656, +4.0438] | 99.7% |

**Ablating 2 of 96 heads removes 85.4% of the model's repeated-sequence copying**,
against a size-matched null of random head *pairs* at +0.0624 ± 0.1074:
**30.6 standard deviations**. First-copy loss barely moves, so this is targeted
loss of a capability rather than general damage.

Mean ablation is the headline because zeroing pushes the residual stream
off-distribution and overstates the effect (Zhang and Nanda, 2024). The
denominator is corrected too: positions 48-95 have more context than 0-47 whether
or not anything repeats, and on non-repeated sequences that is worth **+0.51
nats**, so the real copying benefit is 3.92 rather than the naive 4.43.

### And it is a circuit, not two correlated heads

An induction head cannot work alone. To find "the position after the previous B",
something must first tag each position with what preceded it. That is a
**previous-token head**, and it has to run earlier, because with standard
attention the tag must be written before it can be matched.

`L5.H11` does it, at 6.2x uniform, in layer 5, immediately below the induction
heads in layers 6 and 7. Ablating it costs 1.25 nats, 28% of copying.

The test that earns the word "circuit" is not the loss, it is the **attention
pattern**: ablate L5.H11 and re-measure what the induction heads look at.

| Head | Induction score, intact | With L5.H11 ablated | Fall |
|---|---|---|---|
| L6.H9 | 0.4188 | 0.2504 | **40.2%** |
| L7.H8 | 0.2738 | 0.1671 | **39.0%** |

The upstream head is writing the tag the downstream heads match on. That is
**K-composition**, and it is what makes this a mechanism with parts rather than
two interesting heads.

**The asymmetry is the interesting part.** The previous-token role is redundant,
with L2.H2, L2.H10 and L3.H0 all doing it partially, so removing L5.H11 leaves
weaker copies behind. The induction role is not redundant. Two heads carry it,
and removing both removes the capability.

Method, every control, limitations and 13 references:
**[docs/INDUCTION_HEADS.md](docs/INDUCTION_HEADS.md)**.

```bash
python induction_heads.py       --ckpt weights8b_300epoch.pth   # which heads
python ablate_heads.py          --ckpt weights8b_300epoch.pth   # do they matter
python previous_token_heads.py  --ckpt weights8b_300epoch.pth   # the upstream head
python circuit_controls.py      --ckpt weights8b_300epoch.pth   # the three controls
```

## 5. Where this goes

The two halves of this repository do the same thing: ask the weights a question
instead of trusting the configuration. That is the through-line, and it is what
the next round of work extends.

### The claim this checkpoint supports

> **Induction-circuit formation is robust to a constrained training setup.** At
> context 128, with a constant learning rate and roughly half the
> compute-optimal token budget, a clean and causally necessary induction circuit
> still formed and still carried almost all of the model's copying ability.

**What would falsify it:** train the same architecture at context 32, or stop at
100M tokens, and find no induction heads.

**What is already known:** Olsson et al. found induction heads across every
architecture and scale they examined, so the optimizer-axis version of this is
probably not surprising. Chan et al. (2022) found the opposite along a different
axis: in-context learning fails to emerge when burstiness and Zipfian marginals
are removed from the data. So the open version of the question is about the
**data distribution**, not the optimizer.

**Why it matters:** if a capability like in-context learning emerges reliably
even from a heavily constrained run, you cannot suppress it by training
carelessly. You have to be able to detect it afterwards. That is the argument for
interpretability, and it is why this repository spends as much effort looking
inside the model as building it.

### The interpretability work queued next

Induction heads are the entry point, not the destination. On this checkpoint and
its successors:

- **Activation patching** to localize behaviour causally rather than by ablation
  alone, which measures necessity but not the path.
- **The IOI circuit** (Wang et al., 2022) as the next canonical target, since it
  requires composing several head classes rather than one.
- **Logit lens and direct logit attribution**, to read what each head writes into
  the residual stream instead of only what it attends to.
- **Formation dynamics.** Olsson et al. showed induction heads appear abruptly,
  in a narrow band of training. Seeing *when* this circuit formed needs
  checkpoints saved during training, which the released run does not have.
  `train.py` now saves them, so the next run answers it.

## Verify any of it

```bash
pip install -r requirements.txt
```

**Needs nothing but this repository:**

```bash
python -m pytest tests/ -v                                        # 23 tests
python evaluate.py --model gpt2 --mode wikitext --max-length 128  # the GPT-2 baseline
```

The 23 tests cover the analysis code, not just the model: that captured attention
exactly reproduces what `MultiHeadAttention` computes, that an untrained model
shows no induction (so the probe is not measuring itself), that the ablation hook
zeroes one head and no other, and that the audit recovers a planted context
cliff. A bug in analysis code produces a confident, plausible, wrong claim, which
is much harder to notice than a loss that will not go down.

**Needs the checkpoint.** It is 538 MB, past GitHub's 100 MB file limit, and is
**not published yet**. `upload_to_hf.py` will put it on the Hugging Face Hub;
until that runs, the commands in sections 3 and 4 cannot be reproduced by anyone
but me, and this README says so rather than linking a page with no file behind
it.

Every analysis command runs on CPU in under two minutes. No GPU is needed to
check any claim here.

## The code

```
model.py             LayerNorm, FeedForward, MultiHeadAttention, TransformerBlock,
                     GPTModel. Written out, no transformer library
config.py            every hyperparameter in one place
data.py              memory-mapped batch sampling from a uint16 token file
tokenize_data.py     stream a HuggingFace dataset into that token file
train.py             mixed precision, gradient accumulation, warmup + cosine decay,
                     clipping, resumable, MFU logging, full run records
generate.py          greedy and temperature/top-k sampling
evaluate.py          perplexity, two protocols, GPT-2-small as an in-harness baseline

audit_checkpoint.py     recover architecture and trained context from raw weights
induction_heads.py      probe every head for induction behaviour
ablate_heads.py         zero or mean ablation, size-matched controls, bootstrap CIs
previous_token_heads.py find the other half of the circuit
circuit_controls.py     K-composition, varied repeat period, positional baseline

tests/               23 tests: 9 on the model and training loop, 14 on the analysis
runs/                per-run logs, config, metrics, loss curves
notebooks/           the original Colab training notebook, unedited
```

[docs/CODE_MAP.md](docs/CODE_MAP.md) says which code is verbatim from the
original training notebook, which was restructured and why, and which was written
afterwards.

## Scope

- **Not an assistant.** No instruction tuning, no RLHF, no safety tuning. It
  completes text and nothing else.
- **Not competitive.** GPT-2-small beats it by 3.1x on WikiText-2 perplexity,
  which is the expected result at ~9 tokens per parameter and is reported here
  with the baseline measured on the identical harness.
- **Not a training trajectory.** The interpretability results come from one final
  checkpoint. When the circuit formed is the obvious next question, and this run
  cannot answer it.
- **Not fully reproducible by a stranger yet**, until the checkpoint is
  published.

## Documentation

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | every component, why it is there, parameter budget |
| [docs/RESULTS.md](docs/RESULTS.md) | every measurement with the command that produces it |
| [docs/AUDIT.md](docs/AUDIT.md) | recovering the training run from the weights, with controls |
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
