# I pretrained a 134M-parameter LLM from zero. Now I am doing interpretability on it.

Every component is written from scratch in pytorch. Multi-head attention, the causal mask, LayerNorm, the
feed-forward block, the sampler, the training loop: written from zero in
**[pretrain/model.py](pretrain/model.py)**, 130 lines. No `nn.Transformer`, no
HuggingFace model class, no `x-transformers`. PyTorch tensors and nothing above
them.

Pretraining it was the first half. The second half is interpreting it. A trained
transformer is a black box: a loss curve tells you nothing about what it learned
to *do*. So I started measuring the artifact instead of reading its config, and
found that two of its 96 attention heads form an induction circuit carrying 85%
of this model's ability to copy from its context.

| | |
|---|---|
| Parameters | **134,077,440** |
| Training tokens | **1.2B**, FineWeb-Edu |
| Hardware | a single Tesla P100 16 GB, free Kaggle session |
| Held-out perplexity | **38.89**, against a Chinchilla prediction of **38.70** |
| Throughput | **10,200 tokens/sec** |
| Induction circuit | 2 of 96 heads, **85.4%** of copying, **30.6σ** over null |

### Contents

1. [The model](#1-the-model)
2. [Training](#2-training)
3. [Results](#3-results)
4. [Reading the training run out of the weights](#4-reading-the-training-run-out-of-the-weights)
5. [Reading the circuit out of the weights](#5-reading-the-circuit-out-of-the-weights)
6. [What is next](#6-what-is-next)
7. [Run it](#7-run-it)

---

## 1. The model

| | | | |
|---|---|---|---|
| Layers | 8 | Width | 768 |
| Heads | 12 x dim 64 | Feed-forward | 3072, ReLU |
| Vocabulary | 50257, GPT-2 BPE | Normalization | pre-norm LayerNorm |
| Positions | learned absolute | Output head | **untied**, 768 x 50257 |
| Context allocated | 256 | **Context trained** | **128** |

| Component | Parameters |
|---|---|
| Token embedding, 50257 x 768 | 38,597,376 |
| Positional embedding, 256 x 768 | 196,608 |
| 8 transformer blocks | 56,682,240 |
| Final LayerNorm | 1,536 |
| Output head, 768 x 50257 | 38,597,376 |
| **Total** | **134,077,440** |

The output head is untied from the input embedding. Tying it, as GPT-2 does, would
drop 38.6M parameters and make this a 96M model.

## 2. Training

FineWeb-Edu is **streamed** from Hugging Face, tokenized as it arrives, and written
to a memory-mapped `uint16` array. The dataset never lands on disk in full and peak
RAM stays at one chunk, which is what makes this run on a free notebook. `uint16`
because GPT-2's vocabulary stops at 50256, so `int64` would quadruple the file for
nothing. Training data is `CC-MAIN-2024-10`, validation `CC-MAIN-2024-18`, a
different Common Crawl snapshot.

| | | | |
|---|---|---|---|
| Optimizer | AdamW | Learning rate | 4e-4, constant |
| Weight decay | 0.1 | Grad clip | global norm 1.0 |
| Precision | fp16 + `GradScaler` | Batch | 32 x 128 = 4,096 tok/step |
| Optimizer steps | **~234,000** | Tokens | **~1.2B** |
| Throughput | **10,200 tok/s** | Peak GPU memory | **12.24 GB** of 16 |

Three techniques exist because of the 16 GB ceiling:

- **Memory-mapped batches.** `np.memmap` pages in only the sampled windows, so
  training reads a multi-gigabyte token file without loading it. The memmap is
  reopened per call, because holding one open leaks the pages it touches.
- **Mixed precision.** fp16 halves activation memory. `GradScaler` scales the loss
  so small gradients do not underflow, then unscales **before** clipping, or the
  clip normalizes values that are still scaled.
- **Pinned async transfers**, so host-to-device copies overlap with compute.

```bash
python pretrain/tokenize_data.py --out train.bin --dump CC-MAIN-2024-10 --max-tokens 2e9
python pretrain/train.py --train-bin train.bin --val-bin validation.bin \
    --train-tokens 1e9 --lr 4e-4 --batch-size 32 --run-name baseline
```

Every run writes its config, git commit, seed, library versions, GPU name, per-eval
metrics, throughput, peak memory and loss curve to `runs/`.

## 3. Results

| Dataset | Perplexity | Context |
|---|---|---|
| **Held-out FineWeb-Edu** `CC-MAIN-2024-18` | **38.89** | 128 |
| TinyStories | 35.41 | 128 |
| WikiText-2 | 184.96 | 128 |


### GPT-2-small on the identical harness

| Model | Params | Perplexity |
|---|---|---|---|
| This model | 134.08M | **184.96** |
| GPT-2-small | 124.44M | **59.69** |

`pretrain/evaluate.py --model gpt2` runs GPT-2-small through the same scoring
function, tokenizer, test set and window. Only the model differs. Both are scored on
the full WikiText-2 raw test set, 285,396 tokens, context 128, non-overlapping
windows, so no token is counted twice.

GPT-2-small wins by 3.10x. It saw ~7x the tokens, on WebText, and WikiText-2 is
encyclopedic prose far from FineWeb-Edu. The comparison also **validates the
harness**: GPT-2-small's published ~29.4 at context 1024 falls to 59.69 at context
128, the right direction and the right magnitude. So 184.96 is a property of this
model, not of my evaluation code.

## 4. Reading the training run out of the weights

The checkpoint is a bare `state_dict`: 117 tensors, no config, no metadata,
nothing but numbers. Every entry in the training table above is recoverable from
those tensors. A config file records what a run intended. The weights record what
it did, and where the two disagree the weights are the artifact you ship.

```bash
python interp/audit_checkpoint.py --ckpt weights8b_300epoch.pth
```

**The trained context, from the positional table.** A row of `pos_emb.weight` only
receives gradient if a batch reached that position, but AdamW decays every
parameter on every step regardless. So untrained rows shrink geometrically while
trained rows hold their norm, and the boundary is legible directly:

```
   positions  0-31   32-63   64-95  96-127 | 128-159 160-191 192-223 224-255
   mean norm 0.2132  0.1886  0.2196  0.2487 |  0.0024  0.0024  0.0024  0.0023
```

![Positional embedding row norms, a 93x cliff exactly at position 128](images/pos_emb_norms.png)

A **93x cliff exactly at position 128** across 256 allocated rows. That is why every
perplexity here is quoted at 128: scoring at 256 puts half of each window on rows
that never trained and roughly doubles perplexity, 38.89 to 74.70 on identical data.
`evaluate.py` reads this from the weights and warns before printing a number.

**The optimizer step count, from the same rows.** Those dead rows are a clock. They
took decay and no gradient, so their total shrinkage integrates the learning rate
across the run, and because the rate was constant that integral is a count.
Initialization is `nn.Embedding` default N(0,1), so a fresh 768-dimensional row has
expected norm `sqrt(768) = 27.71`:

```
final = initial x (1 - lr x weight_decay)^N

0.002346 / 27.7765 = 8.448e-5
ln(8.448e-5) / ln(1 - 4e-4 x 0.1) = 234,478 optimizer steps
```

**Three routes to the same answer, two of which never read the source code:**

| Route | Reads | Result |
|---|---|---|
| Weight-decay clock on `pos_emb` rows 128-255 | the weights | 234,478 steps |
| 517 never-sampled `tok_emb` rows at the same floor | a **disjoint** parameter subspace | 227,000 to 236,600 steps |
| Training loop counts, 300 cycles x 1000 batches | the source | 300,000 steps, 1.23B tokens |

The two weight-based routes bracket each other to within a few thousand steps,
measured in parameter subspaces that share no parameters. Both sit 22% below the
loop count, which is the `GradScaler` skip rate: a step skipped on fp16 overflow
runs forward and backward but applies no update and no decay. So the clock counts
steps that changed the weights and is a lower bound on steps attempted.

**The controls.** The method assumes those rows are untouched initialization that
was only scaled down, not rows that trained and then decayed. Two properties
separate those cases, and both check out:

| | Norm spread (std/mean) | Adjacent-row cosine |
|---|---|---|
| Fresh initialization | 2.75% | 0.026 |
| **Checkpoint rows 128-255** | **2.72%** | **0.028** |
| Checkpoint rows 0-127 (trained) | 23.10% | 0.822 |

The dead rows kept their random directions and their even norms. They were scaled,
not trained. The 2.75% figure is also what theory predicts: the relative standard
deviation of the norm of a 768-dimensional Gaussian vector is `1/sqrt(2 x 768) =
2.6%`. `tests/test_interpretability.py` plants a synthetic cliff in a fresh model
and asserts the audit recovers it, so the detector is tested against a known answer.

Full method, the failure modes, and what the clock cannot establish:
**[interp/CHECKPOINT_AUDIT.md](interp/CHECKPOINT_AUDIT.md)**.

## 5. Reading the circuit out of the weights

Same method, harder question. Section 4 asked what the weights say about the *run*.
This asks what they say about the *computation*.

An induction head implements one rule: *"I have seen this token before. What came
next last time? Attend to that."* It is the leading mechanistic account of
in-context learning (Olsson et al., 2022). The question was whether a 134M model
trained on a free GPU at context 128 forms one at all.

**Technique 1: attention probing.** Feed the model a random sequence repeated twice
and measure where each of the 96 heads looks. **Random tokens are the point**: the
model cannot fall back on memorized English, so any copying comes from the context.
16 sequences. A causal head at position `i` spreads attention over `i + 1` positions,
so an indifferent head puts `1/(i+1)` on any single one, which averages to the
**uniform baseline of 0.0142** across the measured queries. A head scoring 1x is
doing nothing.

![Induction score by head](images/induction_heads.png)

| Head | Attention on the induction target | vs uniform |
|---|---|---|
| **L6.H9** | **0.4188** | **29.5x** |
| **L7.H8** | **0.2738** | **19.3x** |
| L6.H7 | 0.0864 | 6.1x |
| everything else | ~0.014 | ~1x |

`L6.H9` puts **42% of its attention on a single position** out of fifty plus. Three
things it could have been instead, and is not:

- **A duplicate-token head**, which notices repetition without predicting. L6.H9 puts
  0.4188 on the *next* token against 0.0136 on the *same* token, a factor of **31**.
- **A fixed positional offset.** This model has learned absolute positions, so a head
  keyed to a constant offset is expressible and would score identically at a fixed
  repeat period. Varying the period across 32, 48 and 56 moves the score by **8%**,
  where a fixed-offset head would collapse everywhere except one value.
- **An artifact of the probe.** The same probe on an untrained model returns ~1x
  uniform across all 96 heads, which `tests/test_interpretability.py` asserts. The
  probe is not measuring itself.

**Technique 2: causal ablation.** Attention is correlational. A head can look at
exactly the right place and contribute nothing to the output. So overwrite the
head's slice of the attention output through a forward hook, before `out_proj` mixes
the heads, and re-measure.

| Intervention | 2nd-copy loss | 1st-copy loss | 95% CI | Copying destroyed |
|---|---|---|---|---|
| **Mean ablation** | **+3.3473** | +0.28 | [+3.2306, +3.4697] | **85.4%** |
| Zero ablation | +3.9089 | +0.28 | [+3.7656, +4.0438] | 99.7% |

Ablating 2 of 96 heads destroys **85.4% of the model's repeated-sequence copying**.
The null is size-matched, random head *pairs* rather than single heads, drawn from
heads with no induction role: **+0.0624 ± 0.1074**, worst control **+0.2431**. The
low end of the treatment's confidence interval clears the worst control by more than
13x, which is **30.6 standard deviations** above the null mean. First-copy loss moves
only +0.28, so this is targeted loss of one capability rather than general damage
from perturbing the network.

Two corrections that move the headline number, both applied:

- **Mean over zero ablation.** Zeroing removes the head's mean contribution as well
  as its input-dependent signal, pushing the residual stream off-distribution and
  overstating the effect (Zhang and Nanda, 2024). The claim survives the stricter
  test; the figure moves from 99.7% to 85.4%.
- **A corrected denominator.** Later positions have more context whether or not
  anything repeats. Measured on non-repeated sequences that positional component is
  worth **+0.51 nats**, so the true copying benefit is 3.92 rather than the naive
  4.43.

**Technique 3: composition analysis.** An induction head cannot work alone, and the
reason is the interesting part. To predict what follows the second `B`, a head must
attend to the position after the *first* `B`. But that position holds `C`, and
nothing about `C` says "I follow a B", so the search fails. Something has to tag each
position with its predecessor first. That is a **previous-token head**, and with
standard attention it has to run in an earlier layer, because the tag must be written
before it can be matched.

So the mechanism predicts a previous-token head below layer 6. `L5.H11` is one, at
6.2x uniform, in layer 5, directly beneath the induction heads in layers 6 and 7.

The test that earns the word "circuit" is not the loss, it is the attention pattern.
If L5.H11 merely correlates with the induction heads, ablating it leaves their
attention unmoved. Ablate it, then re-measure where they look:

| Head | Induction score, intact | With L5.H11 ablated | Fall |
|---|---|---|---|
| L6.H9 | 0.4188 | 0.2504 | **40.2%** |
| L7.H8 | 0.2738 | 0.1671 | **39.0%** |

The upstream head is writing the tag the downstream heads match on. That is
**K-composition**: a mechanism with parts and an order they must run in.

The pattern does not fall all the way to the 0.0142 baseline, and that is consistent
with the rest of the picture. The previous-token role is **redundant**: L2.H2, L2.H10
and L3.H0 all show partial previous-token behaviour, so removing L5.H11 leaves weaker
copies of the same signal behind. The induction role is not redundant. Only two heads
of 96 carry it, which is why removing both removes the capability.

**Scope.** The 85.4% is copying on repeated random tokens, a narrower quantity than
in-context learning on natural text (Crosbie and Shutova, 2024). It comes from one
final checkpoint, so it establishes that the circuit exists and says nothing about
when it formed. Full method, every control, the size-matched null, confidence
intervals, limitations and 13 references:
**[interp/INDUCTION_HEADS.md](interp/INDUCTION_HEADS.md)**.

```bash
python interp/induction_heads.py       --ckpt weights8b_300epoch.pth
python interp/ablate_heads.py          --ckpt weights8b_300epoch.pth --ablation mean
python interp/previous_token_heads.py  --ckpt weights8b_300epoch.pth
python interp/circuit_controls.py      --ckpt weights8b_300epoch.pth
```

## 6. What is next

The induction circuit is a mechanism I can see from the outside. The question I
want to work on is whether a model can report on its own internals, and whether
those reports are faithful. Chain-of-thought and self-report are the cheapest
monitoring interface we have, and they are known to be unfaithful: models deny
using a hint that ablation shows they used (Turpin et al., 2023).

The hard part of that problem is not the verbalization. It is the **ground
truth**: you need to know what was actually in the activations before you can
score a report about them, and that ground truth comes from probe readouts,
feature activations, and the measured effect of ablation and patching. That is
the side I have been building. `ablate_heads.py` already reads and overwrites
internal activations through forward hooks, with a size-matched null and
bootstrap confidence intervals, on a model whose entire training history I can
account for.

The ladder from here, each rung a technique I want to be able to run cold:

- **Linear probes on the residual stream.** Start with a target this model
  demonstrably computes: `L5.H11` writes a previous-token tag, so a probe should
  recover the previous token from the layer-5 residual stream and fail at layer 0.
  A circuit I have already localized causally gives the probe a known answer to
  be checked against, which is the same discipline as the planted-cliff test in
  `tests/`.
- **Activation patching**, to move from necessity to path. Ablation shows a head
  matters; patching shows which downstream components its output actually reaches.
- **Sparse autoencoders** on the residual stream and MLP activations. At 134M with
  8 layers this is trainable on the same free GPU, and having an SAE turns
  "which head" into "which feature", which is the resolution introspection work
  needs.
- **Faithfulness of self-report.** With probes and SAE features in place, the
  internals supply labels for free. First in-context, with no training: given a
  feature that a probe says is active, how much can a model already say about it?
  Then as fine-tuning, using probe readouts and ablation effects as the
  supervision signal, and testing whether it generalizes to unseen inputs and
  unseen features rather than memorizing the probe.

This model cannot answer questions about itself. It is a base model with no
instruction tuning, so the verbalization half needs a chat model. What it is good
for is the measurement half: a transformer small enough to probe end to end on a
CPU, fully specified down to the optimizer step, and mine, so there is no gap
between what I can measure and what I know about how it was made.

Two experiments on this checkpoint that I still want to run:

- **Formation dynamics.** Induction heads appear abruptly during training. Seeing
  when this one formed needs mid-run checkpoints, which `train.py` now saves.
- **The data-distribution experiment.** Chan et al. (2022) showed in-context
  learning fails to emerge when burstiness and Zipfian marginals are stripped from
  the training data. The tokenizer, the streaming pipeline and the training loop
  are all in this repository, so that ablation is a 33-hour run away.

## 7. Run it

```bash
pip install -r requirements.txt

python -m pytest tests/ -v                       # 23 tests, no checkpoint needed
python verify.py --ckpt weights8b_300epoch.pth   # perplexity vs GPT-2-small
```
Weights are on the Hugging Face Hub: **[HUGGING FACE URL]** (538 MB, past GitHub's
file limit). Every analysis command above runs on CPU in under two minutes.

```
pretrain/
  model.py          LayerNorm, FeedForward, MultiHeadAttention, TransformerBlock, GPTModel
  config.py         every hyperparameter in one place
  data.py           memory-mapped batch sampling from a uint16 token file
  tokenize_data.py  stream a HuggingFace dataset into that token file
  train.py          fp16, grad accumulation, warmup + cosine, clipping, resumable, run records
  generate.py       greedy and temperature/top-k sampling
  evaluate.py       perplexity, two protocols, GPT-2-small baseline
interp/
  audit_checkpoint.py      recover architecture and trained context from weights
  induction_heads.py       probe all 96 heads for induction behaviour
  ablate_heads.py          zero and mean ablation, size-matched null, bootstrap CIs
  previous_token_heads.py  the upstream half of the circuit
  circuit_controls.py      K-composition, repeat period, positional baseline
  INDUCTION_HEADS.md       the circuit: method, controls, limitations, 13 references
  CHECKPOINT_AUDIT.md      the run: the weight-decay clock, its controls and its limits

verify.py  one command to check the perplexity claims
```

### Scope

- **Not an assistant.** No instruction tuning, no RLHF. It completes text.
- **Not competitive.** GPT-2-small beats it 3.1x on WikiText-2, which is what ~9
  tokens per parameter buys.
- **Not a trajectory.** The circuit results come from one final checkpoint.

## Credits

Model implementation follows Sebastian Raschka,
[Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch).
Memory-mapped sampler and parts of the training loop follow Andrej Karpathy,
[nanoGPT](https://github.com/karpathy/nanoGPT). Data:
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).