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


### GPT-2-small Comparison:

| Model | Params | Perplexity |
|---|---|---|
| This model | 134.08M | **184.96** |
| GPT-2-small | 124.44M | **59.69** |

`pretrain/evaluate.py --model gpt2` runs GPT-2-small through the same scoring
functionn, tokenizer, test set and window. Only the model differs. Both are scored on
the full WikiText-2 raw test set, 285,396 tokens, context 128, non-overlapping
windows, so no token is counted twice.

GPT-2-small wins by 3.10x. It saw ~7x the tokens, on WebText, and WikiText-2 is
encyclopedic prose far from FineWeb-Edu. The comparison also **validates the
harness**: GPT-2-small's published ~29.4 at context 1024 falls to 59.69 at context
128, the right direction and the right magnitude. So 184.96 is a property of this
model, not of my evaluation code.

## 4. Reading the training run out of the weights

The checkpoint is a bare `state_dict`: 117 tensors, no config, no metadata. Every
number in the training table above came back out of those tensors.

```bash
python interp/audit_checkpoint.py --ckpt weights8b_300epoch.pth
```

The model has 256 positional rows. The run used 128. So half of `pos_emb.weight` got
weight decay on every step and gradient on none. Those rows are a recording of the
run, and two numbers come out of them.

### The trained context is 128

A positional row gets a gradient only if a batch reaches that position. AdamW decays
it either way. So rows the run never reached shrink toward zero, and rows it reached
hold their norm.

```
   positions  0-31   32-63   64-95  96-127 | 128-159 160-191 192-223 224-255
   mean norm 0.2132  0.1886  0.2196  0.2487 |  0.0024  0.0024  0.0024  0.0023
```

![Positional embedding row norms, a 93x cliff exactly at position 128](images/pos_emb_norms.png)

A **93x cliff, exactly at position 128**. This sets the evaluation context. The same
data scored at 256 gives 74.70 instead of 38.89, and all of that gap is measurement
error, so `evaluate.py` reads the cliff off the weights and warns before printing.

### The run took 234,478 optimizer steps

Those 128 rows were multiplied by the same decay factor on every step that landed.
Multiply it out and the shrinkage counts the steps. `nn.Embedding` starts from
N(0,1), so a fresh row of width 768 has norm `sqrt(768) = 27.71`:

```
final = initial x (1 - lr x weight_decay)^N

0.002346 / 27.7765 = 8.448e-5
ln(8.448e-5) / ln(1 - 4e-4 x 0.1) = 234,478 steps
```

Three routes agree, and two never read the source code:

| Route | Reads | Result |
|---|---|---|
| Decay clock on `pos_emb` rows 128-255 | the weights | 234,478 steps |
| 517 never-sampled `tok_emb` rows at the same floor | a **different** part of the weights | 227,000 to 236,600 steps |
| Training loop counts, 300 cycles x 1000 batches | the source | 300,000 steps, 1.23B tokens |

The two weight-based routes share no parameters and agree to within a few thousand
steps out of 234,000. Both come in 22% under the loop count, which is the
`GradScaler` skip rate: a step skipped on fp16 overflow still runs forward and
backward, but applies no update and no decay. The clock counts steps that changed
the weights.

### The controls

This only works if those rows are untouched initialization that was scaled down, not
rows that trained and then decayed. Training pulls neighbouring rows into alignment
and spreads their norms. Scaling does neither.

| | Norm spread (std/mean) | Adjacent-row cosine |
|---|---|---|
| Fresh initialization | 2.75% | 0.026 |
| **Checkpoint rows 128-255** | **2.72%** | **0.028** |
| Checkpoint rows 0-127 (trained) | 23.10% | 0.822 |

Both numbers match fresh initialization and are nowhere near the trained rows. 2.75%
is also what theory gives: for a random vector of width 768 the norm varies by
`1/sqrt(2 x 768) = 2.6%`. `tests/` plants a fake cliff in a fresh model and checks
that the audit finds it, so the detector has been tested against a known answer.

Full method and limits: **[interp/CHECKPOINT_AUDIT.md](interp/CHECKPOINT_AUDIT.md)**.

## 5. Reading the circuit out of the weights

Same method, harder question. Section 4 asked what the weights say about the *run*.
This asks what they say about the *computation*.

An induction head follows one rule: *"I have seen this token before. What came next
last time? Attend to that."* It is the leading account of in-context learning
(Olsson et al., 2022). The question: does a 134M model trained on a free GPU at
context 128 build one at all.

### Two heads out of 96 do induction

Feed the model a random sequence repeated twice and measure where every head looks.
**Random tokens are the point.** The model cannot fall back on memorized English, so
any copying has to come from the context. A causal head at position `i` splits its
attention over `i + 1` positions, so a head with no preference puts `1/(i+1)` on any
one of them. Averaged over the measured queries that is **0.0142**, the baseline
below. 16 sequences.

![Induction score by head](images/induction_heads.png)

| Head | Attention on the induction target | vs baseline |
|---|---|---|
| **L6.H9** | **0.4188** | **29.5x** |
| **L7.H8** | **0.2738** | **19.3x** |
| L6.H7 | 0.0864 | 6.1x |
| everything else | ~0.014 | ~1x |

`L6.H9` puts **42% of its attention on one position** out of fifty plus. Three things
it could have been instead, and is not:

| Alternative | Ruled out by |
|---|---|
| A duplicate-token head, which spots repeats without predicting | 0.4188 on the **next** token against 0.0136 on the **same** token, a factor of **31** |
| A fixed positional offset, which this model could express because its positions are learned | Changing the repeat period across 32, 48 and 56 moves the score **8%**; a fixed offset would collapse at every period but one |
| An artifact of my probe | The same probe on an untrained model returns ~1x across all 96 heads, asserted in `tests/` |

### Removing them destroys 85.4% of the copying

Attention is correlational: a head can look in exactly the right place and contribute
nothing. So overwrite its slice of the attention output through a forward hook,
before `out_proj` mixes the heads, and measure again.

| Intervention | 2nd-copy loss | 1st-copy loss | 95% CI | Copying destroyed |
|---|---|---|---|---|
| **Mean ablation** | **+3.3473** | +0.28 | [+3.2306, +3.4697] | **85.4%** |
| Zero ablation | +3.9089 | +0.28 | [+3.7656, +4.0438] | 99.7% |

The null is size-matched: random head *pairs*, not single heads, drawn from heads
with no induction role. It comes out at **+0.0624 ± 0.1074**, worst control
**+0.2431**. The bottom of the treatment's confidence interval beats the worst
control by more than 13x, which is **30.6 standard deviations** above the null mean.
First-copy loss moves only +0.28, so this removes one capability rather than damaging
the network in general.

Two corrections, both applied, both of which lower the headline number:

- **Mean ablation, not zero.** Zeroing also removes the head's average contribution,
  which pushes the residual stream somewhere the rest of the network never sees and
  overstates the damage (Zhang and Nanda, 2024). The claim survives the stricter
  test. The figure drops from 99.7% to 85.4%.
- **A corrected denominator.** Later positions have more context whether anything
  repeats or not. On non-repeated sequences that is worth **+0.51 nats**, so the real
  copying benefit is 3.92, not 4.43.

### It is a circuit, not two correlated heads

An induction head cannot work alone. To predict what follows the second `B`, it has
to attend to the position after the *first* `B`. But that position holds `C`, and
nothing about `C` says "I come after a B", so the search fails. Something has to tag
each position with the token before it first. That is a **previous-token head**, and
it has to run in an earlier layer, because the tag must be written before it can be
matched.

So the mechanism predicts one below layer 6. `L5.H11` is it: 6.2x baseline, layer 5,
directly beneath the induction heads in layers 6 and 7.

The test that earns the word "circuit" is not the loss, it is the attention pattern.
If the two heads only correlate, ablating L5.H11 leaves the induction pattern alone.

| Head | Induction score, intact | With L5.H11 ablated | Fall |
|---|---|---|---|
| L6.H9 | 0.4188 | 0.2504 | **40.2%** |
| L7.H8 | 0.2738 | 0.1671 | **39.0%** |

The upstream head is writing the tag the downstream heads match on. That is
**K-composition**: a mechanism with parts and an order they have to run in.

The pattern does not fall all the way to baseline, and that fits the rest of the
picture. The previous-token job is **redundant**: L2.H2, L2.H10 and L3.H0 all do part
of it, so removing L5.H11 leaves weaker copies behind. The induction job is not. Two
heads out of 96 carry it, which is why removing both removes the capability.

**Scope.** 85.4% is copying on repeated random tokens, which is narrower and easier
than in-context learning on real text (Crosbie and Shutova, 2024). It comes from one
final checkpoint, so it shows the circuit exists and says nothing about when it
formed.

Full method, all controls and 13 references:
**[interp/INDUCTION_HEADS.md](interp/INDUCTION_HEADS.md)**.

```bash
python interp/induction_heads.py       --ckpt weights8b_300epoch.pth
python interp/ablate_heads.py          --ckpt weights8b_300epoch.pth --ablation mean
python interp/previous_token_heads.py  --ckpt weights8b_300epoch.pth
python interp/circuit_controls.py      --ckpt weights8b_300epoch.pth
```

## 6. What is next

Both sections above read something out of the weights. The direction I want is
harder: can a model **report** on its own internals, and is the report true?
Self-report is the cheapest monitoring interface there is, and it is known to be
unfaithful. Models deny using a hint that ablation proves they used (Turpin et al.,
2023).

The bottleneck is not the verbalization. It is the **ground truth**: you cannot
score a report about an activation until you know what was in it. That ground truth
comes from probe readouts, feature activations, and measured ablation and patching
effects. It is the half I have been building. `ablate_heads.py` already reads and
overwrites activations through forward hooks, with a size-matched null and bootstrap
CIs, on a model I can account for down to the optimizer step.

| Next | Why it, and why now |
|---|---|
| **Linear probes** on the residual stream | Start where the answer is known: `L5.H11` writes a previous-token tag, so a probe must recover the previous token at layer 5 and fail at layer 0. Same discipline as the planted-cliff test in `tests/`. |
| **Activation patching** | Ablation shows a head is necessary. Patching shows which downstream components its output reaches. |
| **Sparse autoencoders** on residual and MLP activations | Turns "which head" into "which feature". At 134M this trains on the same free GPU. |
| **Self-report faithfulness** | With probes and features in place, the internals label themselves. In-context first, no training: how much can a model already say about a feature a probe says is active? Then fine-tune on those labels and test generalization to unseen inputs and unseen features. |

This model cannot answer questions about itself. It is a base model, so the
verbalization half needs a chat model. Its job is the measurement half: small enough
to probe end to end on a CPU, fully specified, and mine.

Two experiments left on this checkpoint. **Formation dynamics**: induction heads
appear abruptly during training, and seeing when this one formed needs the mid-run
checkpoints `train.py` now saves. **The data-distribution experiment**: Chan et al.
(2022) showed in-context learning fails to emerge when burstiness and Zipfian
marginals are stripped from the data, and the tokenizer, pipeline and training loop
here make that ablation a 33-hour run.

## 7. Run it

```bash
pip install -r requirements.txt

python -m pytest tests/ -v                       # 23 tests, no checkpoint needed
python verify.py --ckpt weights8b_300epoch.pth   # perplexity vs GPT-2-small
```

Weights: **[huggingface.co/umerateeq/zerotogpt-134m](https://huggingface.co/umerateeq/zerotogpt-134m)**
(538 MB, past GitHub's file limit), SHA-256 `958084909bbdcd20afb4f764b7628da6e8aef3d5212bd74b4097a158ae91bf49`.
Every analysis command in this README runs on CPU in under two minutes.

```bash
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; \
print(hf_hub_download('umerateeq/zerotogpt-134m', 'weights8b_300epoch.pth'))"
```

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
  CHECKPOINT_AUDIT.md      the run: the weight-decay clock, its controls, its limits
  INDUCTION_HEADS.md       the circuit: method, controls, limitations, 13 references
tests/                     23 tests, 14 on the analysis code
verify.py                  one command to check the perplexity claims
```

**Scope.** Not an assistant: no instruction tuning, no RLHF, it completes text. Not
competitive: GPT-2-small beats it 3.1x on WikiText-2, which is what ~9 tokens per
parameter buys. Not a trajectory: the circuit results come from one final checkpoint.

## Credits

Implementation follows Sebastian Raschka,
[Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch),
and Andrej Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT), for the
memory-mapped sampler and parts of the training loop. Data:
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). Method:
[Elhage et al., A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html);
[Olsson et al., In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).