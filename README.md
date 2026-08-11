# I pretrained a 134M-parameter LLM from zero. Now I am doing interpretability on it.

Every component is written out in pytorch. Multi-head attention, the causal mask,
LayerNorm, the feed-forward block and the block wiring are in
**[pretrain/model.py](pretrain/model.py)**, 190 lines; the sampler is in
[generate.py](pretrain/generate.py) and the training loop in
[train.py](pretrain/train.py). No `nn.Transformer`, no HuggingFace model class, no
`x-transformers`. PyTorch tensors and nothing above them. The architecture follows
Sebastian Raschka's [Build a Large Language Model (From
Scratch)](https://github.com/rasbt/LLMs-from-scratch); everything downstream of the
checkpoint is mine.

Pretraining it was the first half. The second half is interpreting it. A trained
transformer is a black box: a loss curve tells you nothing about what it learned
to *do*. So I started measuring the artifact instead of reading its config. Two of
its 96 attention heads turn out to carry 85% of its ability to copy from context,
and ablating them takes that capability away while leaving ordinary prediction
intact. That measurement, and the controls that make it trustworthy, is where the
interpretability half of this repo starts.

| | |
|---|---|
| Parameters | **134,077,440** |
| Training tokens | **1.2B**, FineWeb-Edu |
| Hardware | a single Tesla P100 16 GB, free Kaggle session |
| Held-out perplexity | **38.89**, against a Chinchilla prediction of **38.70** ([derivation](#the-chinchilla-check)) |
| Throughput | **10,200 tokens/sec**, **31.7% MFU** |
| Induction heads | 2 of 96 heads carry **85.4%** of copying, **42σ** over a size-matched null |

### Contents

1. [The model](#1-the-model)
2. [Training](#2-training)
3. [Results](#3-results)
4. [Reading the training run out of the weights](#4-reading-the-training-run-out-of-the-weights)
5. [Measuring what the model learned to do](#5-measuring-what-the-model-learned-to-do)
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
| 8 transformer blocks | 56,684,544 |
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
| Dropout | 0.1 | Seed | 123 |
| Throughput | **10,200 tok/s** | Peak GPU memory | **12.24 GB** of 16 |
| Achieved | **5.93 TFLOP/s** | MFU | **31.7%** of the P100's fp16 peak |

MFU is the comparable figure: `6N + 12·n_layers·context·d_model` FLOPs per token with
`N` the **non-embedding** parameters (95,283,456), because a `tok_emb` lookup is a
gather and costs no FLOPs. At 10,200 tok/s that is 5.93 TFLOP/s against the P100's
18.7 TFLOP/s fp16 peak. `train.py` computes this per run.

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

Training consumed ~1.2B tokens of a larger corpus, so no window was seen twice.

Every run writes its config, git commit, seed, library versions, GPU name, per-eval
metrics, throughput, MFU, peak memory and loss curve to `runs/`, and with
`--save-every N` keeps intermediate checkpoints alongside them. The released
checkpoint predates that instrumentation, which is why section 4 recovers its
training run from the weights instead of reading it off a log.

## 3. Results

| Dataset | Perplexity | Context |
|---|---|---|
| **Held-out FineWeb-Edu** `CC-MAIN-2024-18` | **38.89** | 128 |
| TinyStories | 35.41 | 128 |
| WikiText-2 | 184.96 | 128 |


### The Chinchilla check

The held-out number is worth stating against a prediction rather than alone. The
Hoffmann et al. parametric form, at this model's **non-embedding** parameter count
and its measured token budget:

```
L(N, D) = 1.69 + 406.4 / N^0.34 + 410.7 / D^0.28
        = 1.69 + 406.4 / 95,283,456^0.34 + 410.7 / 1.2e9^0.28
        = 3.656 nats  ->  perplexity 38.70

measured on held-out FineWeb-Edu:      38.89
```

Non-embedding N because that is the convention the scaling law is fit in. This is a
consistency check, not a validation of the scaling law: it says the run landed where
a 95M-non-embedding-parameter model trained on 1.2B tokens should land, so nothing
went quietly wrong with the optimization.

### GPT-2-small on the identical harness

| Model | Params | WikiText-2 perplexity |
|---|---|---|
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

The checkpoint is a bare `state_dict`: 117 tensors, no config, no logs, no metadata.
The architecture, the context it actually trained at, the number of optimizer steps
it took and the tokens that landed in a weight update all come back out of those
tensors. The rest of the training table (optimizer, learning rate, precision,
throughput, memory) does not: no tensor carries it.

```bash
python interp/audit_checkpoint.py --ckpt weights8b_300epoch.pth
```

That command prints the architecture and the context cliff. The step count and its
controls below are derived in
**[interp/CHECKPOINT_AUDIT.md](interp/CHECKPOINT_AUDIT.md)**.

The model has 256 positional rows. The run used 128. The other 128 got weight decay
every step and gradient never, which makes them a record of the run.

### The trained context is 128

Decay shrinks a row toward zero. Gradient holds it up. So rows the run never reached
collapse:

```
   positions  0-31   32-63   64-95  96-127 | 128-159 160-191 192-223 224-255
   mean norm 0.2132  0.1886  0.2196  0.2487 |  0.0024  0.0024  0.0024  0.0023
```

![Positional embedding row norms, a 93x cliff exactly at position 128](images/pos_emb_norms.png)

A **93x cliff at position 128**. This fixes the evaluation context: the same data
scored at 256 reads 74.70 instead of 38.89, all of it measurement error.

### The run took 234,478 optimizer steps

Every landed step multiplied those dead rows by the same factor. Undo it and you get
the step count. `nn.Embedding` starts at N(0,1), so a fresh row of width 768 has norm
`sqrt(768) = 27.71`:

```
final = initial x (1 - lr x weight_decay)^N

0.002346 / 27.7765 = 8.448e-5
ln(8.448e-5) / ln(1 - 4e-4 x 0.1) = 234,478 steps
```

Three routes agree. Two never read the source:

| Route | Reads | Result |
|---|---|---|
| Decay clock on `pos_emb` rows 128-255 | the weights | 234,478 steps |
| 517 never-sampled `tok_emb` rows | a different part of the weights | 227,000 to 236,600 steps |
| Loop counts, 300 cycles x 1000 batches | the source | 300,000 steps, 1.23B tokens |

The two weight routes share no parameters and agree to a few thousand steps. Both
land 22% under the loop count, which is the `GradScaler` skip rate: a skipped step
runs forward and backward but applies no update and no decay.

### The controls

This works only if those rows are initialization that was scaled down, not rows that
trained and then decayed. Training aligns neighbouring rows and spreads their norms.
Scaling does neither.

| | Norm spread (std/mean) | Adjacent-row cosine |
|---|---|---|
| Fresh initialization | 2.75% | 0.026 |
| **Checkpoint rows 128-255** | **2.72%** | **0.028** |
| Checkpoint rows 0-127 (trained) | 23.10% | 0.822 |

They match fresh initialization and are nowhere near the trained rows. 2.75% is also
what theory gives: a random vector of width 768 varies by `1/sqrt(2 x 768) = 2.6%`.
`tests/` plants a fake cliff and checks the audit finds it.

Full method and limits: **[interp/CHECKPOINT_AUDIT.md](interp/CHECKPOINT_AUDIT.md)**.

## 5. Measuring what the model learned to do

Sections 3 and 4 read the *run* out of the weights. This asks a smaller and harder
question of the same tensors: is there a specific, localizable computation in there,
and can I show it causally rather than by eye?

This section is the beginning of an interpretability track, not a finished one. Every
number below is measured and controlled; the claims are deliberately kept to what the
measurements support, and what they do not support is named.

**All figures: 32 random sequences of length 48 repeated twice, seed 123, on CPU,
from the single released checkpoint.**

### Two heads out of 96 do induction

An induction head follows one rule: *"I have seen this token before. What came next
last time? Attend to that."* It is a leading mechanistic account of in-context
learning (Olsson et al., 2022). Does a 134M model trained on a free GPU build one?

Feed it a random sequence repeated twice and measure where every head looks. **Random
tokens are the point**: the model cannot use memorized English, so any copying comes
from the context. A head with no preference scores **0.0142**.

`model.py` is never modified, because it is the code the checkpoint was trained with.
The probe hooks the tensor entering each attention module and recomputes the softmax
from that module's own `W_query` and `W_key`, and `tests/` asserts that recomputation
matches the module's real output to **1e-6**. Every number below rests on that
readout, so it is checked rather than assumed.

![Induction score by head](images/induction_heads.png)

| Head | Attention on the induction target | std | vs baseline |
|---|---|---|---|
| **L6.H9** | **0.4263** | 0.0380 | **30.0x** |
| **L7.H8** | **0.2777** | 0.0352 | **19.6x** |
| L6.H7 | 0.0891 | 0.0163 | 6.3x |
| everything else | ~0.014 | | ~1x |

`L6.H9` puts **43% of its attention on one position** out of fifty plus. Three things
it is not:

| Alternative | Ruled out by |
|---|---|
| A duplicate-token head | 0.4263 on the **next** token, 0.0131 on the **same** token: **32x** |
| A fixed positional offset | Changing the repeat period across 32, 48, 56 moves the score **10%**, where a fixed-offset head would collapse off-period |
| An artifact of my probe | On an untrained model the same probe finds no head above 3x baseline, asserted in `tests/` on a toy config |

### Removing them destroys 85.4% of the copying

A head can look in the right place and do nothing. So overwrite its slice of the
attention output with a forward hook and measure again.

| Intervention | 2nd-copy loss | 1st-copy loss | 95% CI | Copying destroyed |
|---|---|---|---|---|
| **Mean ablation** | **+3.3473** | **-0.3236** | [+3.2279, +3.4725] | **85.4%** |
| Zero ablation | +3.9089 | +0.28 | [+3.7656, +4.0438] | 99.7% |

Under mean ablation, second-copy loss rises by 3.35 nats while first-copy loss
*falls* by 0.32. The heads contribute nothing to ordinary next-token prediction and
almost everything to copying, which is what makes this targeted rather than general
damage.

The null is size-matched: random head *pairs*, not single heads, drawn from every head
except the five with a measured induction or previous-token role, since leaving
circuit members in the null would contaminate it. Over 12 pairs it comes out at
**+0.0465 ± 0.0779**, worst control **+0.2431**, putting the treatment **42 standard
deviations** above the null mean. That standard deviation is itself estimated from 12
samples, so read it as "far outside the null" rather than as a calibrated p-value.

Two corrections, both applied, both lowering the number:

- **Mean over zero ablation.** Zeroing strips the head's average output as well as its
  input-dependent signal, pushing the residual stream off-distribution and overstating
  damage (Zhang and Nanda, 2024). 99.7% becomes 85.4%. The mean is taken over the same
  repeated-random-token inputs the metric is defined on; a natural-text reference mean
  is the stricter variant and was not run.
- **A corrected denominator.** Later positions have more context whether anything
  repeats or not, worth **+0.51 nats** measured on non-repeated sequences, so the real
  copying benefit is 3.92 rather than the naive 4.43.

### The upstream half, and what it does not show

An induction head cannot work alone. To predict what follows the second `B` it must
attend to the position after the *first* `B`. That position holds `C`, and nothing
about `C` says "I come after a B". Something has to tag each position with the token
before it, in an earlier layer, because the tag must exist before it can be matched.
That is a **previous-token head**, and `L5.H11` is one: 6.2x baseline, in layer 5,
directly below the induction heads in 6 and 7.

It is load-bearing on its own terms. Ablating it costs **+1.22 nats** on the repeated
half, 31.1% of the copying benefit, while first-copy loss again *improves*, by 0.05.

And the induction heads' attention depends on it, which is a stronger statement than
the loss result:

| Head | Induction score, intact | With L5.H11 ablated | Fall |
|---|---|---|---|
| L6.H9 | 0.4263 | 0.2565 | **39.8%** |
| L7.H8 | 0.2777 | 0.1723 | **37.9%** |

**What this does not establish is K-composition.** Ablating L5.H11 removes it from the
query path, the key path, the value path and the intervening MLPs at once. Key
composition is the mechanism that *predicts* this result; isolating it needs path
patching, which is the next technique on the list and is not built yet. What is
measured here is causal dependence of the downstream attention pattern on the upstream
head.

**A hypothesis this repo tested and did not confirm.** The obvious reading of "the
pattern falls 40% but not to baseline" is that the previous-token role is redundant,
since L2.H2, L2.H10, L2.H6 and L3.H0 all show partial previous-token attention. If so,
ablating all four together should cost far more than ablating L5.H11 alone. It does
not:

| Ablated | Copying destroyed |
|---|---|
| L5.H11 alone | 31.1% |
| L5.H11 + L2.H2 + L2.H10 + L3.H0 | 36.8% |

Three extra heads buy 5.7 points, and the size-matched null rises from +0.031 to
+0.090 across that change, so most of the gain is the generic cost of ablating more
heads. The previous-token contribution to copying is **concentrated in L5.H11**, not
spread across the layer-2 and layer-3 heads that share its attention signature. Which
means the residual 60% of the induction pattern is coming from somewhere this repo has
not identified, and saying so is more useful than the tidier story.

**Scope.** 85.4% is copying on repeated random tokens, an easier and narrower quantity
than in-context learning on real text (Crosbie and Shutova, 2024). It is one
checkpoint, so it shows the computation exists, not when it formed. Backup-head
compensation (McGrath et al., 2023) is not tested, so these are drops in capability,
not necessarily each head's full share of it.

Full method, all controls, 13 references:
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
| **Linear probes** on the residual stream | Start where the answer is known. Previous-token behaviour is spread across L2.H2, L2.H10, L3.H0 and L5.H11, so the prediction is a *shape*, not a binary: probe accuracy for the previous token at chance in layer 0, rising by layers 2-3, rising again after 5. The causal cross-check is that ablating L5.H11 should cost accuracy at layer 6 and not at layer 3. Same discipline as the planted-cliff test in `tests/`: validate the detector against an answer you already know. |
| **Activation patching** | Ablation shows a head is necessary. Patching shows which downstream components its output reaches, which is exactly what separates K-composition from the generic causal dependence measured above. |
| **Sparse autoencoders** on residual and MLP activations | Turns "which head" into "which feature". At 134M this trains on the same free GPU. |
| **Self-report faithfulness** | With probes and features in place, the internals label themselves. In-context first, no training: how much can a model already say about a feature a probe says is active? Then fine-tune on those labels and test generalization to unseen inputs and unseen features. |

**How the two halves join.** A faithfulness claim is only defined when the report and
the measured activation come from the same forward pass of the same model, so the
experiment itself has to run inside one instruction-tuned model with its own probes
and its own SAE. This 134M model is not that model: it is a base model and cannot
answer questions about itself. Its role is the substrate where the measurement half
gets built and validated against known answers, on something small enough to probe
end to end on a CPU, specified down to the optimizer step, and mine.

Two experiments this repository is set up for, neither run yet. **Formation
dynamics**: induction heads appear abruptly during training, and seeing when this one
formed needs weights saved *during* the run. The released checkpoint kept only its
final state, so this needs a rerun; `train.py --save-every N` now keeps them. **The
data-distribution experiment**: Chan et al. (2022) showed in-context learning fails to
emerge when burstiness and Zipfian marginals are stripped from the data. The
tokenizer, the streaming pipeline and the training loop are all here, so that
ablation is 1.2B tokens at the 10,200 tok/s this run measured, about **33 hours** on
the same free GPU.

## 7. Run it

```bash
pip install -r requirements.txt

python -m pytest tests/ -v                       # 24 tests, no checkpoint needed
python verify.py --ckpt weights8b_300epoch.pth   # perplexity vs GPT-2-small
```

Weights: **[huggingface.co/umerateeq/zerotogpt-134m](https://huggingface.co/umerateeq/zerotogpt-134m)**
(538 MB, past GitHub's file limit), SHA-256 `958084909bbdcd20afb4f764b7628da6e8aef3d5212bd74b4097a158ae91bf49`.

```bash
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('umerateeq/zerotogpt-134m', 'weights8b_300epoch.pth', local_dir='.')"
```

Every command in section 5 runs on CPU in under two minutes. `verify.py` is the
exception: it scores 285,396 tokens through two models, roughly an hour on CPU and a
few minutes on any GPU.

```
pretrain/
  model.py          LayerNorm, FeedForward, MultiHeadAttention, TransformerBlock, GPTModel
  config.py         every hyperparameter in one place
  data.py           memory-mapped batch sampling from a uint16 token file
  tokenize_data.py  stream a HuggingFace dataset into that token file
  train.py          fp16, grad accum, warmup + cosine, clipping, resumable, run records,
                    --save-every for intermediate checkpoints
  generate.py       greedy and temperature/top-k sampling
  evaluate.py       perplexity, two protocols, GPT-2-small baseline. Reads the trained
                    context off the weights and warns before scoring beyond it
interp/
  audit_checkpoint.py      recover architecture and trained context from weights
  induction_heads.py       probe all 96 heads for induction behaviour
  ablate_heads.py          zero and mean ablation, size-matched null, bootstrap CIs
  previous_token_heads.py  the upstream half of the circuit
  circuit_controls.py      composition, repeat period, positional baseline
  _bootstrap.py            puts pretrain/ on sys.path for the scripts above
  CHECKPOINT_AUDIT.md      the run: the weight-decay clock, its controls, its limits
  INDUCTION_HEADS.md       the circuit: method, controls, limitations, 13 references
tests/                     24 tests, 14 on the analysis code
images/                    figures referenced above
conftest.py                pytest path setup
requirements.txt
verify.py                  exits 0 only if the measured numbers match this README
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