# What can you recover about a training run from the weights alone?

**For this checkpoint: most of the architecture, the context it actually trained at,
the number of optimizer steps it took, and the tokens that landed in a weight update.
No config file, no logs, no filename. The head partition is the one thing that leaves
no trace, and section 1 says why.**

`weights8b_300epoch.pth` is a bare `state_dict`. 117 tensors, no metadata, nothing
attached. That is the interesting case, because it is also the normal case: most
checkpoints in the wild arrive with a config that describes what someone intended to
run, and no evidence about what actually ran.

This document is the method. The companion document
[INDUCTION_HEADS.md](INDUCTION_HEADS.md) asks the other question of the same tensors:
not *what was this trained with* but *what did it learn*.

Reproduce:

```bash
python interp/audit_checkpoint.py --ckpt weights8b_300epoch.pth --plot images/pos_emb_norms.png
```

## Why the weights are the better witness

A config records intent. It is written before the run, it is not updated when the
run diverges from it, and the ways a run diverges are exactly the ways that leave no
error message: a shape read from a stale global, a scheduler stepping an optimizer
that no longer owns the parameters, a flag whose default silently wins. None of
these raise. All of them are visible in the weights afterwards.

This checkpoint is a case in point. It allocates 256 positional rows and trained on
128 of them, and the file name records neither. The positional embedding table
records both.

That gap is the reason for this method, and it sets the standard the rest of the
repository holds to: do not accept a description of a model, measure the artifact.

## 1. Architecture, from tensor shapes

The easy part, and worth doing first because it constrains everything after it.

| Property | Recovered from |
|---|---|
| Vocabulary 50257, width 768 | `tok_emb.weight.shape` |
| Positions allocated 256 | `pos_emb.weight.shape[0]` |
| Layers 8 | highest index in `trf_blocks.N.*` |
| Feed-forward expansion 4x | `trf_blocks.0.ff.layers.0.weight.shape[0] / 768` |
| QKV bias absent | `trf_blocks.0.att.W_query.bias` not in the state dict |
| Output head **untied** | `out_head.weight` present as its own tensor |
| Custom LayerNorm | `final_norm.scale` present rather than `weight` |

**Head count is not recoverable this way and the script says so.** Every attention
projection is `d_model x d_model` regardless of how the heads partition it, so
12 heads of 64 and 8 heads of 96 produce byte-identical shapes. It is a run-time
choice with no trace in the parameter shapes. Reporting it as recovered would be a
lie by omission, so the tool prints it as an assumption.

## 2. The trained context, from the positional table

The argument in one line: **a row of `pos_emb.weight` only receives gradient if some
batch was long enough to reach that position, but AdamW applies weight decay to every
parameter on every step whether or not a gradient arrived.**

So an untrained row is multiplied by `(1 - lr * wd)` on every step and decays
geometrically toward zero, while a trained row is pushed around by gradients and
holds its norm. The boundary between the two regimes is the context the run actually
used.

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

![Positional embedding row norms, a 93x cliff exactly at position 128](../images/pos_emb_norms.png)

**A 93x cliff, exactly at position 128.** Not a gradual falloff, not a noisy
boundary: two regimes two orders of magnitude apart with nothing in between.

The detector thresholds at `0.1 x max row norm` rather than at a fraction of the
median. That choice matters here: when half the table is dead, the median itself sits
inside the dead cluster and a median-referenced threshold finds nothing.
`tests/test_interpretability.py::test_audit_threshold_survives_a_half_dead_table`
pins that case.

**This is not a curiosity, it changes every number the model reports.** Scoring at
context 256 puts half of every window on rows that never received gradient, and
perplexity goes from 38.89 to 74.70 on identical data for reasons that have nothing
to do with model quality. `pretrain/evaluate.py` reads the cliff off the weights and
warns before printing.

## 3. The optimizer step count, from the same dead rows

The dead rows are doing something more useful than marking a boundary. **They are a
clock.**

A row that received decay and never received gradient has been multiplied by the same
factor on every successful step. Its total shrinkage is therefore the product of that
factor over the entire life of the run:

```
final = initial x (1 - lr x weight_decay)^N
```

Everything except `N` is known. The learning rate was constant at 4e-4, decay was
0.1, the final norm is in the file, and the initial norm follows from the
initialization: `nn.Embedding` defaults to N(0, 1), so a fresh 768-dimensional row has
expected norm `sqrt(768) = 27.71`, and the measured mean over a fresh table is 27.7765.

```
0.002346 / 27.7765 = 8.448e-5
ln(8.448e-5) / ln(1 - 4e-4 x 0.1) = 234,478 successful optimizer steps
```

At 4,096 tokens per step that is **~0.96B tokens' worth of applied updates**.

### Three routes, one answer

A single derivation with this many assumptions is a hypothesis, not a measurement. So
it is checked against two others:

| Route | What it reads | Result |
|---|---|---|
| Weight-decay clock on `pos_emb` rows 128-255 | the weights | 234,478 steps |
| 517 never-sampled `tok_emb` rows sitting at the same decay floor | a **disjoint** parameter subspace | 227,000 to 236,600 steps |
| The training loop's own counts, 300 cycles x 1000 batches | the source code | 300,000 steps, 1.23B tokens |

The second route is the one that makes this convincing. Those 517 token-embedding rows
correspond to BPE tokens that never appeared in 1.2B tokens of FineWeb-Edu, so they
took decay and no gradient for exactly the same reason the dead positional rows did,
in a completely different tensor. The two estimates share no parameters and no
assumption beyond the decay law itself, and they bracket each other to within a few
thousand steps out of 234,000.

Both sit 22% below the loop count, and that residual has an explanation rather than
being absorbed as noise: `GradScaler` skips a step on fp16 gradient overflow, and a
skipped step still runs the forward and backward pass but applies no update and
therefore no decay. So the clock counts steps that changed the weights, which is a
lower bound on steps attempted, and a 22% skip rate is unremarkable for fp16 without
loss-scale tuning.

**~1.2B tokens** is the figure the repository reports. The Hoffmann et al. parametric
form, evaluated at the non-embedding parameter count 95,283,456 and D = 1.2e9, gives
`1.69 + 406.4/N^0.34 + 410.7/D^0.28 = 3.656` nats, a perplexity of **38.70** against
the **38.89** measured on held-out FineWeb-Edu.

## 4. The controls

The clock assumes rows 128-255 are **untouched initialization that was only scaled
down**. The alternative that would break it is rows that trained for a while and then
decayed, which would also end up small. Two properties separate those cases.

**Direction.** Gradient descent aligns nearby positional rows, because adjacent
positions play similar roles. Pure scaling preserves whatever random directions the
rows started with. So the cosine between adjacent rows should stay near zero if the
rows are untouched, and rise if they trained.

**Norm uniformity.** Random initialization gives rows a tight norm distribution.
Training spreads it, because different positions get different amounts of gradient.
Scaling every row by the same factor leaves the relative spread unchanged.

| | Norm spread (std/mean) | Adjacent-row cosine |
|---|---|---|
| Fresh initialization | 2.75% | 0.026 |
| **Checkpoint rows 128-255** | **2.72%** | **0.028** |
| Checkpoint rows 0-127 (trained) | 23.10% | 0.822 |

Both statistics land on the fresh-initialization value to two decimal places, and both
are an order of magnitude away from the trained rows. **The dead rows were scaled, not
trained.**

The 2.75% figure is independently predictable. The relative standard deviation of the
norm of a `d`-dimensional standard Gaussian is approximately `1/sqrt(2d)`, which for
`d = 768` gives 2.6%. The measured 2.75% agrees, so the fresh-init column is not just
an empirical reference, it is the value theory requires.

**The detector is tested against known answers**, in `tests/test_interpretability.py`:

| Test | Asserts |
|---|---|
| `test_audit_finds_a_planted_context_cliff` | a synthetic cliff planted in a fresh model is recovered at the planted position |
| `test_audit_reports_no_cliff_when_every_position_was_trained` | no false positive on a fully trained table |
| `test_audit_threshold_survives_a_half_dead_table` | the max-referenced threshold does not collapse when half the rows are dead |

The first is the important one. A detector that has never been run against a case
where the answer is known in advance is not a detector, it is an opinion.

## 5. What this method cannot establish

**It counts successful steps only.** Any optimizer step that was skipped, by
`GradScaler` or otherwise, applies no decay and leaves no mark. The clock is therefore
a lower bound on steps attempted and an exact count of steps applied. These are
different quantities and the distinction is load-bearing: the 22% gap against the loop
count is precisely this.

**It needs a constant learning rate.** Under a live schedule the same measurement
returns the integral of the learning rate over training rather than the step count,
and the two cannot be separated without knowing the schedule, which is the thing the
method exists to avoid assuming. A checkpoint trained with warmup and cosine decay
carries a number, not a step count.

**It needs weight decay applied to the embedding tables.** Many training setups
explicitly exclude embeddings, LayerNorm gains and biases from decay. On such a
checkpoint the dead rows sit at their initialization values and there is no clock at
all. The method reads a side effect of a specific optimizer configuration, not a
universal property of transformers.

**It needs untrained parameters to exist.** A run whose context exactly matched its
allocated positions leaves no dead rows, and the never-sampled token rows are the only
remaining channel. A model with full vocabulary coverage would close that one too.

**The initial norm is inferred, not recorded.** It comes from the initialization
scheme, so a checkpoint whose init is unknown or non-standard requires that assumption
to be stated and checked. Here it is checkable two ways: the `sqrt(768)` prediction and
the norm-spread control both agree with `nn.Embedding` defaults.

## 6. Why this is worth doing

Two reasons, one narrow and one general.

The narrow one: every interpretability claim in this repository is a claim about a
specific model, and a claim about a model you cannot precisely describe is worth
less. The induction-circuit result is quoted at context 128 because the weights say
128, not because a config said so.

The general one is the method itself. Descriptions of a system are cheap to produce,
are written before the fact, and fail silently when the system diverges from them.
Measurements taken from the system's own internals do not. The supervision signal
here (decay applied, gradient not applied) came free from the artifact, and it is
checkable against a known answer: `tests/test_interpretability.py` plants a cliff at
a position it chooses and asserts the audit recovers that position.

That is the same shape as the problem of an unfaithful self-report, one level down.
A report about a system is only worth what the independent measurement of that
system is worth, which is why the measurement has to be validated first.

## References

- Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay Regularization.* The
  decoupled update `theta <- theta(1 - lr * lambda)` that makes the clock linear in the
  step count. Under L2-coupled decay the same reasoning does not hold.
- Micikevicius, P., et al. (2018). *Mixed Precision Training.* Loss scaling and the
  skipped-step behaviour that accounts for the 22% gap.
- Hoffmann, J., et al. (2022). *Training Compute-Optimal Large Language Models.* The
  token budget the recovered step count is checked against in the README.
