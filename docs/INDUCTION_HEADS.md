# Does this model have induction heads?

**Yes. Two of its 96 attention heads are clear induction heads, both in the last
two layers.**

An induction head implements one rule: *"I have seen this token before. What came
next last time? Attend to that."* Given a sequence that repeats, a head sitting on
the second occurrence of a token should attend back to whatever followed the
first occurrence. It is thought to be the main mechanism behind in-context
learning in transformers (Olsson et al., *In-context Learning and Induction
Heads*, Anthropic 2022).

I wanted to know whether a small model, trained from scratch on a free GPU under
a badly configured run, develops them at all.

Reproduce:

```bash
python induction_heads.py --ckpt weights8b_300epoch.pth --plot docs/images/induction_heads.png
```

## Method

1. Build a random token sequence of length 48 and concatenate it with itself, so
   the model sees `[seq, seq]`, 96 tokens total. **Random tokens are the point**:
   the model cannot fall back on memorized English, so any copying behaviour has
   to come from the pattern in the context rather than from training data.
2. Capture every head's attention pattern.
3. For each head, look only at query positions in the second copy. A query at
   position `i` is on a token it already saw at `i - 48`. The **induction target**
   is `i - 48 + 1`: the token that came immediately after that earlier occurrence.
4. Score the head by its average attention weight on that target.
5. Average over 16 random sequences, so the result describes the model rather
   than one lucky draw.

Scores are reported as multiples of a **uniform baseline**. A causal head at
position `i` spreads attention over `i + 1` positions, so an indifferent head
puts `1/(i+1)` on any single one; averaged over the measured queries that is
0.0142. A head scoring 1x is doing nothing special. 29x is not an accident.

## Result

![Induction score by head](images/induction_heads.png)

| Head | Score | vs uniform |
|---|---|---|
| **L6.H9** | 0.4188 | **29.5x** |
| **L7.H8** | 0.2738 | **19.3x** |
| L6.H7 | 0.0864 | 6.1x |
| L7.H6 | 0.0497 | 3.5x |
| L7.H3 | 0.0216 | 1.5x |
| everything else | ~0.014 | ~1x |

`L6.H9` puts **42% of its total attention mass** on a single position, the
induction target, when it could be spreading it over 50-plus positions. That is a
head doing one job.

## The other half of the circuit

**The induction heads are in layers 6 and 7 of 8, and layers 0 to 5 are empty.**
That is not a coincidence, and the reason is the most interesting part of the
mechanism.

An induction head cannot work alone. Consider what it has to do. You are at
position 40 looking at token `B`, and you want to predict what follows. You saw
`B` before at position 12, and `C` followed it at position 13. So you want to
attend to position 13.

But how do you *find* position 13? You would have to search for "the position
that comes after a `B`". Position 13 holds `C`, and nothing about `C` says "I
follow a B". The search fails.

**A previous-token head fixes this.** It attends from every position to the one
before it, copying information about token i-1 forward into position i. After it
runs, position 13 carries not only "I am C" but also "the token before me was B".
Now a later head can match on that tag and land on position 13.

Two heads, two layers, in that order. With standard attention it cannot be done
in one layer, because the tag has to be written before it can be matched.

**That is narrower than it first appears, and an earlier version of this document
overstated it.** Olsson et al. ran a "smeared keys" experiment in which each head's
key is a learned mixture of the current and previous token's key, and a *one-layer*
model built that way does form induction heads. So the constraint is on standard
attention, not on induction as such. There is also a second route this model could
in principle use: Elhage et al. (2021) note that induction can be implemented by
positional pointer arithmetic rather than token matching, and this model has
learned absolute positional embeddings. The control that rules that out is varying
the repeat period, which is Control 2 below.

So the circuit predicts a previous-token head somewhere before layer 6. There is
one:

```bash
python previous_token_heads.py --ckpt weights8b_300epoch.pth
```

![Previous-token score by head](images/prev_token_heads.png)

| Head | Attention on position i-1 | vs uniform |
|---|---|---|
| **L5.H11** | 0.2420 | **6.2x** |
| L2.H2 | 0.1650 | 4.3x |
| L2.H10 | 0.1591 | 4.1x |
| L2.H6 | 0.1567 | 4.0x |
| L3.H0 | 0.1426 | 3.7x |

**L5.H11 sits in layer 5, immediately before the induction heads in layers 6 and
7.** The composition the circuit requires is available, and the ordering is
exactly what the mechanism predicts. Several weaker previous-token heads appear
in layers 2 and 3, so the model has the ingredient in more than one place.

This is the part that makes the finding a *circuit* rather than two interesting
heads: a mechanism with parts, an order they have to run in, and a reason why.

### The upstream head is causally necessary too

The same ablation was run on L5.H11:

```bash
python ablate_heads.py --ckpt weights8b_300epoch.pth --heads 5.11
```

| | Loss, 1st copy | Loss, 2nd copy | In-context benefit lost |
|---|---|---|---|
| Intact | 12.7327 | 8.3056 | |
| L5.H11 ablated | 12.6638 (**-0.07**) | 9.5581 (**+1.25**) | **28.3%** |
| L6.H9 + L7.H8 ablated | 13.0137 (+0.28) | 12.2145 (+3.91) | 85.4% (mean ablation) |

Removing the previous-token head costs 1.25 nats on the repeated half, about 7x
the worst control. First-copy loss actually **improves slightly** (-0.07), which is
about as clean a control as this experiment can produce: the head contributes
nothing to ordinary prediction and a great deal to copying.

**The upstream head matters less than the downstream ones (28% against 85%), and
that asymmetry is informative.** The previous-token role is redundant: L2.H2,
L2.H10, L2.H6 and L3.H0 all show partial previous-token behaviour, so removing
L5.H11 leaves weaker copies of the same signal for the induction heads to match
on. The induction role is not redundant. Only two heads in the model do it, and
removing both removes almost the whole capability.

A mechanism with a redundant first stage and a bottleneck second stage is a more
interesting object than "we found two heads".

**Only 2 heads out of 96, about 2%, show strong induction.** The rest are doing
something else entirely, or nothing legible by this measure. The model has enough
capacity to form the circuit once, but not to form many redundant copies of it,
which is what makes the ablation below so decisive.

### Is it induction, or just duplicate detection?

A head that attends to the *same* token's earlier occurrence, position `i - L`, is
a **duplicate-token head**, not an induction head. It notices repetition without
predicting anything. The two are easy to confuse, so they were measured against
each other on the same sequences:

| L6.H9 attends to | Attention | |
|---|---|---|
| `i - L + 1`, the **next** token (induction) | **0.4188** | std 0.0397 across 16 sequences |
| `i - L`, the **same** token (duplicate) | 0.0136 | std 0.0023 |
| `i - L + 2`, off by two | 0.0371 | std 0.0087 |

**31x more attention on the induction target than on the duplicate**, and 11x more
than on the position one further along. L7.H8 behaves the same way (0.2738 against
0.0134 and 0.0271). These are induction heads, not duplicate-token heads, and not
an artifact of attending vaguely near the right region.

The standard deviations are across sequences, and at 0.0397 on a mean of 0.4188
the effect is roughly 10x its own spread.

## Are they causally responsible, or just watching?

Attention scores are correlational. A head can look at exactly the right place
and contribute nothing to the output. So the two heads were ablated and the
model re-measured.

```bash
python ablate_heads.py --ckpt weights8b_300epoch.pth
```

**The experiment.** Feed the model `[seq, seq]` with random tokens. On the first
copy the tokens are unpredictable, so its loss is the model's floor. On the
second copy the model can copy from context, so loss should drop. That drop *is*
in-context learning, measured directly. Then zero a head's slice of the attention
output before `out_proj` mixes the heads, and measure again.

First, the denominator. Positions 48-95 have more context than 0-47 whether or
not anything repeats, so part of the raw gap is not copying. On **non-repeated**
random sequences that positional component is **+0.5071 nats**, so:

```
raw first-to-second gap    4.4271 nats
minus positional baseline  0.5071
TRUE copying benefit       3.9201 nats
```

Now the ablation, reported under both interventions:

| Intervention | 2nd-copy loss change | 95% CI | Share of copying destroyed |
|---|---|---|---|
| **Mean ablation** (field standard) | **+3.3473** | [+3.2306, +3.4697] | **85.4%** |
| Zero ablation | +3.9089 | [+3.7656, +4.0438] | 99.7% |

**Ablating two of 96 heads destroys 85.4% of the model's repeated-sequence
copying.** Mean ablation is the number to quote: zeroing removes the head's mean
contribution as well as its input-dependent signal, pushing the residual stream
off-distribution (Zhang and Nanda, 2024), so it overstates. The claim survives the
stricter test; the specific figure moves from 99.7% to 85.4%.

The first-copy loss barely moves (+0.28), so this is not general damage from
poking the network: the model is just as good as before at the thing that does
not require copying, and almost entirely unable to do the thing that does.

**Control, size-matched.** The treatment removes two heads, so the null removes
random *pairs*, drawn from heads with no induction role. Comparing a two-head
ablation against one-head controls, as an earlier version did, inflates the ratio:

| Head pair | Change in 2nd-copy loss |
|---|---|
| L5.H10 + L4.H7 | +0.2431 (worst) |
| L4.H0 + L6.H1 | +0.1429 |
| L2.H1 + L0.H0 | +0.0101 |
| L0.H4 + L1.H9 | +0.0089 |
| L6.H8 + L1.H11 | +0.0044 |
| L1.H2 + L6.H10 | -0.0352 |
| **mean, sd** | **+0.0624, 0.1074** |

Under mean ablation the treatment costs +3.3473 nats against a null of
+0.0624 ± 0.1074, which puts it **30.6 standard deviations above the null mean**,
and the low end of its confidence interval still clears the worst control by more
than 13x. Reporting a z-score against the null's spread is more honest than
dividing by a near-zero signed mean, which an earlier version did and which can be
inflated arbitrarily.

## Two controls that could have killed this, and did not

Both come from an adversarial review of this repository and are reproducible with:

```bash
python circuit_controls.py --ckpt weights8b_300epoch.pth
```

### Is the "circuit" a circuit, or two correlated heads?

Everything above shows a previous-token head *exists* upstream and is separately
load-bearing. That is not the same as showing it feeds the induction heads. The
test that distinguishes them: ablate L5.H11 and re-measure the induction heads'
**attention pattern**, not the model's loss. If the pattern is unmoved, the two are
independent and "circuit" is not earned.

| Head | Induction score, intact | With L5.H11 ablated | Fall |
|---|---|---|---|
| L6.H9 | 0.4188 | 0.2504 | **40.2%** |
| L7.H8 | 0.2738 | 0.1671 | **39.0%** |

Removing the previous-token head degrades what the induction heads attend to by
roughly 40%. That is K-composition: the upstream head is writing the tag the
downstream heads match on. It does not fall to the uniform baseline (0.0142),
which is consistent with the redundancy noted above, since L2.H2, L2.H10 and L3.H0
still supply a weaker version of the same signal.

### Is it induction, or a fixed positional offset?

Every measurement so far used a repeat period of 48, so "attends to `i-L+1`" and
"attends to absolute offset `i-47`" were the same measurement. This model has
learned absolute positional embeddings, so a head keyed to a constant offset is
expressible and would score identically. Varying the period separates them:

| Repeat period L | L6.H9 score | vs uniform |
|---|---|---|
| 32 | 0.4553 | 21.5x |
| 48 | 0.4445 | 31.3x |
| 56 | 0.4184 | 34.3x |

The score varies **8%** while the target moves 24 positions. A fixed-offset head
would collapse everywhere except L=48. This one tracks the repeat period, so it is
matching on content, not counting positions.

## Limitations, stated plainly

**First-copy loss (12.73) is higher than uniform guessing over the vocabulary
(ln 50257 = 10.82).** That is expected and worth saying: uniformly random tokens
are far outside the training distribution of educational web text, so the model
is confidently wrong. It does not affect the comparison, which is between the
first and second copy of the *same* sequence under the *same* model.

**The probe uses uniformly random tokens.** Deliberate, since it isolates
in-context copying from memorization, but it means these numbers describe
behaviour on synthetic input rather than natural text. The same experiment on
repeated natural text would be a useful complement.

**This is one checkpoint, not a training trajectory.** Olsson et al. found that
induction heads appear abruptly, in a narrow band of training, and that the
appearance coincides with a jump in in-context learning ability. That result
needs checkpoints saved *during* training, which this run does not have, because
it recorded nothing (audit bug 4). Analysing the final checkpoint says the
circuit exists; it says nothing about when it formed.

**Ablation is zeroing, not resampling.** Zeroing a head's output moves the
residual stream off the distribution the rest of the network expects. Mean
ablation, replacing the head's output with its average over a data distribution,
is the more careful control and would tighten the claim.

## Why this is in a pretraining repo

The rest of this project is about whether the training run did what it claimed.
[AUDIT.md](AUDIT.md) recovers what actually happened from the statistics of the
weights. This asks a different question of the same weights: not *was it trained
correctly* but *what did it learn*.

Both come from the same habit of not taking a model's word for what it is.

## References

The method here follows established interpretability practice. Where this repo
departs from it, the departure is named above.

**The circuit and the mechanism**

- Elhage, N., Nanda, N., Olsson, C., et al. (2021). *A Mathematical Framework for
  Transformer Circuits.* Transformer Circuits Thread. The origin of induction
  heads, QK/OV circuits, and the K-composition argument used throughout the
  "other half of the circuit" section above.
- Olsson, C., Elhage, N., Nanda, N., et al. (2022). *In-context Learning and
  Induction Heads.* Transformer Circuits Thread. The prefix-matching
  operationalization this probe implements, the induction-bump result, and the
  smeared-keys experiment cited above.

**What a circuit claim requires**

- Wang, K., Variengien, A., Conmy, A., Shlegeris, B., Steinhardt, J. (2022).
  *Interpretability in the Wild: a Circuit for Indirect Object Identification in
  GPT-2 Small.* The standard for faithfulness, completeness and minimality, and
  for mean ablation over a reference distribution.
- Conmy, A., Mavor-Parker, A., Lynch, A., Heimersheim, S., Garriga-Alonso, A.
  (2023). *Towards Automated Circuit Discovery for Mechanistic Interpretability.*

**Why the ablation here is reported two ways**

- Zhang, F., Nanda, N. (2024). *Towards Best Practices of Activation Patching in
  Language Models: Metrics and Methods.* Directly on why zero ablation misleads,
  and the reason the mean-ablation figure is the one quoted.
- Chan, L., Garriga-Alonso, A., Goldowsky-Dill, N., et al. (2022). *Causal
  Scrubbing.* Resample ablation, the stricter intervention not run here.
- McGrath, T., Rahtz, M., Kramar, J., Mikulik, V., Legg, S. (2023). *The Hydra
  Effect: Emergent Self-repair in Language Model Computations.* Why ablation
  damage can be masked by downstream compensation, which this repo does not test.

**On the "in-context learning" label**

- Crosbie, J., Shutova, E. (2024). *Induction Heads as an Essential Mechanism for
  Pattern Matching in In-context Learning.* Measures induction-head ablation on
  natural pattern-matching tasks. The 85.4% here is repeated-random-token copying,
  which is a different and easier quantity.
- Hendel, R., Geva, M., Globerson, A. (2023). *In-Context Learning Creates Task
  Vectors*; Todd, E., et al. (2024). *Function Vectors in Large Language Models.*
  Partly independent accounts of few-shot ICL, which is why this document says
  induction heads are *a leading mechanistic account* rather than *the* mechanism.

**On circuit formation, relevant to the hypothesis in the README**

- Singh, A. K., Moskovitz, T., Hill, F., Chan, S. C. Y., Saxe, A. (2024). *What
  needs to go right for an induction head? A mechanistic study of in-context
  learning circuits and their formation.* The closest existing answer to the
  robustness question this repo raises.
- Chan, S. C. Y., Santoro, A., Lampinen, A., et al. (2022). *Data Distributional
  Properties Drive Emergent In-Context Learning in Transformers.* The known
  **negative** result: ICL fails to emerge when burstiness and Zipfian marginals
  are removed. So the honest positioning is robust along the optimizer and
  context-length axes this run degraded, fragile along the data-distribution axis
  it did not touch.
- Bietti, A., Cabannes, V., Bouchacourt, D., Jegou, H., Bottou, L. (2023).
  *Birth of a Transformer: A Memory Viewpoint.*
- Nanda, N., Bloom, J. (2022). *TransformerLens.* The reference implementation of
  the induction score, for comparing the 29.5x reported here against published
  values.
