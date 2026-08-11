# Architecture

A decoder-only transformer in the GPT-2 style, written from scratch. Every
component below is implemented in [../model.py](../model.py) with no transformer
library involved.

## Specification

| Component | Choice | Parameters |
|---|---|---|
| Token embedding | 50257 x 768 | 38,597,376 |
| Positional embedding | learned, 256 x 768 | 196,608 |
| Transformer blocks | 8 | 56,682,240 |
| Final LayerNorm | scale + shift | 1,536 |
| Output head | 768 x 50257, no bias, **untied** | 38,597,376 |
| | **Total trainable** | **134,077,440** |
| Causal masks | 8 x 256 x 256, buffers, not trained | 524,288 |

`tests/test_model.py::test_parameter_count_is_exactly_134M` asserts these
numbers, so they cannot drift from the code.

| Hyperparameter | Value |
|---|---|
| Model width (d_model) | 768 |
| Attention heads | 12, head dimension 64 |
| Feed-forward hidden | 3072 (4x expansion), ReLU |
| Normalization | custom LayerNorm, learned scale and shift, **pre-norm** |
| Dropout | 0.1 on embeddings, attention weights and residual branches |
| Context length | 256 allocated, **128 actually trained** (see [AUDIT.md](AUDIT.md)) |
| Vocabulary | 50257, GPT-2 BPE via `tiktoken` |

## Data flow

```
token ids                      (batch, tokens)
   |
   +-- token embedding         (batch, tokens, 768)
   +-- positional embedding    (tokens, 768), broadcast and added
   |
  dropout
   |
   +--> TransformerBlock 1 ----+
   |      LayerNorm            |
   |      MultiHeadAttention   |  residual
   |      dropout              |
   |    <----------------------+
   |      LayerNorm            |
   |      FeedForward          |  residual
   |      dropout              |
   |    <----------------------+
   |
   ... 8 blocks total ...
   |
final LayerNorm
   |
output head (768 -> 50257)     (batch, tokens, 50257)
```

Logits at position `i` are the model's prediction for the token at position
`i + 1`. The loss is cross-entropy over every position at once, so a batch of
32 x 128 contributes 4096 independent next-token predictions.

## Why each piece is there

**Multi-head attention.** Every token forms a query, a key and a value. Scores
are dot products of queries against keys, divided by `sqrt(head_dim)` so the
softmax does not saturate as the head dimension grows. Splitting 768 dimensions
into 12 heads of 64 lets different heads specialize on different relationships
in parallel, at the same cost as one 768-wide head.

**The causal mask.** Every score for a future position is set to `-inf` before
the softmax, so those positions receive exactly zero weight. This is what makes
next-token prediction a valid objective: without it the model could simply read
the token it is asked to predict, and the loss would collapse to nothing while
the model learned nothing. `test_attention_cannot_see_the_future` verifies it by
changing the final token and asserting no earlier position's logits move.

**Pre-norm residuals.** Normalization is applied *before* each sublayer rather
than after, so the residual path from input to output is never normalized. This
keeps gradients well behaved through 8 stacked blocks. The original post-norm
transformer needs careful warmup to train at all; pre-norm is what made deep
stacks routine.

**Untied output head.** The input embedding and the output projection are
separate 50257 x 768 matrices. Tying them, as GPT-2 does, would save 38.6M
parameters and put this model near 96M. Keeping them separate gives the model
freedom to represent a token differently as an input than as a prediction
target, at the cost of parameters.

**Learned positional embeddings.** Attention is permutation-invariant on its own,
so position has to be injected explicitly. This model adds a learned vector per
absolute position, which is simple and is what GPT-2 did. Its weakness is that
it cannot extrapolate past the trained positions, which is precisely what made
the context bug in [AUDIT.md](AUDIT.md) both possible and detectable.

## Scope

This is a GPT-2-era architecture, implemented directly and trained end to end.
Every component above is the one the released checkpoint was trained with, so the
published weights and the published code always match.
