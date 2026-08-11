# What is what in this repository

A guide to reading the code, and an honest split between the parts written
during the original project and the parts added afterwards when it was packaged
and audited.

## Written for the original training run, kept verbatim

These are unchanged from the training notebook, character for character. They are
what the released checkpoint was trained with, and changing them would mean the
published weights no longer match the published code.

| File | Contents | Lines |
|---|---|---|
| [../model.py](../model.py) | `MultiHeadAttention`, `FeedForward`, `LayerNorm`, `TransformerBlock`, `GPTModel` | 133 |
| [../generate.py](../generate.py) | `generate`, `text_to_token_ids`, `token_ids_to_text` | ~55 |
| [../train.py](../train.py) | `calc_loss_batch`, `estimate_loss`, and the body of `train_model` | ~90 |
| [../data.py](../data.py) | `get_batch` | ~15 |

`model.py`'s five classes are verified verbatim against
[../notebooks/original_colab_training.ipynb](../notebooks/original_colab_training.ipynb).
Their comments, spacing and variable names are the originals.

**Two deliberate exceptions**, both in `get_batch` and both explained in
[CHANGES_FROM_NOTEBOOK.md](CHANGES_FROM_NOTEBOOK.md):

1. `batch_size`, `block_size` and the file path are arguments rather than module
   globals. This is the fix for bug 1 in [AUDIT.md](AUDIT.md), where a later cell
   reassigned those globals and silently changed the batch shape.
2. The random start offset is bounded by `len(data) - block_size - 1`, one lower
   than the original, because the target window reaches one token further than
   the input window.

## Rewritten when the notebook became a package

| File | Why it differs | What to know |
|---|---|---|
| [../train.py](../train.py) | The notebook's training loop carried three of the four audit bugs | `train_model` keeps the notebook's structure: same nested epoch / batch loop, same accumulate-then-step pattern, same `Ep N (Step NNNNNN): Train loss ..., Val loss ..., Perplexity ..., LR ...` output, same per-epoch sample generation. `calc_loss_batch` and `estimate_loss` are the originals. See the four inline fixes below |
| [../config.py](../config.py) | The notebook set hyperparameters in two different cells, which is what allowed bug 1 | One dict per concern. `GPT_CONFIG_124M` is renamed `GPT_CONFIG_134M` because the real parameter count is 134,077,440 |
| [../tokenize_data.py](../tokenize_data.py) | **Rewritten, not the notebook's code.** The notebook's `process_streaming_dataset` wrote into a pre-allocated 8B-token memmap, then copied the whole thing into a second file to trim it, which needs twice the disk. This version truncates in place and takes the dataset, crawl and token budget as flags instead of having them hard-coded in two separate cells | Same approach: stream the dataset, tokenize with GPT-2 BPE, write uint16 into a memmap. Different code |
| [../evaluate.py](../evaluate.py) | The notebook had no held-out evaluation, only training-time loss | Two perplexity protocols, plus GPT-2-small run through the identical function as a baseline |

### The four changes inside `train_model`

Everything else in that function is the notebook's. If you read one thing in
`train.py`, read these four:

1. **`get_batch(train_bin, batch_size, block_size, device)`** instead of
   `get_batch("train")`. The sampler no longer reads globals, so nothing can
   silently change the batch shape. **Bug 1.**
2. **`lr_schedule` is a function, not a scheduler object.** Two lines set the
   learning rate on the live optimizer each step. The notebook passed a scheduler
   built around an optimizer the training cell had already replaced, so it
   updated an object the model never saw. **Bug 2.**
3. **`min_lr` defaults to `lr / 10`** in `main`, derived from the peak instead of
   typed separately. The notebook had a floor five times above its peak. **Bug 3.**
4. **`tokens_seen += input_batch.numel()`** on every micro-batch, counted from the
   tensor that actually went through the model. The notebook multiplied by
   `gradient_accumulation_steps` inside the accumulation branch, which happened to
   be right, but was computed rather than observed. Plus `logger.log(...)` at each
   evaluation. **Bug 4.**

Two additions that are not bug fixes: `compute_mfu` turns tokens/sec into Model
FLOPs Utilization so throughput is comparable across GPUs, and `--compile` /
`--fused-adam` are optional speed flags that skip themselves with a printed
reason on hardware that cannot use them.

## Added afterwards: the analysis

Neither of these existed during training. They examine the finished checkpoint.

| File | What it does | Core idea in one sentence |
|---|---|---|
| [../audit_checkpoint.py](../audit_checkpoint.py) | Recovers the architecture and the true training context from the weights alone | A positional embedding row only gets gradient if some batch was long enough to reach it, and weight decay shrinks the rest toward zero, so the trained context is visible as a cliff in the row norms |
| [../induction_heads.py](../induction_heads.py) | Finds which attention heads implement in-context copying | Feed the model a random sequence repeated twice; an induction head sitting on the second copy attends back to the token that followed the same token in the first copy |
| [../ablate_heads.py](../ablate_heads.py) | Tests whether those heads are causally responsible | Zero a head's slice of the attention output and see how much worse the model gets at predicting the repeated half, with other heads as controls |

`induction_heads.py` deliberately does **not** modify `model.py`. It hooks each
attention module to capture the tensor going into it, then recomputes the same
softmax using that module's own `W_query` and `W_key`. The arithmetic is copied
line for line from `MultiHeadAttention.forward`. That is why the model file
could stay verbatim.

## Reading order, if you have an hour

1. **[../model.py](../model.py)**, top to bottom. This is the whole transformer
   and it is the original code. Around 130 lines.
2. **[AUDIT.md](AUDIT.md)**, sections "Bug 1" and "Bug 2". The evidence for both
   is short and the plot makes bug 1 obvious.
3. **[../induction_heads.py](../induction_heads.py)**, the module docstring, then
   `induction_scores`. The measurement is about 15 lines; the rest is printing.
4. **[INDUCTION_HEADS.md](INDUCTION_HEADS.md)**, especially the limitations
   section, so you can say what the result does *not* show.

## The three questions most likely to be asked

**"How does the attention work?"** Every token makes a query, a key and a value.
Scores are query-key dot products, scaled by `1/sqrt(head_dim)` so the softmax
does not saturate. The causal mask sets future positions to `-inf` so they get
zero weight after the softmax, which is what makes next-token prediction a valid
objective. 768 dimensions are split into 12 heads of 64, computed as one batched
matmul rather than a loop. It is all in `MultiHeadAttention.forward`.

**"How did you find the context bug?"** The config said 256 but the model was
producing odd results past 128. `pos_emb.weight` has one row per position, and a
row only receives gradient if a batch reached it. AdamW's weight decay shrinks
everything it touches, so untrained rows decay toward zero. Plotting the row
norms shows a 93x cliff exactly at 128. Then the notebook confirmed it: a cell
eighteen cells after `get_batch` was defined rebound the globals it read.

**"What is an induction head and what did you find?"** A head that implements
"I have seen this token before, so attend to whatever came after it last time".
It is thought to underlie in-context learning. I fed the model random sequences
repeated twice and measured, for each of the 96 heads, how much attention went to
that target. Two heads stand out, L6.H9 at 29.5x uniform and L7.H8 at 19.3x, and
both are in the last two layers, which is expected because the circuit needs a
previous-token head in an earlier layer to compose with. It measures attention,
not causal contribution; ablating the head and measuring the loss change is the
next step, and it is done in `ablate_heads.py`.
