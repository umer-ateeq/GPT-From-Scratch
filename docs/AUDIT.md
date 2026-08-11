# Audit of the released checkpoint

`weights8b_300epoch.pth` is a bare `state_dict`: 117 tensors, no config, no
optimizer state, no seed, no metadata. The training notebook recorded nothing
about the run that produced it.

Before reporting any number from it, I reconstructed what actually happened
during that run from the weights themselves. Four bugs turned up. **An
adversarial review of this repository later found two more, one of which
invalidated a conclusion already published here** (bugs 5 and 6). Each one below
is provable either from the notebook source, preserved unedited at
[../notebooks/original_colab_training.ipynb](../notebooks/original_colab_training.ipynb),
or from the checkpoint, and each is reproducible by anyone who clones this repo.

The bugs changed the headline training scale by 4x. An earlier version of my CV
quoted the pre-audit figures. Finding them, correcting the numbers publicly, and
building the infrastructure that makes them impossible to repeat is the actual
engineering content of this project.

Bugs 5 and 6 are kept here in full rather than folded silently into the others,
because the failure they represent, publishing a confident conclusion built on an
unchecked assumption, is exactly the failure this document exists to catch. The
audit missing two instances of its own headline bug class is the most useful thing
in it.

Reproduce the weight-based half in one command:

```bash
python audit_checkpoint.py --ckpt weights8b_300epoch.pth
```

---

## Bug 1: the run used batch 32 x context 128, not the configured 64 x 256

`GPT_CONFIG_124M` set `context_length: 256` and `batch_size: 64`, and the batch
sampler read module-level globals at call time:

```python
# cell 9
block_size = GPT_CONFIG_124M['context_length']   # 256
batch_size = GPT_CONFIG_124M['batch_size']       # 64

def get_batch(split):
    ...
    ix = torch.randint(len(data) - block_size, (batch_size,))   # reads the globals, live
```

Eighteen cells later, the hyperparameter cell rebound both names:

```python
# cell 27
batch_size = 32   # changed from 16, better gradient estimate
block_size = 128  # changed from 64, capture longer range dependencies
```

`get_batch` closes over the *names*, not the values, so from that point every
batch it produced was 32 x 128. The config object still said 64 x 256, and so
did every summary derived from it.

### The weights prove it independently

`pos_emb.weight` holds one row per position, and a row only receives gradient
when a training batch is at least that long. AdamW's weight decay shrinks every
parameter it touches on every step, so rows that were never in a batch decay
toward zero while trained rows keep their norm. Output of `audit_checkpoint.py`:

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

VERDICT: positions 0-127 are trained (mean norm 0.2175); positions 128-255
never received gradient (mean norm 0.002346).
The trained rows carry 93x the norm of the dead rows.
Effective training context = 128, not the configured 256.
```

![Positional embedding row norms](images/pos_emb_norms.png)

A 93x cliff exactly at 128, on a log scale. The source bug and the weights agree.
The plot is regenerated from the checkpoint by the same command:

```bash
python audit_checkpoint.py --ckpt weights8b_300epoch.pth --plot docs/images/pos_emb_norms.png
```

### Consequence for the token count

```
300 cycles x 1000 batches/cycle x 32 rows x 128 tokens = 1.23B tokens
```

not the 4.92B the configured 64 x 256 would have produced. Bug 5 shows gradient
accumulation never ran, so this is 300,000 optimizer steps of 4,096 tokens, and
the weight-decay clock in [RESULTS.md](RESULTS.md) independently measures 234,478
successful steps, or ~0.96B tokens. **Two methods, ~1B tokens, no conflict.**
Note that even the
configured setup would not have reached the 7B this project was once described
with: the **8B figure is the size of the tokenized corpus** that
`tokenize_data.py` wrote to disk, not the number of tokens the model consumed.
Batches were random windows over that corpus, so the model saw roughly 15% of it.

### Consequence for perplexity

The model must be evaluated at context 128. Scoring it at 256 places half of
every window on positional rows it has never seen, and perplexity roughly
doubles: **38.89 at context 128 against 74.70 at context 256** on identical
held-out data. The 128 figure is the model's real capability; the 256 figure is
an artefact of the bug. `evaluate.py` detects this from the weights and warns
before printing a number.

---

## Bug 2: the learning-rate schedule was attached to a discarded optimizer

Cell 27 builds an optimizer and wraps it in warmup-then-cosine:

```python
# cell 27
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                              betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9)
scheduler_warmup = LinearLR(optimizer, total_iters=warmup_steps)
scheduler_decay  = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps,
                                     eta_min=min_lr)
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay],
                         milestones=[warmup_steps])
```

The training cell then builds a **second** optimizer and passes it alongside the
schedule from cell 27:

```python
# cell 33
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

train_losses, val_losses, tokens_seen = train_model(
    model=model,
    optimizer=optimizer,     # the new one
    scheduler=scheduler,     # still holds a reference to the OLD one
    ...)
```

A PyTorch scheduler mutates the `param_groups` of the optimizer it was
constructed with. That optimizer is now unreachable garbage. Every
`scheduler.step()` inside the training loop dutifully computed a learning rate
and wrote it into an object that no longer touched the model.

So the released checkpoint was **not** trained with warmup or cosine decay. It
trained at a flat 4e-4, with default betas `(0.9, 0.999)` rather than the
intended `(0.9, 0.95)`, because the surviving optimizer is the bare one from
cell 33. Any claim about learning-rate scheduling based on this checkpoint would
be a claim about code that ran but had no effect.

---

## Bug 3: the cosine floor sat above the peak learning rate

```python
# cell 27
learning_rate = 1e-4   # more stable training, earlier 1e-4
min_lr        = 5e-4   # lower rate, earlier 5e-4
```

`min_lr` is passed as `eta_min`, the value cosine annealing decays *down* to.
Here it is five times **larger** than the peak it decays from, so the schedule
would have climbed rather than decayed. Because of bug 2 it never reached the
model, which is the only reason it did no damage. Two bugs cancelling is not a
working schedule.

---

## Bug 5: gradient accumulation never ran, and the audit missed it for weeks

This one was found by an adversarial review of this repository, not by the
original audit, and it matters more than any of the others because a wrong
conclusion had already been published on top of it.

The hyperparameter cell sets a module-level global:

```python
# notebook cell 27
gradient_accumulation_steps = 32   # reduced from 50
```

The training function declares a parameter with the same name and a default:

```python
# notebook cell 30
def train_model(model, optimizer, scheduler, device, num_epochs,
                num_batches_per_epoch, eval_freq, eval_iter, start_context,
                tokenizer, gradient_accumulation_steps=1, precision_dtype=torch.float16):
```

The parameter shadows the global inside the function body. And the call site never
passes it:

```python
# notebook cell 33
train_model(model=model, optimizer=optimizer, scheduler=scheduler, device=device,
            num_epochs=num_epochs,
            num_batches_per_epoch=GPT_CONFIG_124M["num_batches_per_epoch"],
            eval_freq=100, eval_iter=10,
            start_context="I am a language model, who is ", tokenizer=tokenizer)
#           ^ no gradient_accumulation_steps argument
```

So `gradient_accumulation_steps` was **1** for the entire run. Since
`(batch_idx + 1) % 1 == 0` is always true, **every micro-batch triggered an
optimizer step**. There was no accumulation: 32 sequences of 128 tokens went into
each step, 4,096 tokens, not 131,072.

**This is bug 1 again, three lines below it in the same cell.** The audit caught
the `batch_size` and `block_size` instance of the shadowing and walked straight
past the `gradient_accumulation_steps` instance.

### What it invalidated

An earlier version of this repository used the weight-decay clock to argue that
the checkpoint had taken ~234,000 optimizer steps against the ~9,300 the filename
implied, called that a 25x discrepancy, derived a range of 12B to 31B tokens, and
presented the whole thing as an unresolved conflict between the weights and the
model's quality.

Every part of that was downstream of assuming 131,072 tokens per step. At the real
4,096:

| Method | Optimizer steps | Tokens |
|---|---|---|
| Weight-decay clock | 234,478 | 0.96B |
| 300 cycles x 1000 batches | 300,000 | 1.23B |

**A second, independent clock confirms it.** 517 rows of `tok_emb.weight` sit at
the same pure-decay floor (minimum norm 0.002150): tokens that never appeared in
any sampled batch, so like the dead positional rows they received decay and no
gradient. They give N = 227,000 to 236,600, bracketing the positional clock's
234,478. Two disjoint parameter subspaces, same answer.

The two methods **agree to within 22%**, and the residual is what `GradScaler`
step-skipping on fp16 overflow produces: a skipped step applies neither the Adam
update nor the weight decay, so the clock counts successful steps and is a lower
bound. There was never a conflict. There was an arithmetic error dressed as
intellectual honesty, which is worse than an ordinary mistake, and it is recorded
here rather than quietly corrected.

### Consequences elsewhere

- `config.py` listed `gradient_accumulation_steps: 32` under a comment asserting
  "these are the values that actually reached the model". Corrected to 1.
- The README described gradient accumulation as one of the techniques that made
  the run fit in 16 GB. It did not run. Activation memory is set by the
  micro-batch, which was 32 either way, so the memory argument was wrong
  independently of the accumulation factor.
- `train.py` implements accumulation correctly and it does work there. The claim
  is now scoped to `train.py` rather than to the released checkpoint.

## Bug 6: gradients were clipped while still scaled

Also found by review, not by the original audit.

```python
# notebook cell 30
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)     # <- unscaling happens in here, AFTER the clip
```

`GradScaler` multiplies the loss by a large factor (typically 65536) before
backward so fp16 gradients do not underflow, then divides it out inside
`scaler.step()`. The notebook never calls `scaler.unscale_(optimizer)`, so
`clip_grad_norm_` operated on the **scaled** gradients and normalized them to unit
norm in the scaled domain. `scaler.step()` then divided by the scale again, so the
gradient that actually reached the optimizer was roughly `g / (S * ||g||)`.

Adam is largely scale-invariant, so this is bounded rather than fatal. The damage
runs through `eps`: cell 33's optimizer takes the default `eps=1e-8`, and
per-coordinate gradient magnitudes after the double shrink land near 1e-9, below
eps, which suppresses the effective step size in a parameter-dependent way. That
is a better-grounded explanation for slow progress than "a flat learning rate
never lets the model settle".

`train.py` calls `scaler.unscale_(optimizer)` before clipping. The fix was made
while rewriting the loop and was never written down until this review.

## Bug 4: nothing about the run was recorded

No config, no seed, no git commit, no throughput, no peak memory, no loss curve
survived. The filename `weights8b_300epoch.pth` is the entire provenance record,
and two of its three claims are wrong: `8b` refers to corpus size rather than
tokens trained, and "epoch" means one 1000-batch cycle of random windows rather
than a pass over the data.

Bugs 1 to 3 were each invisible for the same reason. Nothing compared what was
*configured* against what actually *ran*.

---

## What changed as a result

| Failure mode | What now prevents it |
|---|---|
| Globals shadowed the configured batch shape | `data.get_batch` takes the shape as arguments, `config.json` records it, and `tokens_seen` accumulates `x.numel()` from the tensors that reached the model |
| Scheduler bound to a dead optimizer | No scheduler object. `get_lr(step)` returns a float that the loop writes into the live optimizer, so there is nothing to desynchronize |
| Inverted cosine floor | `min_lr` defaults to `lr / 10`, derived from the peak rather than typed separately |
| No provenance | Every run writes `config.json` (args, model config, git commit, seed, library versions, GPU name, launch command), `metrics.jsonl`, `summary.json` and a loss curve. Checkpoints embed their own config |

Each bug also has a test in `tests/test_model.py`, so a regression fails CI
rather than silently costing another training run.

## Honest scorecard for this checkpoint

| Claim | Status |
|---|---|
| 134M parameters | **Correct.** 134,077,440 trainable, verified by counting loaded tensors |
| 8 layers, 12 heads, 768 wide | **Correct**, recovered from tensor shapes |
| Trained on 7B tokens | **Wrong as stated.** The 8B figure is corpus size, not tokens consumed. One documented session gives 1.23B, which is a floor; the total is not determined by the evidence. See [RESULTS.md](RESULTS.md) |
| WikiText-2 perplexity 31.23 | **Wrong.** That number was `exp(val_loss)` on the in-distribution validation split during training, not WikiText-2. The real figure is 184.96, against 59.69 for GPT-2-small on the identical harness. 31.23 was never reachable at this scale: it is below GPT-2-small's own published score |
| Trained with cosine decay and linear warmup | **Wrong for this checkpoint** (bug 2). Correct for `train.py`, where it is implemented, logged and tested |
| Mixed precision, gradient accumulation, gradient clipping, async host-to-device batching | **Correct.** Present in the notebook and in `train.py` |
| Throughput 75K tokens/sec | **Wrong, by about 7x.** It was never measured; no instrumented run existed behind it. The real figure on the same hardware and batch shape is **10,200 tokens/sec**, with peak memory 6.1 GB of 16 GB. See [RESULTS.md](RESULTS.md) |
