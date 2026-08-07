# Audit of the released checkpoint

`weights8b_300epoch.pth` is a bare `state_dict`: 117 tensors, no config, no
optimizer state, no seed, no metadata. The training notebook recorded nothing
about the run that produced it.

Before reporting any number from it, I reconstructed what actually happened
during that run from the weights themselves. Four bugs turned up. Each one below
is provable either from the notebook source, preserved unedited at
[../notebooks/original_colab_training.ipynb](../notebooks/original_colab_training.ipynb),
or from the checkpoint, and each is reproducible by anyone who clones this repo.

The bugs changed the headline training scale by 4x. An earlier version of my CV
quoted the pre-audit figures. Finding them, correcting the numbers publicly, and
building the infrastructure that makes them impossible to repeat is the actual
engineering content of this project.

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

A 93x cliff exactly at 128. The source bug and the weights agree.

### Consequence for the token count

```
300 cycles x 1000 batches/cycle x 32 rows x 128 tokens = 1.23B tokens
```

not the 4.92B the configured 64 x 256 would have produced. Note that even the
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
| Trained on 7B tokens | **Wrong. 1.23B.** The 8B figure is corpus size |
| WikiText-2 perplexity 31.23 | **Wrong.** That number was `exp(val_loss)` on the in-distribution validation split during training, not WikiText-2. The real figure is 184.96, against 59.69 for GPT-2-small on the identical harness. 31.23 was never reachable at this scale: it is below GPT-2-small's own published score |
| Trained with cosine decay and linear warmup | **Wrong for this checkpoint** (bug 2). Correct for `train.py`, where it is implemented, logged and tested |
| Mixed precision, gradient accumulation, gradient clipping, async host-to-device batching | **Correct.** Present in the notebook and in `train.py` |
| Throughput 75K tokens/sec | **Unmeasured.** No instrumented run exists. `train.py` logs `tokens_per_sec`; the number goes into RESULTS.md when a run produces it |
