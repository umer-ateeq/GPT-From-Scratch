# What changed between the notebook and this package

The model was trained by the Colab notebook preserved unedited at
[../notebooks/original_colab_training.ipynb](../notebooks/original_colab_training.ipynb).
This package is that notebook reorganized into importable modules.

**The maths is unchanged.** Attention, the feed-forward block, layer
normalization, the loss, the sampler and the training loop are the same
implementations, moved rather than rewritten. `tests/test_model.py` asserts the
architecture still produces exactly 134,077,440 parameters and still loads the
original checkpoint with `strict=True`, which would fail if any layer had been
altered.

Everything that did change is listed below. Nothing is omitted.

## Structural: cells to modules

| Notebook cell | Now lives in |
|---|---|
| 2, 27 configuration dicts and hyperparameter cell | `config.py` |
| 4, 5 FineWeb-Edu streaming tokenization | `tokenize_data.py` |
| 7, 8 TinyStories tokenization | `tokenize_data.py` via `--dataset roneneldan/TinyStories` |
| 9 `get_batch` | `data.py` |
| 11-15 `MultiHeadAttention`, `FeedForward`, `LayerNorm`, `TransformerBlock`, `GPTModel` | `model.py` |
| 20, 21 `generate_text_simple`, `generate`, token/text helpers | `generate.py` |
| 24, 29 `calc_loss_batch`, `estimate_loss` | `train.py` |
| 30 `train_model` | `train.py` |
| 31 `plot_losses` | `train.py`, as `RunLogger._plot` |
| 33 the training invocation | `train.py` `main()` with command-line flags |

## Behavioural: four bug fixes

Each is explained in full, with its evidence, in [AUDIT.md](AUDIT.md).

### 1. The batch shape is passed, not read from a global

The notebook defined `batch_size` and `block_size` at cell 9, and `get_batch`
read those names at call time. Cell 27 rebound both:

```python
# cell 9
block_size = GPT_CONFIG_124M['context_length']   # 256
batch_size = GPT_CONFIG_124M['batch_size']       # 64

# cell 27, eighteen cells later
batch_size = 32
block_size = 128
```

From then on every batch was 32 x 128 while the config object still said
64 x 256. `data.get_batch(path, batch_size, block_size, device)` now takes both
explicitly, and `test_batch_shape_follows_arguments_not_globals` locks it in.

### 2. One optimizer instead of two

Cell 27 built an optimizer and wrapped it in a warmup-plus-cosine schedule.
Cell 33 then built a **second** optimizer and passed it to the training loop
alongside the schedule from cell 27, which still referenced the first one:

```python
# cell 33
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
train_model(model=model, optimizer=optimizer, scheduler=scheduler, ...)
#                        ^^ the new one      ^^ still points at the old one
```

A PyTorch scheduler mutates the `param_groups` of the optimizer it was
constructed with, so every `scheduler.step()` wrote a learning rate into an
object that no longer touched the model. The released checkpoint therefore
trained at a flat 4e-4 with default betas `(0.9, 0.999)`, not the intended
schedule and not the intended `(0.9, 0.95)`.

`train.py` replaces the scheduler object with a pure function, `get_lr(step)`,
whose return value is written into the live optimizer inside the loop. There is
no second object that can fall out of sync.

### 3. The cosine floor is derived, not typed

```python
# cell 27
learning_rate = 1e-4
min_lr        = 5e-4   # the floor is 5x ABOVE the peak
```

`eta_min` is the value cosine annealing decays *down* to, so this would have
been a climb. It never reached the model only because of bug 2. `train.py`
defaults `min_lr` to `lr / 10`, derived from the peak so it cannot exceed it,
and `test_lr_warms_up_then_decays_and_never_climbs` asserts the schedule is
monotonically non-increasing after warmup.

### 4. Runs are recorded

The notebook saved no configuration, seed, git commit, throughput, or loss
history. `weights8b_300epoch.pth` is a bare `state_dict` whose only provenance
is its filename, and two of that filename's three claims are wrong. Bugs 1 to 3
were each invisible for the same reason: nothing compared what was configured
against what ran.

Every run now writes `config.json`, `metrics.jsonl`, `summary.json` and
`loss_curve.png` under `runs/`, and `tokens_seen` accumulates `x.numel()` from
the tensors that actually reach the model rather than being computed from the
config.

## Smaller corrections

| Change | Reason |
|---|---|
| `GPT_CONFIG_124M` renamed `GPT_CONFIG_134M` | The model has 134,077,440 parameters. The old name came from being modelled on GPT-2-small, which has 12 layers and a tied head; this has 8 layers and an untied head |
| `calc_loss_batch` takes `vocab_size` as an argument | It previously read `GPT_CONFIG_124M['vocab_size']` from a global, the same pattern that caused bug 1 |
| `get_batch` takes a file path rather than a `"train"`/`"val"` string | The notebook hard-coded `train.bin` and `validation.bin` inside the function |
| Training runs on CPU as well as GPU | The notebook's `autocast(device_type='cuda')` and bare `GradScaler()` crash without a GPU. Mixed precision is now enabled only when a CUDA device is present, so the test suite can run in CI |
| `torch.randint(len(data) - block_size - 1, ...)` | The notebook used `- block_size`, which can select a start offset whose *target* window runs one token past the end of the file |
| Unused parameters removed from `train_model` | `num_workers` and `Stride` in the config were never read by anything |
| Seeding | The notebook's `torch.manual_seed(123)` was commented out, so no run was reproducible. Seeding is now on by default and recorded in `config.json` |

## What was deliberately not changed

- **The attention implementation.** It computes scores, applies the causal mask,
  softmaxes and weights the values explicitly. Swapping in
  `F.scaled_dot_product_attention` would be faster, but the explicit version is
  the point of a from-scratch project and it is what the released checkpoint was
  trained with.
- **ReLU rather than GELU, learned positional embeddings rather than RoPE,
  LayerNorm rather than RMSNorm, untied output head.** These are the choices the
  checkpoint was trained under. Changing them would mean the published weights no
  longer match the published code.
- **The notebook itself**, which is kept unedited because [AUDIT.md](AUDIT.md)
  cites specific cells in it as evidence.
