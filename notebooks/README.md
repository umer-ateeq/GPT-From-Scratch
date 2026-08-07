# The original notebook

`original_colab_training.ipynb` is the Colab notebook that actually produced the
released checkpoint. It is kept here **unedited, bugs included**, because
[../docs/AUDIT.md](../docs/AUDIT.md) cites specific cells in it as evidence.
Cleaning it up would destroy the audit trail.

**Do not train with it.** It contains four known bugs:

| Cell | Bug |
|---|---|
| 9 and 27 | `get_batch` reads `batch_size` and `block_size` as globals; cell 27 rebinds both, so the run silently used 32 x 128 instead of the configured 64 x 256 |
| 27 and 33 | the warmup-plus-cosine scheduler is built around one optimizer, and cell 33 passes a different one to the training loop, so the schedule never reaches the model |
| 27 | the cosine floor `min_lr=5e-4` sits five times above the peak `learning_rate=1e-4` |
| everywhere | no run records a config, seed, commit, throughput or loss history |

The package in the parent directory is this notebook reorganized into modules
with all four fixed and each covered by a test. Every difference is listed in
[../docs/CHANGES_FROM_NOTEBOOK.md](../docs/CHANGES_FROM_NOTEBOOK.md).

The notebook also contains the two tokenization paths that built the corpora:
FineWeb-Edu streaming (cells 4 and 5) and TinyStories (cells 7 and 8). Both are
now in [../tokenize_data.py](../tokenize_data.py), which takes a `--dataset`
flag instead of having them as separate cell blocks.
