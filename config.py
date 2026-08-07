"""Model and training configuration.

Every value here is the one that produced the released checkpoint, except where
a comment says otherwise. Keeping the configuration in one importable place is
the single change that would have prevented the most serious bug in the original
training run: the notebook defined `batch_size` and `block_size` twice, in two
different cells, and the batch sampler silently read the second pair. See
docs/AUDIT.md.
"""

# Architecture of the released checkpoint.
#
# Naming note: the original notebook called this dict GPT_CONFIG_124M because it
# was modelled on GPT-2-small. The actual parameter count is 134,077,440, since
# this model has 8 layers rather than GPT-2-small's 12 and does not tie the
# output head to the input embedding. It is renamed here to match reality;
# tests/test_model.py asserts the exact count.
GPT_CONFIG_134M = {
    "vocab_size": 50257,      # GPT-2 BPE vocabulary (tiktoken "gpt2")
    "context_length": 256,    # positions the model allocates embeddings for
    "emb_dim": 768,           # model width, d_model
    "n_heads": 12,            # 768 / 12 = 64 dimensions per head
    "n_layers": 8,            # transformer blocks
    "drop_rate": 0.1,         # dropout on embeddings, attention and residuals
    "qkv_bias": False,        # no bias on the Q/K/V projections
}


# Hyperparameters of the run that produced the released checkpoint.
#
# These are the values that actually reached the model. The original notebook
# also contained a second, unused set (peak lr 1e-4, betas 0.9/0.95, a
# warmup-plus-cosine schedule) that was attached to an optimizer the training
# cell then replaced, so it never affected training. docs/AUDIT.md explains it.
TRAIN_CONFIG = {
    "batch_size": 32,                    # sequences per micro-batch
    "block_size": 128,                   # tokens per sequence, the REAL context used
    "gradient_accumulation_steps": 32,   # micro-batches per optimizer step
    "num_batches_per_epoch": 1000,       # micro-batches in one "epoch" of random windows
    "num_epochs": 1,                     # the training cell was re-run repeatedly
    "learning_rate": 4e-4,               # flat: see AUDIT.md, the schedule never applied
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "eval_freq": 100,                    # optimizer steps between evaluations
    "eval_iter": 10,                     # batches averaged per loss estimate
    "precision_dtype": "float16",        # mixed precision via torch.amp
}


# Where the tokenized corpora live. These are uint16 memory-mapped token files
# produced by tokenize_data.py, not text.
DATA_CONFIG = {
    "train_bin": "train.bin",
    "val_bin": "validation.bin",
    "dataset": "HuggingFaceFW/fineweb-edu",
    "dump": "CC-MAIN-2024-10",   # the Common Crawl snapshot used for training
    "max_tokens": 8_000_000_000,  # size of the corpus written to disk
}


def describe(cfg=GPT_CONFIG_134M):
    """Human-readable one-liner, used by the scripts when they start up."""
    return (f"{cfg['n_layers']}L / {cfg['n_heads']}H / {cfg['emb_dim']}d, "
            f"context {cfg['context_length']}, vocab {cfg['vocab_size']}")
