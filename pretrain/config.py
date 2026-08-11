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
    # BUG 5 (docs/AUDIT.md): the notebook set 32, but train_model defaulted this
    # to 1 and the call site never passed it, so accumulation never ran. Every
    # micro-batch was an optimizer step. This is the value that reached the model.
    "gradient_accumulation_steps": 1,
    "num_batches_per_epoch": 1000,       # micro-batches in one "epoch" of random windows
    "num_epochs": 1,                     # the training cell was re-run repeatedly
    "learning_rate": 4e-4,               # flat: see AUDIT.md, the schedule never applied
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "eval_freq": 100,                    # optimizer steps between evaluations
    "eval_iter": 10,                     # batches averaged per loss estimate
    "precision_dtype": "float16",        # mixed precision via torch.amp
}

# The learning rate above is FIXED. No schedule was applied to the released
# checkpoint: the notebook built one but bound it to an optimizer the training
# cell then replaced, so it never reached the weights (docs/AUDIT.md, bug 2).
# train.py implements warmup and cosine decay correctly for future runs.

# Hardware the released checkpoint was trained on. Recorded here because
# throughput and memory figures are meaningless without it, and because the
# original run logged nothing at all.
HARDWARE = {
    "gpu": "NVIDIA Tesla P100-PCIE-16GB",
    "provider": "Kaggle, free tier",
    "vram_gb": 16,
    "measured_tokens_per_sec": 10_200,   # steady state at batch 32 x 128, fp16
    "measured_peak_memory_gb": 6.1,      # 38% of the card; see docs/RESULTS.md
    "notes": "P100 supports full-rate fp16 but not bfloat16, and its compute "
             "capability 6.0 is below the 7.0 that torch.compile's Triton "
             "backend requires. Hence --dtype float16 and no compilation. "
             "Kaggle's preinstalled PyTorch is also built for sm_70+ and cannot "
             "launch kernels on this card at all; torch==2.5.1+cu121 still ships "
             "Pascal kernels and works. See docs/MEASURE.md.",
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
