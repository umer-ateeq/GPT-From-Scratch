"""Model and training configuration.

Every value here is the one that produced the released checkpoint, except where
a comment says otherwise. One importable place, read by every script, so that no
two parts of the codebase can disagree about what the run was.
"""

# Architecture of the released checkpoint.
#
# Named for its actual parameter count, 134,077,440: 8 layers rather than
# GPT-2-small's 12, and an output head that is not tied to the input embedding.
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


# Hyperparameters of the run that produced the released checkpoint. These are the
# values that actually reached the model, recovered from the weights where the
# weights carry them (see interp/CHECKPOINT_AUDIT.md).
TRAIN_CONFIG = {
    "batch_size": 32,                    # sequences per micro-batch
    "block_size": 128,                   # tokens per sequence, the REAL context used
    "gradient_accumulation_steps": 1,    # every micro-batch was an optimizer step
    "num_batches_per_epoch": 1000,       # micro-batches in one "epoch" of random windows
    "num_epochs": 1,                     # the training cell was re-run repeatedly
    "learning_rate": 4e-4,               # constant for the whole run, no schedule
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "eval_freq": 100,                    # optimizer steps between evaluations
    "eval_iter": 10,                     # batches averaged per loss estimate
    "precision_dtype": "float16",        # mixed precision via torch.amp
}

# The learning rate above is CONSTANT for the entire run. This matters beyond
# reproducibility: the weight-decay clock in interp/audit_checkpoint.py recovers
# the optimizer step count only because the rate never varied, so under a
# schedule the same measurement returns an integral rather than a count.
# train.py implements warmup and cosine decay for future runs.

# Hardware the released checkpoint was trained on. Recorded here because
# throughput and memory figures are meaningless without it, and because the
# original run logged nothing at all.
HARDWARE = {
    "gpu": "NVIDIA Tesla P100-PCIE-16GB",
    "provider": "Kaggle, free tier",
    "vram_gb": 16,
    "measured_tokens_per_sec": 10_200,   # steady state at batch 32 x 128, fp16
    "measured_peak_memory_gb": 12.24,    # 77% of the card
    "notes": "P100 supports full-rate fp16 but not bfloat16, and its compute "
             "capability 6.0 is below the 7.0 that torch.compile's Triton "
             "backend requires. Hence --dtype float16 and no compilation. "
             "Kaggle's preinstalled PyTorch is also built for sm_70+ and cannot "
             "launch kernels on this card at all; torch==2.5.1+cu121 still ships "
             "Pascal kernels and works.",
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
