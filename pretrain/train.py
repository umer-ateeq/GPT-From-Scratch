"""Pretraining: mixed precision, gradient accumulation, warmup + cosine decay.

The techniques here exist because of one constraint, a single 16 GB P100:

  Mixed precision (fp16)   halves activation memory and uses the GPU's faster
                           fp16 path. A GradScaler multiplies the loss before
                           backward so small gradients do not underflow to zero,
                           then unscales before the optimizer step.
  Gradient accumulation    micro-batches accumulate before one optimizer step,
                           giving the gradient quality of a large batch at the
                           memory cost of a small one. Available here via
                           --grad-accum. NOTE: it did NOT run for the released
                           checkpoint (docs/AUDIT.md bug 5).
  Gradient clipping        caps the global gradient norm at 1.0, so one bad batch
                           cannot destabilize training. scaler.unscale_() is
                           called FIRST so the clip sees real gradients; the
                           notebook clipped scaled ones (docs/AUDIT.md bug 6).
  Pinned async transfers   see data.py: host-to-device copies overlap with compute.

`train_model` below keeps the structure of the original notebook's training
function: the same nested epoch / batch loop, the same accumulate-then-step
pattern, the same evaluation checkpoints, the same per-epoch sample generation.
Four things inside it are fixed, each documented in docs/AUDIT.md:

  1. The batch shape is passed to get_batch instead of read from module globals,
     which is what silently changed the run from 64 x 256 to 32 x 128.
  2. There is no scheduler object. The notebook built one around an optimizer
     that the training cell then replaced, so every scheduler.step() updated an
     object that no longer touched the model. Here `lr_schedule(step)` returns a
     float that is written into the live optimizer, so nothing can desynchronize.
  3. The cosine floor is derived as lr/10 rather than typed separately. The
     notebook set min_lr=5e-4 against a peak of 1e-4, so its "decay" was a climb.
  4. Every run writes its config, metrics, summary and loss curve to runs/.
     The original run recorded nothing, which is why the three bugs above went
     unnoticed for months.

Usage:
    python train.py --train-bin train.bin --val-bin validation.bin \
        --train-tokens 100e6 --lr 4e-4 --run-name baseline
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext

import numpy as np
import tiktoken
import torch
from torch.amp import GradScaler, autocast

from config import GPT_CONFIG_134M, TRAIN_CONFIG
from data import get_batch
from generate import generate, text_to_token_ids, token_ids_to_text
from model import GPTModel


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--train-bin", default="train.bin")
    p.add_argument("--val-bin", default="validation.bin")
    # architecture (defaults reproduce the released checkpoint)
    p.add_argument("--n-layers", type=int, default=GPT_CONFIG_134M["n_layers"])
    p.add_argument("--n-heads", type=int, default=GPT_CONFIG_134M["n_heads"])
    p.add_argument("--emb-dim", type=int, default=GPT_CONFIG_134M["emb_dim"])
    p.add_argument("--context", type=int, default=GPT_CONFIG_134M["context_length"])
    p.add_argument("--dropout", type=float, default=GPT_CONFIG_134M["drop_rate"])
    # optimization
    p.add_argument("--batch-size", type=int, default=TRAIN_CONFIG["batch_size"])
    p.add_argument("--grad-accum", type=int,
                   default=TRAIN_CONFIG["gradient_accumulation_steps"])
    p.add_argument("--lr", type=float, default=TRAIN_CONFIG["learning_rate"],
                   help="peak learning rate, reached at the end of warmup")
    p.add_argument("--min-lr", type=float, default=None,
                   help="cosine floor; defaults to lr/10 so it cannot exceed the peak")
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=TRAIN_CONFIG["weight_decay"])
    p.add_argument("--grad-clip", type=float, default=TRAIN_CONFIG["grad_clip"])
    p.add_argument("--num-epochs", type=int, default=1,
                   help="outer loop, as in the original notebook")
    p.add_argument("--train-tokens", type=float, default=100e6,
                   help="total token budget. Sets num_batches_per_epoch, so runs "
                        "with different batch shapes stay comparable.")
    # evaluation and bookkeeping
    p.add_argument("--eval-freq", type=int, default=50,
                   help="optimizer steps between evaluations")
    p.add_argument("--eval-iter", type=int, default=20,
                   help="batches averaged per loss estimate")
    p.add_argument("--start-context", default="I am a language model, who is ",
                   help="prompt sampled at the end of each epoch")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"],
                   default="float16",
                   help="float16 on P100/T4/V100, bfloat16 on A100/H100, float32 on CPU")
    p.add_argument("--run-name", default="run")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--resume", default=None, help="path to a ckpt_last.pt to continue from")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model. Needs compute capability 7.0+; "
                        "skipped with a message on older cards such as the P100 (6.0)")
    p.add_argument("--sdpa", action="store_true",
                   help="use fused scaled_dot_product_attention instead of the "
                        "notebook's explicit attention. Mathematically identical; "
                        "only reaches the FlashAttention backend on sm_80+ hardware")
    p.add_argument("--fused-adam", action="store_true",
                   help="use the fused AdamW kernel. Small win here, since there is "
                        "one optimizer step per --grad-accum micro-batches")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loss and evaluation. Both are from the original notebook.
# ---------------------------------------------------------------------------

def calc_loss_batch(input_batch, target_batch, model, device):
    batch_size, seq_len = input_batch.shape
    tbatch_size, tseq_len = target_batch.shape
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    # Changed from the notebook: vocab_size is read off the model's own output
    # instead of a module-level config dict, so this works for any model passed in.
    vocab_size = logits.shape[-1]
    loss = torch.nn.functional.cross_entropy(logits.view(batch_size * seq_len, vocab_size), target_batch.view(tbatch_size * tseq_len)) #logits: shape [2, 3, 4]  logits.flatten(0, 1):shape[6, 4]
    return loss


def estimate_loss(model, path, eval_iter, device, batch_size, block_size, ctx):
    """Average loss over several random batches.

    One batch is far too noisy to compare checkpoints with, so this averages
    `eval_iter` of them. model.eval() disables dropout for the measurement and
    model.train() puts it back, because evaluating with dropout active reports a
    loss the model does not actually have.
    """
    losses = []
    model.eval()
    with torch.no_grad():
        for _ in range(eval_iter):
            X, y = get_batch(path, batch_size, block_size, device)
            with ctx:
                loss = calc_loss_batch(X, y, model, device)
            losses.append(loss.item())
    model.train()
    return np.mean(losses)


def generate_and_print_sample(model, tokenizer, device, start_context, context_size):
    """Sample from the model at the end of each epoch, as the notebook did."""
    model.eval()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    token_ids = generate(model=model, idx=encoded, max_new_tokens=50,
                         context_size=context_size, temperature=0.0)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    # An untrained model emits tokens from the whole GPT-2 vocabulary, including
    # non-Latin scripts. A default Windows console is cp1252 and raises
    # UnicodeEncodeError on them, which would kill the training run at the first
    # sample. Drop anything the console cannot render rather than crash.
    safe = decoded_text.replace("\n", " ")
    encoding = sys.stdout.encoding or "utf-8"
    safe = safe.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print("  sample:", safe)  # Compact print format
    model.train()


# ---------------------------------------------------------------------------
# Learning rate, throughput accounting, run logging. Added after the audit.
# ---------------------------------------------------------------------------

def get_lr(step, peak_lr, min_lr, warmup_steps, max_steps):
    """Linear warmup, then cosine decay to the floor.

    Warmup: parameters start random, so the first gradients are large and badly
    directed. Ramping up from near zero avoids wrecking the weights before the
    model has learned anything.

    Cosine decay: big steps early to cover ground, smaller steps later to settle
    into a minimum rather than bounce around it.

    Returns a float. The caller writes it into the live optimizer, which is what
    keeps schedule and optimizer from ever falling out of sync.
    """
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(progress, 1.0)  # clamp, so the LR never rises again past the end
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


# Peak throughput per GPU, used only to turn tokens/sec into an MFU percentage.
# fp16 figures are tensor-core peaks where the card has tensor cores; the P100
# has none, so its fp16 number is the 2x-rate fp16 FMA peak.
GPU_PEAK_TFLOPS = {
    "P100": {"float16": 18.7, "bfloat16": None, "float32": 9.3},
    "T4":   {"float16": 65.0, "bfloat16": None, "float32": 8.1},
    "V100": {"float16": 125.0, "bfloat16": None, "float32": 15.7},
    "A100": {"float16": 312.0, "bfloat16": 312.0, "float32": 19.5},
    "L4":   {"float16": 121.0, "bfloat16": 121.0, "float32": 30.3},
}


def flops_per_token(n_params, n_layers, context, emb_dim, vocab_size=None):
    """Forward + backward FLOPs for one token.

    6N covers the dense parameters: roughly 2N multiply-accumulates forward and
    4N backward. The second term is the attention score and value matmuls, which
    are not parameter-bound and grow with context length.

    **The token embedding must be excluded from N.** Looking up a row of
    `tok_emb` is a gather, not a matrix multiply, so it costs no FLOPs.
    nanoGPT includes its embedding table in N only because its output head is
    *tied* to it, so the one parameter block is counted once and does the work
    once. This model's head is untied: `out_head` is a genuine 768 x 50257
    matmul and belongs in N, while `tok_emb` is a separate table of the same
    size that does not. Counting both inflates MFU by about 40% relative, which
    an earlier version of this file did.

    Pass `vocab_size` to have the embedding subtracted. Positional embeddings are
    a gather too, but at 256 x 768 they are negligible either way.
    """
    if vocab_size is not None:
        n_params = n_params - vocab_size * emb_dim - context * emb_dim
    return 6 * n_params + 12 * n_layers * context * emb_dim


def compute_mfu(tokens_per_sec, n_params, n_layers, context, emb_dim, device_name,
                dtype, vocab_size=None):
    """Model FLOPs Utilization: achieved FLOPs as a fraction of the GPU's peak.

    This is what makes a throughput figure comparable across hardware. Returns
    (achieved_tflops, mfu_percent_or_None); the percentage is None for a GPU not
    in the table above, in which case the achieved TFLOP/s still stands alone.
    """
    achieved = flops_per_token(n_params, n_layers, context, emb_dim,
                               vocab_size) * tokens_per_sec
    achieved_tflops = achieved / 1e12

    peak = None
    for key, table in GPU_PEAK_TFLOPS.items():
        if key in device_name:
            peak = table.get(dtype)
            break
    if not peak:
        return round(achieved_tflops, 2), None
    return round(achieved_tflops, 2), round(achieved_tflops / peak * 100, 1)


def git_commit():
    """Record which version of the code produced a run, or None outside git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


class RunLogger:
    """Writes everything needed to reconstruct and trust a run.

    config.json    every hyperparameter, the git commit, the seed, library
                   versions, the GPU name, and the exact command line
    metrics.jsonl  one line per evaluation
    summary.json   final losses, tokens, throughput, MFU, peak memory
    loss_curve.png train and validation loss against tokens seen
    """

    def __init__(self, out_dir, run_name, config):
        self.dir = os.path.join(out_dir, run_name)
        os.makedirs(self.dir, exist_ok=True)
        self.t0 = time.time()
        self.history = []

        config = dict(config)
        config["git_commit"] = git_commit()
        config["command"] = " ".join(sys.argv)
        config["python_version"] = sys.version.split()[0]
        config["torch_version"] = torch.__version__
        config["device_name"] = (torch.cuda.get_device_name(0)
                                 if torch.cuda.is_available() else "cpu")
        self.config = config
        self._write("config.json", config)
        print(f"[logger] writing to {self.dir}")

    def _write(self, name, obj):
        with open(os.path.join(self.dir, name), "w") as f:
            json.dump(obj, f, indent=2)

    def log(self, **metrics):
        metrics["wall_time_s"] = round(time.time() - self.t0, 1)
        self.history.append(metrics)
        with open(os.path.join(self.dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def finish(self, **summary):
        summary["total_wall_time_s"] = round(time.time() - self.t0, 1)
        self._write("summary.json", summary)
        self._plot()
        print(f"[logger] run complete: {self.dir}")

    def _plot(self):
        if not self.history:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")  # no display on a headless machine
            import matplotlib.pyplot as plt
        except ImportError:
            return
        tokens = [m["tokens_seen"] for m in self.history]
        plt.figure(figsize=(7, 4))
        plt.plot(tokens, [m["train_loss"] for m in self.history], label="train loss")
        plt.plot(tokens, [m["val_loss"] for m in self.history], label="val loss")
        plt.xlabel("tokens seen")
        plt.ylabel("cross-entropy loss")
        plt.title(os.path.basename(self.dir))
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir, "loss_curve.png"), dpi=120)
        plt.close()


# ---------------------------------------------------------------------------
# The training loop, in the shape of the notebook's train_model.
# ---------------------------------------------------------------------------

def train_model(
    model,
    optimizer,
    lr_schedule,          # was `scheduler`: now a function, see bug 2 above
    device,
    num_epochs,
    num_batches_per_epoch,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    train_bin,            # added: the sampler takes its data and shape as
    val_bin,              # arguments now, rather than reading globals (bug 1)
    batch_size,
    block_size,
    grad_clip,
    logger,
    n_params,
    gradient_accumulation_steps=1,
    precision_dtype=torch.float16,
):
    # Track metrics
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Mixed precision on GPU only, so the same code still runs on a CPU machine
    use_amp = device.type == "cuda" and precision_dtype != torch.float32
    ctx = autocast(device_type="cuda", dtype=precision_dtype) if use_amp else nullcontext()
    # Loss scaling is only needed for float16; bfloat16 has fp32's exponent range
    scaler = GradScaler(enabled=use_amp and precision_dtype == torch.float16)

    max_steps = max(1, (num_epochs * num_batches_per_epoch) // gradient_accumulation_steps)
    best_val = float("inf")
    window_tokens, window_t0, train_start = 0, time.time(), time.time()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        for batch_idx in range(num_batches_per_epoch):
            # Get batch and move to GPU
            input_batch, target_batch = get_batch(train_bin, batch_size, block_size, device)

            # Forward pass in mixed precision
            with ctx:
                loss = calc_loss_batch(input_batch, target_batch, model, device)

            # Normalize loss for accumulation
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

            # Count tokens from the tensor that actually went through the model,
            # never from the config. This is what the original run got wrong.
            tokens_seen += input_batch.numel()
            window_tokens += input_batch.numel()

            # Gradient accumulation step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # The schedule reaches the model through these two lines. There is
                # no scheduler object to hold a reference to a dead optimizer.
                lr = lr_schedule(global_step + 1, max_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                scaler.unscale_(optimizer)  # clip real gradients, not scaled ones
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)      # unscale grads back to float32 from float16
                scaler.update()             # learns whether to raise or lower the scale
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                # Evaluation checkpoint
                if global_step % eval_freq == 0:
                    train_loss = estimate_loss(model, train_bin, eval_iter, device,
                                               batch_size, block_size, ctx)
                    val_loss = estimate_loss(model, val_bin, eval_iter, device,
                                             batch_size, block_size, ctx)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)

                    tok_per_s = window_tokens / max(time.time() - window_t0, 1e-9)
                    peak_mem = (torch.cuda.max_memory_allocated() / 1e9
                                if device.type == "cuda" else 0.0)
                    tflops, mfu = compute_mfu(
                        tok_per_s, n_params, len(model.trf_blocks), block_size,
                        model.tok_emb.embedding_dim, logger.config["device_name"],
                        str(precision_dtype).replace("torch.", ""),
                        vocab_size=model.tok_emb.num_embeddings)

                    logger.log(step=global_step, tokens_seen=tokens_seen, lr=lr,
                               train_loss=round(float(train_loss), 4),
                               val_loss=round(float(val_loss), 4),
                               val_perplexity=round(math.exp(val_loss), 2),
                               tokens_per_sec=round(tok_per_s),
                               achieved_tflops=tflops, mfu_percent=mfu,
                               peak_gpu_mem_gb=round(peak_mem, 2))

                    mfu_str = f" | MFU {mfu:.1f}%" if mfu is not None else ""
                    print(
                        f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, "
                        f"Perplexity {math.exp(val_loss):.3f}, LR {lr:.6f}, "
                        f"{tok_per_s/1e3:.1f}K tok/s, {tflops:.1f} TFLOP/s{mfu_str}, "
                        f"{peak_mem:.1f} GB"
                    )

                    if val_loss < best_val:
                        best_val = val_loss
                    window_tokens, window_t0 = 0, time.time()

        # Sample generation after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context, block_size)

    total_time = time.time() - train_start
    return train_losses, val_losses, track_tokens_seen, {
        "tokens_seen": tokens_seen,
        "optimizer_steps": global_step + 1,
        "best_val_loss": round(float(best_val), 4),
        "total_train_time_s": round(total_time, 1),
    }


def main():
    args = parse_args()

    # Seed everything, so a rerun with the same flags gives the same run. Without
    # this two "identical" runs differ by more than whatever is being compared.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                       "float32": torch.float32}[args.dtype]

    model_cfg = {
        "vocab_size": GPT_CONFIG_134M["vocab_size"],
        "context_length": args.context,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.dropout,
        "qkv_bias": GPT_CONFIG_134M["qkv_bias"],
    }
    if args.min_lr is None:
        args.min_lr = args.lr / 10  # derived, so it can never exceed the peak

    # Turn the token budget into the notebook's loop counts
    tokens_per_micro_batch = args.batch_size * args.context
    total_micro_batches = max(args.grad_accum,
                              int(args.train_tokens) // tokens_per_micro_batch)
    num_batches_per_epoch = max(args.grad_accum, total_micro_batches // args.num_epochs)
    tokens_per_step = tokens_per_micro_batch * args.grad_accum
    max_steps = max(1, (args.num_epochs * num_batches_per_epoch) // args.grad_accum)

    logger = RunLogger(args.out_dir, args.run_name, {
        "args": vars(args),
        "model_config": model_cfg,
        "num_batches_per_epoch": num_batches_per_epoch,
        "tokens_per_optimizer_step": tokens_per_step,
        "max_optimizer_steps": max_steps,
    })

    model = GPTModel(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params / 1e6:.2f}M parameters on {device}")
    print(f"{tokens_per_step:,} tokens/step x {max_steps:,} steps "
          f"= {tokens_per_step * max_steps / 1e6:.1f}M tokens")

    # betas=(0.9, 0.95) rather than the default 0.999: the shorter second-moment
    # window adapts faster, which is standard for language model pretraining.
    adam_kwargs = dict(lr=args.lr, betas=(0.9, 0.95),
                       weight_decay=args.weight_decay, eps=1e-9)
    if args.fused_adam and device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(model.parameters(), fused=True, **adam_kwargs)
            print("optimizer: fused AdamW")
        except (RuntimeError, ValueError) as e:
            optimizer = torch.optim.AdamW(model.parameters(), **adam_kwargs)
            print(f"optimizer: AdamW (fused unavailable: {e})")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **adam_kwargs)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        print(f"resumed from {args.resume}")

    # Keep a handle on the uncompiled module. torch.compile returns a wrapper
    # whose state_dict prefixes every key with "_orig_mod.", so a checkpoint
    # saved from it cannot be loaded by GPTModel().load_state_dict() without
    # stripping that prefix. Saving from raw_model avoids the whole problem.
    raw_model = model

    if args.compile:
        # torch.compile's Triton backend requires a recent NVIDIA architecture
        # (Triton currently states compute capability 8.0+, i.e. Ampere). Rather
        # than hard-code a threshold that moves between releases, try it and fall
        # back cleanly. On a P100 (6.0) or a T4 (7.5) this prints why it skipped.
        if device.type != "cuda":
            print("--compile ignored: no CUDA device")
        else:
            major, minor = torch.cuda.get_device_capability(0)
            try:
                model = torch.compile(model)
                print(f"torch.compile enabled on sm_{major}{minor} "
                      f"(the first step will be slow)")
            except Exception as e:
                print(f"--compile ignored: {torch.cuda.get_device_name(0)} is "
                      f"sm_{major}{minor}, and torch.compile failed to initialise "
                      f"({type(e).__name__}: {e}). Training continues uncompiled.")

    if args.sdpa:
        from fast_attention import enable_sdpa
        backend = enable_sdpa(model, device)
        print(f"fused attention enabled, backend: {backend}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    tokenizer = tiktoken.get_encoding("gpt2")

    def lr_schedule(step, total_steps):
        return get_lr(step, args.lr, args.min_lr, args.warmup_steps, total_steps)

    train_losses, val_losses, track_tokens_seen, stats = train_model(
        model=model,
        optimizer=optimizer,
        lr_schedule=lr_schedule,
        device=device,
        num_epochs=args.num_epochs,
        num_batches_per_epoch=num_batches_per_epoch,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        start_context=args.start_context,
        tokenizer=tokenizer,
        train_bin=args.train_bin,
        val_bin=args.val_bin,
        batch_size=args.batch_size,
        block_size=args.context,
        grad_clip=args.grad_clip,
        logger=logger,
        n_params=n_params,
        gradient_accumulation_steps=args.grad_accum,
        precision_dtype=precision_dtype,
    )

    torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                "model_config": model_cfg, "args": vars(args),
                "tokens_seen": stats["tokens_seen"]},
               os.path.join(logger.dir, "ckpt_last.pt"))

    # Steady-state throughput is the median of the per-eval windows, which
    # excludes the first window's startup cost. avg_tokens_per_sec below is the
    # end-to-end figure and runs lower because it includes evaluation passes.
    windows = sorted(m["tokens_per_sec"] for m in logger.history)
    steady = windows[len(windows) // 2] if windows else 0
    steady_tflops, steady_mfu = compute_mfu(
        steady, n_params, args.n_layers, args.context, args.emb_dim,
        logger.config["device_name"], args.dtype,
        vocab_size=GPT_CONFIG_134M["vocab_size"])

    final_val = val_losses[-1] if val_losses else float("nan")
    logger.finish(
        final_val_loss=round(float(final_val), 4),
        final_val_perplexity=round(math.exp(final_val), 2),
        best_val_loss=stats["best_val_loss"],
        tokens_seen=stats["tokens_seen"],
        optimizer_steps=stats["optimizer_steps"],
        steady_tokens_per_sec=steady,
        steady_achieved_tflops=steady_tflops,
        steady_mfu_percent=steady_mfu,
        avg_tokens_per_sec=round(stats["tokens_seen"] / max(stats["total_train_time_s"], 1e-9)),
        peak_gpu_mem_gb=(round(torch.cuda.max_memory_allocated() / 1e9, 2)
                         if device.type == "cuda" else 0.0),
        n_params=n_params,
    )


if __name__ == "__main__":
    main()
