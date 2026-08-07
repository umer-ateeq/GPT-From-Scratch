"""Pretraining loop: mixed precision, gradient accumulation, warmup + cosine decay.

The techniques here exist because of one constraint: a single free-tier 16 GB
notebook GPU.

  Mixed precision (fp16)   halves activation memory and uses the GPU's tensor
                           cores. A GradScaler multiplies the loss before
                           backward so small gradients do not underflow to zero
                           in fp16, then unscales before the optimizer step.
  Gradient accumulation    32 micro-batches of 32 sequences are accumulated
                           before one optimizer step, giving the gradient
                           quality of a 1024-sequence batch at the memory cost
                           of 32. This is the main reason the run fits in 16 GB.
  Gradient clipping        caps the global gradient norm at 1.0, which stops a
                           single bad batch from destabilizing training.
  Pinned async transfers   see data.py: host-to-device copies overlap with compute.

Three fixes relative to the original notebook, all documented in docs/AUDIT.md
and each covered by a test:

  1. One optimizer. The notebook built a scheduler around one optimizer and then
     passed a different, freshly constructed one to the training loop, so every
     scheduler.step() updated an object that no longer touched the model. Here
     the learning rate is computed by a plain function and written into the live
     optimizer, so there is no second object to fall out of sync with.
  2. The cosine floor is derived as lr/10 rather than typed separately. The
     notebook set min_lr=5e-4 against a peak of 1e-4, so its "decay" would have
     been a climb.
  3. Batch shape is passed explicitly instead of read from globals.

Every run writes its full configuration, the realized token count, throughput
and loss curves to runs/<name>/, because the original run recorded none of that
and the bugs above went unnoticed for months as a direct result.

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
import torch
from torch.amp import GradScaler, autocast

from config import GPT_CONFIG_134M, TRAIN_CONFIG
from data import get_batch
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
    p.add_argument("--train-tokens", type=float, default=100e6,
                   help="stop after this many tokens; a fixed budget makes runs comparable")
    # evaluation and bookkeeping
    p.add_argument("--eval-every", type=int, default=50, help="in optimizer steps")
    p.add_argument("--eval-iters", type=int, default=20, help="batches per loss estimate")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"],
                   default="float16",
                   help="float16 on T4/V100/P100, bfloat16 on A100/H100, float32 on CPU")
    p.add_argument("--run-name", default="run")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--resume", default=None, help="path to a ckpt_last.pt to continue from")
    return p.parse_args()


def calc_loss_batch(input_batch, target_batch, model, vocab_size):
    """Cross-entropy between predicted logits and the next-token targets.

    Logits arrive as (batch, tokens, vocab) and targets as (batch, tokens).
    Both are flattened so every token position across the batch counts as one
    independent classification over the vocabulary.
    """
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size), target_batch.view(-1))


@torch.no_grad()
def estimate_loss(model, path, args, device, ctx):
    """Average loss over several random batches.

    A single batch is far too noisy to compare checkpoints against, so this
    averages `eval_iters` of them. Dropout is disabled via model.eval() and
    turned back on afterwards, since evaluating with dropout active would report
    a loss the model does not actually have.
    """
    model.eval()
    losses = []
    for _ in range(args.eval_iters):
        x, y = get_batch(path, args.batch_size, args.context, device)
        with ctx:
            losses.append(calc_loss_batch(x, y, model, GPT_CONFIG_134M["vocab_size"]).item())
    model.train()
    return float(np.mean(losses))


def get_lr(step, peak_lr, min_lr, warmup_steps, max_steps):
    """Linear warmup, then cosine decay to the floor.

    Warmup: the parameters start random, so the first gradients are large and
    poorly directed. Ramping the learning rate from near zero avoids blowing up
    the weights before the model has learned anything.

    Cosine decay: large steps early to cover ground, progressively smaller steps
    later to settle into a minimum instead of bouncing around it.

    Returns a float. The caller writes it into the optimizer, which is what
    keeps schedule and optimizer from desynchronizing.
    """
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(progress, 1.0)  # clamp so the LR never rises again past the end
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


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
    summary.json   final losses, total tokens, throughput, peak memory
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


def main():
    args = parse_args()

    # Seed everything so a rerun with the same flags gives the same run. Without
    # this, two "identical" ablation runs differ by more than the variable under test.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    use_amp = device.type == "cuda" and dtype != torch.float32
    ctx = autocast(device_type="cuda", dtype=dtype) if use_amp else nullcontext()
    # Loss scaling is only needed for fp16. bfloat16 has fp32's exponent range,
    # so gradients cannot underflow the same way.
    scaler = GradScaler(enabled=use_amp and dtype == torch.float16)

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

    tokens_per_step = args.batch_size * args.context * args.grad_accum
    max_steps = max(1, int(args.train_tokens) // tokens_per_step)

    logger = RunLogger(args.out_dir, args.run_name, {
        "args": vars(args),
        "model_config": model_cfg,
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay, eps=1e-9)

    step, tokens_seen = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step, tokens_seen = ckpt["step"], ckpt["tokens_seen"]
        print(f"resumed from {args.resume}: step {step}, {tokens_seen:,} tokens")

    def save(name):
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step, "tokens_seen": tokens_seen,
                    "model_config": model_cfg, "args": vars(args)},
                   os.path.join(logger.dir, name))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best_val = float("inf")
    window_tokens, window_t0, train_start = 0, time.time(), time.time()

    while step < max_steps:
        lr = get_lr(step, args.lr, args.min_lr, args.warmup_steps, max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr  # the schedule reaches the model through this line

        for _ in range(args.grad_accum):
            x, y = get_batch(args.train_bin, args.batch_size, args.context, device)
            with ctx:
                loss = calc_loss_batch(x, y, model, model_cfg["vocab_size"])
                # Divide by grad_accum so the accumulated gradient is the mean
                # over all micro-batches rather than their sum.
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            # Count tokens from the tensor that actually went through the model,
            # never from the config. This is what the original run got wrong.
            tokens_seen += x.numel()
            window_tokens += x.numel()

        scaler.unscale_(optimizer)  # clip real gradients, not scaled ones
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % args.eval_every == 0 or step == max_steps:
            train_loss = estimate_loss(model, args.train_bin, args, device, ctx)
            val_loss = estimate_loss(model, args.val_bin, args, device, ctx)
            tok_per_s = window_tokens / max(time.time() - window_t0, 1e-9)
            peak_mem = (torch.cuda.max_memory_allocated() / 1e9
                        if device.type == "cuda" else 0.0)

            logger.log(step=step, tokens_seen=tokens_seen, lr=lr,
                       train_loss=round(train_loss, 4), val_loss=round(val_loss, 4),
                       val_perplexity=round(math.exp(val_loss), 2),
                       tokens_per_sec=round(tok_per_s),
                       peak_gpu_mem_gb=round(peak_mem, 2))
            print(f"step {step:5d}/{max_steps} | {tokens_seen / 1e6:7.1f}M tok | "
                  f"train {train_loss:.3f} | val {val_loss:.3f} | "
                  f"ppl {math.exp(val_loss):7.2f} | lr {lr:.2e} | "
                  f"{tok_per_s / 1e3:.1f}K tok/s | {peak_mem:.1f} GB")

            save("ckpt_last.pt")
            if val_loss < best_val:
                best_val = val_loss
                save("ckpt_best.pt")
            window_tokens, window_t0 = 0, time.time()

    total_time = time.time() - train_start
    final_val = estimate_loss(model, args.val_bin, args, device, ctx)
    logger.finish(
        final_val_loss=round(final_val, 4),
        final_val_perplexity=round(math.exp(final_val), 2),
        best_val_loss=round(best_val, 4),
        tokens_seen=tokens_seen,
        optimizer_steps=step,
        avg_tokens_per_sec=round(tokens_seen / max(total_time, 1e-9)),
        peak_gpu_mem_gb=(round(torch.cuda.max_memory_allocated() / 1e9, 2)
                         if device.type == "cuda" else 0.0),
        n_params=n_params,
    )


if __name__ == "__main__":
    main()
