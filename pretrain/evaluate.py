"""Perplexity, measured two ways, with GPT-2-small available as a baseline.

Perplexity is exp(average cross-entropy per token). Read it as "on average the
model was as uncertain as if it were choosing uniformly among this many tokens".
Lower is better, and the floor is 1.

Two protocols, because a perplexity number without its protocol is not
comparable to anything:

  --mode bin        sequential non-overlapping windows over a tokenized .bin
                    corpus. Every token scored exactly once. Strict and fast.
  --mode wikitext   the strided sliding window from the GPT-2 paper, run over
                    the WikiText-2 test set. Each token gets up to
                    `max_length - 1` tokens of left context, the window advances
                    by `stride`, and overlap tokens that exist only to provide
                    context are masked out of the loss so nothing is counted
                    twice.

`--model gpt2` runs HuggingFace's GPT-2-small through the identical scoring
function, so a comparison isolates the model rather than the evaluation code.
That baseline is what makes any number reported here checkable: GPT-2-small's
published WikiText-2 perplexity is about 29.4 at context 1024, so if this
harness produces a sane figure for it at a shorter context, the harness is sound.

Context warning: the released checkpoint only trained positions 0-127. Scoring
it at 256 measures it on positional rows it has never seen, which roughly
doubles perplexity for reasons unrelated to model quality. This script detects
that from the weights and warns.

Usage:
    python evaluate.py --ckpt weights.pth --mode wikitext --max-length 128
    python evaluate.py --model gpt2 --mode wikitext --max-length 128
    python evaluate.py --ckpt weights.pth --mode bin --data-bin validation.bin --context 128
"""
import argparse
import math
import os
import time

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from config import GPT_CONFIG_134M
from model import GPTModel


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="scratch", choices=["scratch", "gpt2"],
                   help="'scratch' loads --ckpt; 'gpt2' loads GPT-2-small as a baseline")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--mode", default="wikitext", choices=["wikitext", "bin"])
    p.add_argument("--data-bin", default="validation.bin", help="--mode bin only")
    p.add_argument("--context", type=int, default=128, help="--mode bin window length")
    p.add_argument("--max-length", type=int, default=128, help="--mode wikitext window")
    p.add_argument("--stride", type=int, default=None,
                   help="window advance; defaults to max_length (non-overlapping)")
    p.add_argument("--batch-size", type=int, default=8, help="--mode bin only")
    p.add_argument("--max-tokens", type=int, default=0, help="0 = the whole set")
    return p.parse_args()


def trained_positions(state):
    """Recover how many positional rows actually received gradient.

    Untrained rows decay toward zero under AdamW weight decay while trained rows
    keep a healthy norm, so the boundary is visible as a cliff in the row norms.
    See audit_checkpoint.py for the full argument.
    """
    norms = state["pos_emb.weight"].float().norm(dim=1)
    threshold = 0.1 * norms.max().item()
    alive = norms >= threshold
    n = 0
    for i in range(norms.numel()):
        if not alive[i]:
            break
        n = i + 1
    return n


def load_scratch_model(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model = GPTModel(GPT_CONFIG_134M)
    model.load_state_dict(state)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    return (lambda x: model(x)), GPT_CONFIG_134M["context_length"], n_params, state


def load_gpt2_baseline(device):
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    return (lambda x: model(x).logits), model.config.n_positions, n_params, None


@torch.no_grad()
def strided_perplexity(forward, token_ids, max_length, stride, device):
    """Sliding-window perplexity with the overlap masked out of the loss."""
    n = token_ids.size(0)
    nll_sum, n_scored, prev_end = 0.0, 0, 0

    for begin in range(0, n, stride):
        end = min(begin + max_length, n)
        target_len = end - prev_end  # tokens in this window not yet scored
        if target_len <= 0:
            continue
        ids = token_ids[begin:end].unsqueeze(0).to(device)
        targets = ids.clone()
        targets[:, :-target_len] = -100  # -100 is ignored by cross_entropy

        logits = forward(ids)
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            targets[:, 1:].reshape(-1),
            ignore_index=-100, reduction="sum")

        nll_sum += loss.item()
        n_scored += int((targets[:, 1:] != -100).sum().item())
        prev_end = end
        if end == n:
            break

    avg_nll = nll_sum / n_scored
    return math.exp(avg_nll), avg_nll, n_scored


@torch.no_grad()
def bin_perplexity(forward, path, context, batch_size, device, max_batches=None):
    """Sequential non-overlapping windows over a uint16 token file."""
    data = np.fromfile(path, dtype=np.uint16)
    n_batches = ((len(data) - 1) // context) // batch_size
    if max_batches:
        n_batches = min(n_batches, max_batches)

    total_loss, total_tokens = 0.0, 0
    for b in range(n_batches):
        starts = [(b * batch_size + i) * context for i in range(batch_size)]
        x = torch.stack([torch.from_numpy(data[s:s + context].astype(np.int64))
                         for s in starts]).to(device)
        y = torch.stack([torch.from_numpy(data[s + 1:s + 1 + context].astype(np.int64))
                         for s in starts]).to(device)
        logits = forward(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg = total_loss / total_tokens
    return math.exp(avg), avg, total_tokens


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 1)

    if args.model == "gpt2":
        forward, model_ctx, n_params, state = load_gpt2_baseline(device)
        label = "GPT-2-small (baseline)"
    else:
        if not args.ckpt:
            raise SystemExit("--ckpt is required unless --model gpt2")
        forward, model_ctx, n_params, state = load_scratch_model(args.ckpt, device)
        label = f"from-scratch 134M ({os.path.basename(args.ckpt)})"

    window = args.max_length if args.mode == "wikitext" else args.context
    if window > model_ctx:
        raise SystemExit(f"window {window} exceeds the model context {model_ctx}")

    print(f"[model] {label} | {n_params / 1e6:.2f}M params | context {model_ctx} | {device}")

    # Warn before producing a number that would understate the model
    if state is not None:
        trained = trained_positions(state)
        if window > trained:
            print(f"[WARN]  only positions 0-{trained - 1} of this checkpoint were "
                  f"trained, but the window is {window}.")
            print(f"[WARN]  positions {trained}-{window - 1} never received gradient. "
                  f"Re-run with a window of {trained} for the honest number.")

    t0 = time.time()
    if args.mode == "wikitext":
        from datasets import load_dataset
        stride = args.stride or args.max_length
        enc = tiktoken.get_encoding("gpt2")
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        tokens = torch.tensor(enc.encode_ordinary("\n\n".join(ds["text"])),
                              dtype=torch.long)
        if args.max_tokens and args.max_tokens < tokens.size(0):
            tokens = tokens[:args.max_tokens]
        print(f"[data]  WikiText-2 raw test: {tokens.size(0):,} tokens")
        print(f"[eval]  max_length {args.max_length} | stride {stride}", flush=True)
        ppl, nll, scored = strided_perplexity(forward, tokens, args.max_length,
                                              stride, device)
    else:
        print(f"[data]  {args.data_bin}")
        print(f"[eval]  sequential non-overlapping windows | context {args.context}",
              flush=True)
        ppl, nll, scored = bin_perplexity(forward, args.data_bin, args.context,
                                          args.batch_size, device)

    print(f"[result] perplexity {ppl:.2f} | avg NLL {nll:.4f} | "
          f"{scored:,} tokens scored | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
