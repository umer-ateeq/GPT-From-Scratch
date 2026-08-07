"""Audit a checkpoint: recover its true architecture and training context from
the weights alone, without trusting any config file, notebook, or filename.

Why this exists: the released checkpoint `weights8b_300epoch.pth` is a bare
state_dict with no metadata. The training notebook *configured* batch 64 x
context 256, but a later cell reassigned the globals that the batch sampler
read, so the run actually used batch 32 x context 128. Nothing in the file says
so. The positional embedding table does.

The argument: `pos_emb.weight` has one row per position. A row only receives
gradient when a training batch is at least that long. AdamW weight decay pulls
every parameter it touches toward zero on every step, so rows that were never
in a batch decay toward zero while trained rows keep a healthy norm. Reading
the per-row norms therefore recovers the true training context length, and the
token budget follows from it.

Usage:
    python audit_checkpoint.py --ckpt weights8b_300epoch.pth
    python audit_checkpoint.py --ckpt weights8b_300epoch.pth --cycles 300 --batches-per-cycle 1000 --batch-size 32
"""
import argparse
import math

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="weights8b_300epoch.pth")
    p.add_argument("--dead-ratio", type=float, default=0.1,
                   help="a position counts as untrained when its norm is below "
                        "this fraction of the largest row norm. Referenced to the "
                        "max, not the median: when half the table is dead the "
                        "median itself falls inside the dead cluster.")
    # only used to turn the recovered context into a token budget
    p.add_argument("--cycles", type=int, default=300, help="outer training cycles")
    p.add_argument("--batches-per-cycle", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32,
                   help="micro-batch rows actually used by the run")
    return p.parse_args()


def load_state_dict(path):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in obj:
                return obj[key], key
    return obj, None


def recover_architecture(state):
    """Infer the config from tensor shapes only."""
    vocab_size, emb_dim = state["tok_emb.weight"].shape
    context_length = state["pos_emb.weight"].shape[0]
    n_layers = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("trf_blocks."))
    ffn_hidden = state["trf_blocks.0.ff.layers.0.weight"].shape[0]
    # head count is not recoverable from shapes (all projections are d_model x
    # d_model); it is a run-time choice. 768/64 = 12 is what the run used.
    return {
        "vocab_size": vocab_size,
        "emb_dim": emb_dim,
        "context_length": context_length,
        "n_layers": n_layers,
        "ffn_expansion": ffn_hidden // emb_dim,
        "qkv_bias": "trf_blocks.0.att.W_query.bias" in state,
        "out_head_tied": "out_head.weight" not in state,
        "norm_type": "custom LayerNorm (scale/shift)"
        if "final_norm.scale" in state else "nn.LayerNorm",
    }


def positional_embedding_audit(state, dead_ratio):
    pos = state["pos_emb.weight"].float()
    norms = pos.norm(dim=1)
    n = norms.numel()

    threshold = dead_ratio * norms.max().item()
    alive = (norms >= threshold)

    # the trained prefix is the longest run of alive rows from position 0
    trained_prefix = 0
    for i in range(n):
        if not alive[i]:
            break
        trained_prefix = i + 1

    return {
        "n_positions": n,
        "norms": norms,
        "trained_prefix": trained_prefix,
        "threshold": threshold,
        "n_dead": int((~alive).sum().item()),
    }


def main():
    args = parse_args()
    state, wrapper = load_state_dict(args.ckpt)

    print(f"=== checkpoint: {args.ckpt} ===")
    print(f"container      : {'wrapped under ' + repr(wrapper) if wrapper else 'bare state_dict (no metadata)'}")
    print(f"tensors        : {len(state)}")

    trainable = sum(v.numel() for k, v in state.items() if not k.endswith(".mask"))
    buffers = sum(v.numel() for k, v in state.items() if k.endswith(".mask"))
    print(f"parameters     : {trainable:,} trainable "
          f"({trainable / 1e6:.2f}M) + {buffers:,} mask buffers "
          f"= {trainable + buffers:,} total")

    print("\n=== architecture recovered from tensor shapes ===")
    for key, value in recover_architecture(state).items():
        print(f"  {key:18s} {value}")

    print("\n=== positional embedding audit ===")
    audit = positional_embedding_audit(state, args.dead_ratio)
    norms, n = audit["norms"], audit["n_positions"]
    prefix = audit["trained_prefix"]

    print(f"  pos_emb rows     : {n}")
    print(f"  dead threshold   : {audit['threshold']:.4f} "
          f"({args.dead_ratio:g} x max row norm {norms.max():.4f})")

    # print the norm profile in blocks so the cliff is visible
    block = max(1, n // 8)
    print(f"\n  {'positions':>16s}  {'mean norm':>10s}  {'min':>8s}  {'max':>8s}")
    for start in range(0, n, block):
        chunk = norms[start:start + block]
        print(f"  {f'{start}-{start + len(chunk) - 1}':>16s}  "
              f"{chunk.mean():10.4f}  {chunk.min():8.4f}  {chunk.max():8.4f}")

    if prefix == n:
        print(f"\n  VERDICT: all {n} positions were trained. "
              f"Effective context = configured context = {n}.")
        effective_context = n
    else:
        trained = norms[:prefix]
        dead = norms[prefix:]
        ratio = trained.mean().item() / max(dead.mean().item(), 1e-12)
        print(f"\n  VERDICT: positions 0-{prefix - 1} are trained "
              f"(mean norm {trained.mean():.4f}); positions {prefix}-{n - 1} "
              f"never received gradient (mean norm {dead.mean():.6f}).")
        print(f"  The trained rows carry {ratio:,.0f}x the norm of the dead rows.")
        print(f"  Effective training context = {prefix}, not the configured {n}.")
        print(f"  => perplexity must be measured at context {prefix}. Evaluating at "
              f"{n} scores the model on positions it has never seen.")
        effective_context = prefix

    print("\n=== token budget implied by the recovered context ===")
    tokens = args.cycles * args.batches_per_cycle * args.batch_size * effective_context
    print(f"  {args.cycles} cycles x {args.batches_per_cycle} batches "
          f"x {args.batch_size} rows x {effective_context} tokens")
    print(f"  = {tokens:,} tokens seen ({tokens / 1e9:.2f}B)")
    naive = args.cycles * args.batches_per_cycle * 64 * 256
    print(f"  (the configured 64 x 256 would have given {naive / 1e9:.2f}B, "
          f"a {naive / tokens:.1f}x overcount)")


if __name__ == "__main__":
    main()
