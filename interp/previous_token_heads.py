"""Find the previous-token heads: the other half of the induction circuit.

An induction head cannot work alone, and the reason is worth stating carefully
because it is the whole argument for why the circuit needs two layers.

When an induction head sits on the second occurrence of token B and wants to
attend to whatever followed B last time, it has to *find* that position. It does
that by matching: it looks for a position whose stored information says "the
token before me was B". But nothing puts that information there by default. Each
position's residual stream starts out holding its own token, not its
predecessor's.

So an earlier head has to do it first. A **previous-token head** attends from
position i to position i-1 and copies information about token i-1 into position
i's residual stream. Now position i carries "I follow B", and a later head can
match on it.

That is why induction heads appear in later layers: the composition needs an
earlier layer to have run. Olsson et al., "In-context Learning and Induction
Heads" (Anthropic, 2022), calls this a two-head circuit.

## What this script measures

For every head, the average attention weight from each position i to position
i-1, on ordinary token sequences. A head scoring near 1.0 does almost nothing
but look one token back. Scores are reported against the uniform baseline, which
is what an indifferent causal head would put on any single position.

Random tokens are used here too, so the result reflects a positional habit rather
than anything about specific words.

Usage:
    python previous_token_heads.py --ckpt weights8b_300epoch.pth
    python previous_token_heads.py --ckpt weights8b_300epoch.pth --plot images/prev_token_heads.png
"""
import argparse

import torch

import _bootstrap  # noqa: F401  (puts pretrain/ on sys.path)

from config import GPT_CONFIG_134M  # noqa: E402
from induction_heads import capture_attention  # noqa: E402
from model import load_checkpoint  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seq-len", type=int, default=96,
                   help="sequence length; must fit the trained context of 128")
    p.add_argument("--n-seqs", type=int, default=16)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--plot", default=None, help="save a layer x head heatmap here")
    return p.parse_args()


@torch.no_grad()
def previous_token_scores(model, seq_len, n_seqs, vocab_size, seed):
    """Mean attention from position i to position i-1, per (layer, head)."""
    blocks = list(model.trf_blocks)
    n_layers = len(blocks)
    n_heads = blocks[0].att.num_heads
    totals = torch.zeros(n_layers, n_heads)

    generator = torch.Generator().manual_seed(seed)

    # Skip position 0 (no predecessor) and position 1, where attending back is
    # trivially the only option other than itself.
    queries = torch.arange(2, seq_len)
    targets = queries - 1

    for _ in range(n_seqs):
        tokens = torch.randint(0, vocab_size, (1, seq_len), generator=generator)
        for layer, attn in enumerate(capture_attention(model, tokens)):
            totals[layer] += attn[:, queries, targets].mean(dim=1)

    return totals / n_seqs


def uniform_baseline(seq_len):
    """What an indifferent causal head would put on any single position."""
    queries = torch.arange(2, seq_len)
    return (1.0 / (queries + 1).float()).mean().item()


def main():
    args = parse_args()
    if args.seq_len > GPT_CONFIG_134M["context_length"]:
        raise SystemExit(f"seq_len must be <= {GPT_CONFIG_134M['context_length']}")

    model = load_checkpoint(args.ckpt, GPT_CONFIG_134M, device="cpu")
    model.eval()

    print(f"checkpoint : {args.ckpt}")
    print(f"probe      : {args.n_seqs} random sequences of {args.seq_len} tokens")

    scores = previous_token_scores(model, args.seq_len, args.n_seqs,
                                   GPT_CONFIG_134M["vocab_size"], args.seed)
    baseline = uniform_baseline(args.seq_len)
    n_layers, n_heads = scores.shape

    print(f"heads      : {n_layers} layers x {n_heads} heads = {n_layers * n_heads}")
    print(f"baseline   : {baseline:.4f}\n")

    flat = scores.flatten()
    order = torch.argsort(flat, descending=True)

    print(f"top {args.top} heads by attention to position i-1:\n")
    print(f"  {'layer.head':>12}  {'score':>8}  {'x uniform':>10}")
    for rank in range(min(args.top, flat.numel())):
        idx = order[rank].item()
        layer, head = divmod(idx, n_heads)
        score = flat[idx].item()
        print(f"  {f'L{layer}.H{head}':>12}  {score:8.4f}  {score / baseline:10.1f}x")

    # The circuit argument: previous-token heads must precede the induction heads
    # at L6.H9 and L7.H8 for composition to be possible.
    best_idx = order[0].item()
    best_layer = best_idx // n_heads
    print(f"\nstrongest previous-token head is in layer {best_layer}.")
    if best_layer < 6:
        print("It sits BEFORE the induction heads in layers 6 and 7, so the two-layer")
        print("composition the circuit requires is available. This is the expected")
        print("ordering and it is what makes induction possible at all.")
    else:
        print("It sits at or after the induction heads, which does not fit the")
        print("standard two-layer circuit. Worth investigating.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.2))
        im = ax.imshow(scores.numpy() / baseline, cmap="magma", aspect="auto")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(n_layers))
        ax.set_title("Previous-token score by head (multiples of uniform attention)")
        fig.colorbar(im, ax=ax, label="x uniform")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        plt.close(fig)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
