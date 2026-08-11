"""Does this model have induction heads?

An induction head is an attention head that implements the rule:

    "I have seen this token before. What came NEXT last time? Attend to that."

Given a sequence that repeats, like  A B C D  A B C D, a head sitting on the
second B should attend back to the first C, because C is what followed B last
time. Doing that lets the model copy repeated structure, and it is believed to
be the main mechanism behind in-context learning in transformers. Olsson et al.,
"In-context Learning and Induction Heads" (Anthropic, 2022), is the reference.

## How this script measures it

1. Build a random token sequence of length L and concatenate it with itself, so
   the model sees [seq, seq]. Random tokens matter: the model cannot rely on
   memorized English, so any copying behaviour has to come from the pattern in
   the context rather than from what it learned during training.

2. Run the model and capture every head's attention pattern.

3. For each head, look only at query positions in the SECOND copy. A query at
   position i (where i >= L) is looking at a token it already saw at position
   i - L. The "induction target" is position i - L + 1, the token that came
   immediately after that earlier occurrence.

4. The head's induction score is its average attention weight on that target,
   over all query positions in the second half. A score near 1.0 means the head
   does almost nothing else. A score near 1/i means it is spreading attention
   roughly uniformly and shows no induction behaviour.

5. Repeat over several random sequences and average, so the result is a property
   of the model rather than of one lucky sequence.

## Reading the output

Scores are compared against a uniform-attention baseline, which is what a head
with no preference would score. The ratio of the two is the interpretable
number: 10x uniform is a strong, clear induction head; 1x is nothing.

Usage:
    python induction_heads.py --ckpt weights8b_300epoch.pth
    python induction_heads.py --ckpt weights8b_300epoch.pth --plot images/induction_heads.png
"""
import argparse

import torch

import _bootstrap  # noqa: F401  (puts pretrain/ on sys.path)

from config import GPT_CONFIG_134M  # noqa: E402
from model import load_checkpoint  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seq-len", type=int, default=48,
                   help="length of the repeated block. 2x this must fit the "
                        "trained context of 128")
    p.add_argument("--n-seqs", type=int, default=16,
                   help="how many random sequences to average over")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--top", type=int, default=10, help="how many heads to list")
    p.add_argument("--plot", default=None, help="save a layer x head heatmap here")
    return p.parse_args()


def build_repeated_sequence(seq_len, vocab_size, generator):
    """Return a (1, 2*seq_len) tensor whose second half repeats the first."""
    half = torch.randint(0, vocab_size, (1, seq_len), generator=generator)
    return torch.cat([half, half], dim=1)


@torch.no_grad()
def capture_attention(model, tokens):
    """Get every layer's attention pattern, without modifying the model.

    model.py deliberately does not return attention weights: it is the code the
    checkpoint was trained with and is kept unchanged. So instead of editing it,
    this hooks each attention module to grab the tensor going *into* it, then
    recomputes the same softmax the module computes internally, using that
    module's own W_query and W_key.

    The arithmetic below is copied line for line from MultiHeadAttention.forward.
    Dropout is skipped because the model is in eval mode, where it is the
    identity anyway.

    Returns a list of (n_heads, n_tokens, n_tokens) tensors, one per layer.
    """
    blocks = list(model.trf_blocks)
    captured = {}
    handles = []

    for i, block in enumerate(blocks):
        def grab(module, args, idx=i):
            captured[idx] = args[0].detach()
        handles.append(block.att.register_forward_pre_hook(grab))

    model(tokens)

    for handle in handles:
        handle.remove()

    patterns = []
    for i, block in enumerate(blocks):
        att = block.att
        x = captured[i]
        b, num_tokens, _ = x.shape

        queries = att.W_query(x).view(b, num_tokens, att.num_heads, att.head_dim).transpose(1, 2)
        keys = att.W_key(x).view(b, num_tokens, att.num_heads, att.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = att.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        patterns.append(attn_weights[0])  # drop the batch dimension

    return patterns


@torch.no_grad()
def induction_scores(model, seq_len, n_seqs, vocab_size, seed, offset=1):
    """Mean attention paid to the induction target, per (layer, head).

    `offset` selects what "the target" means, which is how this doubles as its
    own control:
        offset=1  position i-L+1, the token that FOLLOWED the earlier
                  occurrence. This is induction.
        offset=0  position i-L, the earlier occurrence of the SAME token. A head
                  scoring here is a duplicate-token head, which notices repetition
                  without predicting anything.
        offset=2  one further along, a check that a head is not just attending
                  vaguely near the right region.

    Returns (mean, std), each (n_layers, n_heads). The std is across sequences,
    so it says whether a score is a property of the model or of one lucky draw.
    """
    blocks = list(model.trf_blocks)
    n_layers = len(blocks)
    n_heads = blocks[0].att.num_heads
    per_sequence = []

    generator = torch.Generator().manual_seed(seed)

    # Query positions in the second copy. Start at seq_len + 1 so the target
    # index is always a real earlier position, for every offset used here.
    queries = torch.arange(seq_len + 1, 2 * seq_len)
    targets = queries - seq_len + offset

    for _ in range(n_seqs):
        tokens = build_repeated_sequence(seq_len, vocab_size, generator)
        patterns = capture_attention(model, tokens)

        this_seq = torch.zeros(n_layers, n_heads)
        for layer, attn in enumerate(patterns):   # attn: (heads, tokens, tokens)
            picked = attn[:, queries, targets]    # (heads, n_queries)
            this_seq[layer] = picked.mean(dim=1)
        per_sequence.append(this_seq)

    stacked = torch.stack(per_sequence)           # (n_seqs, layers, heads)
    return stacked.mean(dim=0), stacked.std(dim=0)


def uniform_baseline(seq_len):
    """What a head with no preference at all would score.

    A causal head at query position i spreads its attention over i + 1 positions,
    so it puts 1 / (i + 1) on any single one of them. Averaged over the same
    query positions this script measures.
    """
    queries = torch.arange(seq_len + 1, 2 * seq_len)
    return (1.0 / (queries + 1).float()).mean().item()


def main():
    args = parse_args()
    if 2 * args.seq_len > GPT_CONFIG_134M["context_length"]:
        raise SystemExit(f"2 x seq_len must be <= {GPT_CONFIG_134M['context_length']}")

    model = load_checkpoint(args.ckpt, GPT_CONFIG_134M, device="cpu")
    model.eval()

    print(f"checkpoint : {args.ckpt}")
    print(f"probe      : {args.n_seqs} random sequences of {args.seq_len} tokens, "
          f"repeated twice ({2 * args.seq_len} total)")

    scores, stds = induction_scores(model, args.seq_len, args.n_seqs,
                                    GPT_CONFIG_134M["vocab_size"], args.seed)
    baseline = uniform_baseline(args.seq_len)
    n_layers, n_heads = scores.shape

    print(f"heads      : {n_layers} layers x {n_heads} heads = {n_layers * n_heads}")
    print(f"baseline   : {baseline:.4f} attention on any single position "
          f"if a head had no preference\n")

    flat = scores.flatten()
    flat_std = stds.flatten()
    order = torch.argsort(flat, descending=True)

    print(f"top {args.top} heads by induction score "
          f"(std across {args.n_seqs} sequences):\n")
    print(f"  {'layer.head':>12}  {'score':>8}  {'std':>8}  {'x uniform':>10}")
    for rank in range(min(args.top, flat.numel())):
        idx = order[rank].item()
        layer, head = divmod(idx, n_heads)
        score, sd = flat[idx].item(), flat_std[idx].item()
        print(f"  {f'L{layer}.H{head}':>12}  {score:8.4f}  {sd:8.4f}  "
              f"{score / baseline:10.1f}x")

    # Control: the same measurement aimed at the SAME token rather than the next
    # one. A duplicate-token head scores here; an induction head does not.
    dup, _ = induction_scores(model, args.seq_len, args.n_seqs,
                              GPT_CONFIG_134M["vocab_size"], args.seed, offset=0)
    best = order[0].item()
    bl, bh = divmod(best, n_heads)
    print(f"\ncontrol, is L{bl}.H{bh} an induction head or a duplicate-token head?")
    print(f"  attention on i-L+1 (the NEXT token) : {flat[best]:.4f}")
    print(f"  attention on i-L   (the SAME token) : {dup[bl, bh]:.4f}")
    ratio = flat[best].item() / max(dup[bl, bh].item(), 1e-9)
    print(f"  ratio {ratio:.0f}x -> "
          f"{'induction, not duplicate detection' if ratio > 5 else 'AMBIGUOUS, it may be a duplicate-token head'}")

    best = flat.max().item()
    ratio = best / baseline
    print(f"\nstrongest head: {ratio:.1f}x uniform attention on the induction target")
    if ratio >= 10:
        verdict = ("Clear induction behaviour. At least one head reliably attends to "
                   "the token that followed the previous occurrence.")
    elif ratio >= 3:
        verdict = ("Partial induction behaviour. Some heads prefer the induction "
                   "target, but none is a clean, dedicated induction head.")
    else:
        verdict = ("No meaningful induction behaviour. Attention on the induction "
                   "target is close to what an indifferent head would give.")
    print(f"verdict: {verdict}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.2))
        im = ax.imshow(scores.numpy() / baseline, cmap="viridis", aspect="auto")  # noqa: F841
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(n_layers))
        ax.set_title("Induction score by head (multiples of uniform attention)")
        fig.colorbar(im, ax=ax, label="x uniform")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        plt.close(fig)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
