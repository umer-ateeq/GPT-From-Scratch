"""Is the induction head actually doing the work, or just watching?

induction_heads.py finds heads whose *attention* lands on the induction target.
That is correlational: a head can look at the right place and still contribute
nothing to the output. The only way to know is to break it and see if the model
gets worse.

## The experiment

Take a random token sequence and repeat it: [seq, seq]. On the FIRST copy the
model has never seen these tokens, so it can do no better than chance. On the
SECOND copy it can in principle copy from context, so loss should drop sharply.
That drop is in-context learning, measured directly.

Then ablate a head, by zeroing its slice of the attention output before the
output projection mixes the heads together, and repeat the measurement. If loss
on the second copy rises while loss on the first copy stays flat, that head was
causally responsible for the copying, not merely correlated with it.

Random tokens matter for the same reason as before: the model cannot fall back
on memorized English, so anything it does on the second copy has to come from
the context.

## Controls

Ablating any head perturbs the model a little, so the effect has to be compared
against something. The null distribution is **size-matched**: if the treatment
ablates two heads, the controls ablate random *pairs*, because removing two heads
disturbs the residual stream more than removing one and comparing across sizes
inflates the ratio. Heads with real induction scores are excluded from the pool
so a control can never accidentally be a treatment.

## Zero versus mean ablation

`--ablation zero` writes zeros into the head's slice. It is simple but it moves
the residual stream off-distribution by the head's *mean* output as well as its
input-dependent signal, so part of any damage is distribution shock rather than
lost computation. This is a known weakness of zero ablation (Zhang and Nanda,
2024).

`--ablation mean` replaces the head's output with its average over a sample of
inputs instead. The mean contribution stays, only the input-dependent part is
removed, so the model stays much closer to its normal operating regime. Where the
two disagree, the mean-ablation number is the defensible one.

## Uncertainty

Every reported delta comes with a 95% confidence interval from a paired bootstrap
over sequences. The design is paired by construction (the same seed drives the
intact run, the treatment, and every control), so resampling sequences is the
right resampling unit.

Usage:
    python ablate_heads.py --ckpt weights8b_300epoch.pth
    python ablate_heads.py --ckpt weights8b_300epoch.pth --ablation mean
    python ablate_heads.py --ckpt weights8b_300epoch.pth --heads 5.11
"""
import argparse

import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401  (puts pretrain/ on sys.path)

from config import GPT_CONFIG_134M  # noqa: E402
from model import load_checkpoint  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--heads", nargs="*", default=["6.9", "7.8"],
                   help="heads to ablate, as layer.head (default: the two found "
                        "by induction_heads.py)")
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--n-seqs", type=int, default=32)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-controls", type=int, default=12,
                   help="size-matched random head-set ablations used as the null")
    p.add_argument("--ablation", choices=["zero", "mean"], default="zero",
                   help="zero writes 0 into the head slice; mean replaces it with "
                        "the head's average output, which stays on-distribution")
    p.add_argument("--n-boot", type=int, default=2000,
                   help="paired bootstrap resamples for the confidence intervals")
    return p.parse_args()


def parse_head(spec):
    layer, head = spec.split(".")
    return int(layer), int(head)


def make_ablation_hook(head_idx, head_dim):
    """Zero one head's slice of the attention output.

    MultiHeadAttention concatenates the heads before out_proj, so head h occupies
    columns [h*head_dim, (h+1)*head_dim) of that concatenated tensor. Zeroing
    that slice removes the head's contribution while leaving everything else
    untouched. This is a forward *pre*-hook on out_proj, so it intercepts the
    concatenated tensor on its way in and the model itself is not modified.
    """
    lo, hi = head_idx * head_dim, (head_idx + 1) * head_dim

    def hook(module, args):
        x = args[0].clone()
        x[..., lo:hi] = 0.0
        return (x,)

    return hook


def make_mean_ablation_hook(head_idx, head_dim, mean_vec):
    """Replace one head's slice with its average output rather than zeros.

    Zeroing removes the head's mean contribution too, which pushes the residual
    stream somewhere the rest of the network never sees. Substituting the mean
    removes only the input-dependent part, which is the thing the head actually
    computes, and leaves the model on-distribution.
    """
    lo, hi = head_idx * head_dim, (head_idx + 1) * head_dim

    def hook(module, args):
        x = args[0].clone()
        x[..., lo:hi] = mean_vec
        return (x,)

    return hook


@torch.no_grad()
def head_output_means(model, seq_len, n_seqs, vocab_size, seed):
    """Average each head's pre-out_proj output over a sample of inputs."""
    sums, counts = {}, {}
    handles = []

    def recorder(layer):
        def hook(module, args):
            x = args[0].detach()
            flat = x.reshape(-1, x.shape[-1])
            sums[layer] = sums.get(layer, 0) + flat.sum(dim=0)
            counts[layer] = counts.get(layer, 0) + flat.shape[0]
        return hook

    for i, block in enumerate(model.trf_blocks):
        handles.append(block.att.out_proj.register_forward_pre_hook(recorder(i)))

    generator = torch.Generator().manual_seed(seed)
    for _ in range(n_seqs):
        half = torch.randint(0, vocab_size, (1, seq_len), generator=generator)
        model(torch.cat([half, half], dim=1))

    for h in handles:
        h.remove()
    return {i: sums[i] / counts[i] for i in sums}


def bootstrap_ci(deltas, n_boot=2000, seed=0):
    """95% CI for a mean, by resampling the per-sequence deltas."""
    t = torch.tensor(deltas, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    means = torch.stack([t[torch.randint(len(t), (len(t),), generator=g)].mean()
                         for _ in range(n_boot)])
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return t.mean().item(), lo.item(), hi.item()


@torch.no_grad()
def split_loss(model, tokens, seq_len):
    """Cross-entropy on the first copy and on the second copy, separately.

    The first copy is unpredictable by construction, so its loss is the model's
    floor. The second copy is where copying from context can help.
    """
    logits = model(tokens)
    targets = tokens[:, 1:]
    preds = logits[:, :-1, :]

    per_token = F.cross_entropy(
        preds.reshape(-1, preds.size(-1)), targets.reshape(-1), reduction="none")

    # target index i corresponds to token position i+1
    first = per_token[: seq_len - 1].mean().item()
    second = per_token[seq_len - 1:].mean().item()
    return first, second


@torch.no_grad()
def measure(model, seq_len, n_seqs, vocab_size, seed, hooks=()):
    """Average first-copy and second-copy loss, with optional ablation hooks."""
    handles = [module.register_forward_pre_hook(fn) for module, fn in hooks]
    generator = torch.Generator().manual_seed(seed)

    firsts, seconds = [], []
    for _ in range(n_seqs):
        half = torch.randint(0, vocab_size, (1, seq_len), generator=generator)
        tokens = torch.cat([half, half], dim=1)
        a, b = split_loss(model, tokens, seq_len)
        firsts.append(a)
        seconds.append(b)

    for handle in handles:
        handle.remove()

    return (sum(firsts) / len(firsts), sum(seconds) / len(seconds),
            firsts, seconds)


def hooks_for(model, heads, mode="zero", means=None):
    """Build (module, hook) pairs for a list of (layer, head) tuples."""
    out = []
    for layer, head in heads:
        att = model.trf_blocks[layer].att
        if mode == "mean":
            lo, hi = head * att.head_dim, (head + 1) * att.head_dim
            fn = make_mean_ablation_hook(head, att.head_dim, means[layer][lo:hi])
        else:
            fn = make_ablation_hook(head, att.head_dim)
        out.append((att.out_proj, fn))
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model = load_checkpoint(args.ckpt, GPT_CONFIG_134M, device="cpu")
    vocab = GPT_CONFIG_134M["vocab_size"]
    n_layers = len(model.trf_blocks)
    n_heads = model.trf_blocks[0].att.num_heads
    targets = [parse_head(h) for h in args.heads]

    k = len(targets)
    means = (head_output_means(model, args.seq_len, args.n_seqs, vocab, args.seed)
             if args.ablation == "mean" else None)

    print(f"checkpoint : {args.ckpt}")
    print(f"probe      : {args.n_seqs} random sequences of {args.seq_len}, repeated twice")
    print(f"ablating   : {', '.join(f'L{l}.H{h}' for l, h in targets)} "
          f"({args.ablation} ablation)")
    print(f"controls   : {args.n_controls} random sets of {k} head"
          f"{'s' if k > 1 else ''}, size-matched to the treatment\n")

    base_first, base_second, bf, bs = measure(model, args.seq_len, args.n_seqs,
                                              vocab, args.seed)
    raw_gap = base_first - base_second
    print("intact model")
    print(f"  loss on 1st copy (unpredictable) : {base_first:.4f}")
    print(f"  loss on 2nd copy (can copy)      : {base_second:.4f}")
    print(f"  raw first-to-second gap          : {raw_gap:.4f} nats")

    # The positional baseline. Later positions have more context whether or not
    # anything repeats, so part of the raw gap is not copying at all. Measuring it
    # on a NON-repeated sequence is what turns the raw gap into a copying benefit.
    gen = torch.Generator().manual_seed(args.seed)
    nf, ns = [], []
    for _ in range(args.n_seqs):
        toks = torch.randint(0, vocab, (1, 2 * args.seq_len), generator=gen)
        a, b = split_loss(model, toks, args.seq_len)
        nf.append(a)
        ns.append(b)
    pos_gap = sum(nf) / len(nf) - sum(ns) / len(ns)
    copy_benefit = raw_gap - pos_gap
    print(f"  minus positional baseline        : {pos_gap:.4f} nats "
          f"(measured on non-repeated sequences)")
    print(f"  TRUE copying benefit             : {copy_benefit:.4f} nats\n")

    if copy_benefit < 0.1:
        print("The intact model barely improves on the repeat, so there is no")
        print("copying behaviour to ablate. Stopping.")
        return

    ab_first, ab_second, af, asec = measure(
        model, args.seq_len, args.n_seqs, vocab, args.seed,
        hooks=hooks_for(model, targets, args.ablation, means))

    dmg, dlo, dhi = bootstrap_ci([x - y for x, y in zip(asec, bs)], args.n_boot)
    drf, rlo, rhi = bootstrap_ci([x - y for x, y in zip(af, bf)], args.n_boot)

    print(f"with {', '.join(f'L{l}.H{h}' for l, h in targets)} ablated")
    print(f"  1st-copy loss {ab_first:.4f}   change {drf:+.4f}  "
          f"[95% CI {rlo:+.4f}, {rhi:+.4f}]")
    print(f"  2nd-copy loss {ab_second:.4f}   change {dmg:+.4f}  "
          f"[95% CI {dlo:+.4f}, {dhi:+.4f}]")
    print(f"  share of TRUE copying destroyed : {dmg / copy_benefit * 100:.1f}%")
    print(f"  share of RAW gap destroyed      : {dmg / raw_gap * 100:.1f}%  "
          f"(uncorrected, do not report this one)\n")

    # Size-matched null: sets of k heads, drawn from heads with no induction role.
    generator = torch.Generator().manual_seed(args.seed + 1)
    excluded = set(targets) | {(6, 9), (7, 8), (6, 7), (7, 6), (5, 11)}
    controls, seen = [], set()
    while len(controls) < args.n_controls:
        pick = []
        while len(pick) < k:
            layer = int(torch.randint(0, n_layers, (1,), generator=generator))
            head = int(torch.randint(0, n_heads, (1,), generator=generator))
            if (layer, head) in excluded or (layer, head) in pick:
                continue
            pick.append((layer, head))
        key = tuple(sorted(pick))
        if key in seen:
            continue
        seen.add(key)
        controls.append(pick)

    print(f"control: {args.n_controls} random {k}-head sets, same ablation mode")
    control_damage = []
    for pick in controls:
        _, second, _, csec = measure(model, args.seq_len, args.n_seqs, vocab,
                                     args.seed,
                                     hooks=hooks_for(model, pick, args.ablation, means))
        d = second - base_second
        control_damage.append(d)
        label = "+".join(f"L{l}.H{h}" for l, h in pick)
        print(f"  {label:<22} {d:+.4f}")

    t = torch.tensor(control_damage, dtype=torch.float64)
    worst, mean_c, sd_c = t.max().item(), t.mean().item(), t.std().item()
    z = (dmg - mean_c) / sd_c if sd_c > 0 else float("inf")
    print(f"\n  null distribution: mean {mean_c:+.4f}, sd {sd_c:.4f}, max {worst:+.4f}")
    print(f"  treatment is {z:.1f} standard deviations above the null mean")

    print("\n" + "=" * 66)
    if dlo > worst:
        print(f"CAUSAL. The ablation costs {dmg:+.4f} nats [95% CI {dlo:+.4f}, {dhi:+.4f}],")
        print(f"and even the low end of that interval exceeds the worst of "
              f"{args.n_controls} size-matched")
        print(f"controls ({worst:+.4f}). First-copy loss moves only {drf:+.4f}, so this is")
        print("not general damage. The heads are doing the copying, not watching it.")
    elif dmg > worst:
        print(f"SUGGESTIVE. The ablation costs {dmg:+.4f} nats, more than any control")
        print(f"({worst:+.4f}), but the confidence interval overlaps the null. Consistent")
        print("with the heads contributing without being the whole mechanism.")
    else:
        print(f"NOT SUPPORTED. Ablating costs {dmg:+.4f} nats, no more than size-matched")
        print(f"controls ({worst:+.4f}). The attention pattern is real; this experiment")
        print("does not show the heads are causally responsible.")
    print("=" * 66)


if __name__ == "__main__":
    main()
