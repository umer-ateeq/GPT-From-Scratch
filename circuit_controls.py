"""Three controls that decide whether the induction finding survives scrutiny.

`induction_heads.py` shows two heads attend to the induction target.
`ablate_heads.py` shows removing them costs the model dearly. Neither of those
rules out the three most likely ways the result could still be wrong. This
script runs the checks that do.

**Control 1 — K-composition.** An induction head needs a previous-token head to
have written "the token before me was X" into each position first, otherwise it
has nothing to match on. If L5.H11 really feeds L6.H9, then ablating L5.H11
should degrade L6.H9's *attention pattern*, not just the model's loss. If the
pattern is unmoved, the two heads are independent and the word "circuit" is not
earned. This is the test that separates a circuit from two correlated heads.

**Control 2 — is it induction or a fixed positional offset?** Every sequence in
the main probe has the same repeat period L, so "attends to i-L+1" and "attends
to absolute offset i-47" are the same measurement. This model uses learned
absolute positional embeddings, so an offset-selective head is expressible. Vary
L: a genuine induction head tracks the repeat period, a positional head does not.

**Control 3 — the positional baseline.** Positions 48-95 have more context than
positions 0-47 whether or not anything repeats, so part of the first-copy to
second-copy loss drop is plain context length, not copying. Measuring that
component on a NON-repeated sequence lets it be subtracted from both the benefit
and the ablation damage, which is the difference between a headline number that
is roughly right and one that is correct.

Usage:
    python circuit_controls.py --ckpt weights8b_300epoch.pth
"""
import argparse

import torch

from ablate_heads import hooks_for, measure, split_loss
from config import GPT_CONFIG_134M
from induction_heads import induction_scores, uniform_baseline
from model import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--n-seqs", type=int, default=32)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--prev-head", default="5.11", help="the previous-token head")
    p.add_argument("--induction-heads", nargs="*", default=["6.9", "7.8"])
    return p.parse_args()


def parse_head(spec):
    layer, head = spec.split(".")
    return int(layer), int(head)


def main():
    args = parse_args()
    model = load_checkpoint(args.ckpt, GPT_CONFIG_134M, device="cpu")
    model.eval()
    vocab = GPT_CONFIG_134M["vocab_size"]
    prev = parse_head(args.prev_head)
    induction = [parse_head(h) for h in args.induction_heads]
    L = args.seq_len

    # ---------------------------------------------------------------- control 1
    print("=" * 72)
    print(f"CONTROL 1  K-composition: does ablating L{prev[0]}.H{prev[1]} "
          f"degrade the induction PATTERN?")
    print("=" * 72)

    intact, _ = induction_scores(model, L, 16, vocab, args.seed)
    handles = [m.register_forward_pre_hook(fn) for m, fn in hooks_for(model, [prev])]
    ablated, _ = induction_scores(model, L, 16, vocab, args.seed)
    for h in handles:
        h.remove()

    baseline = uniform_baseline(L)
    print(f"  uniform baseline {baseline:.4f}\n")
    print(f"  {'head':>8}  {'intact':>8}  {'prev ablated':>13}  {'fall':>7}")
    worst = 0.0
    for layer, head in induction:
        a, b = intact[layer, head].item(), ablated[layer, head].item()
        fall = (a - b) / a * 100
        worst = max(worst, fall)
        print(f"  {f'L{layer}.H{head}':>8}  {a:8.4f}  {b:13.4f}  {fall:6.1f}%")

    print(f"\n  VERDICT: {'K-composition SUPPORTED' if worst > 30 else 'NOT SUPPORTED'}"
          f" (largest fall {worst:.1f}%)")
    if worst > 30:
        print("  The upstream head is feeding the induction heads' key-side match,")
        print("  which is what makes this a circuit rather than two correlated heads.")

    # ---------------------------------------------------------------- control 2
    print()
    print("=" * 72)
    print("CONTROL 2  Induction, or a fixed positional offset? Vary the period.")
    print("=" * 72)
    print("  A positional head keyed to a constant offset cannot track a changing")
    print("  repeat period. An induction head follows it.\n")
    print(f"  {'period L':>10}  {'score':>8}  {'x uniform':>10}")
    scores = []
    for period in (L * 2 // 3, L, L + L // 6):
        s, _ = induction_scores(model, period, 8, vocab, args.seed + 1)
        val = s[induction[0][0], induction[0][1]].item()
        scores.append(val)
        print(f"  {period:>10}  {val:8.4f}  {val / uniform_baseline(period):9.1f}x")
    spread = (max(scores) - min(scores)) / max(scores) * 100
    print(f"\n  VERDICT: score varies {spread:.0f}% across repeat periods. "
          f"{'Tracks the period, so induction.' if spread < 30 else 'Unstable; investigate.'}")

    # ---------------------------------------------------------------- control 3
    print()
    print("=" * 72)
    print("CONTROL 3  Positional baseline: how much of the gap is just context?")
    print("=" * 72)

    generator = torch.Generator().manual_seed(args.seed)
    firsts, seconds = [], []
    for _ in range(args.n_seqs):
        tokens = torch.randint(0, vocab, (1, 2 * L), generator=generator)  # NOT repeated
        a, b = split_loss(model, tokens, L)
        firsts.append(a)
        seconds.append(b)
    pos_gap = sum(firsts) / len(firsts) - sum(seconds) / len(seconds)

    rep_first, rep_second, _, _ = measure(model, L, args.n_seqs, vocab, args.seed)
    raw_gap = rep_first - rep_second
    true_copy = raw_gap - pos_gap

    ab_first, ab_second, _, _ = measure(model, L, args.n_seqs, vocab, args.seed,
                                        hooks=hooks_for(model, induction))
    damage = ab_second - rep_second

    print(f"  non-repeated sequence, 1st half minus 2nd half : {pos_gap:+.4f} nats")
    print(f"     this is pure context length, no copying possible\n")
    print(f"  repeated sequence, raw gap                     : {raw_gap:+.4f} nats")
    print(f"  minus the positional component                 : {true_copy:+.4f} nats")
    print(f"     <- this is the real copying benefit\n")
    print(f"  damage from ablating "
          f"{', '.join(f'L{l}.H{h}' for l, h in induction):<22}: {damage:+.4f} nats")
    print(f"\n  share of TRUE copying destroyed  : {damage / true_copy * 100:.1f}%")
    print(f"  share of RAW gap destroyed       : {damage / raw_gap * 100:.1f}%  "
          f"<- the uncorrected figure")
    print("\n  The corrected number is the one to report. Without subtracting the")
    print("  positional baseline, the residual looks like surviving copying when")
    print("  it is mostly just context length.")


if __name__ == "__main__":
    main()
