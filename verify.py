"""Check this model's perplexity against the numbers the README reports.

    python verify.py --ckpt weights8b_300epoch.pth

Scores the model on WikiText-2, then scores GPT-2-small through the *same*
function, the same tokenizer, the same test set and the same window, so the two
rows differ only by the model. A perplexity number without a reference measured
the same way is not a claim anyone can check.

Add held-out FineWeb-Edu, if you have the tokenized file:

    python verify.py --ckpt weights8b_300epoch.pth --data-bin validation.bin

The full run downloads WikiText-2 and takes about 30 minutes per model on CPU
(measured: 34 min for this model, 28 min for GPT-2-small), and seconds on a GPU.
Use --max-tokens for a fast partial run.

Exit code is 0 only if every measured number matches the README.
"""
import argparse
import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "pretrain"))

from evaluate import (bin_perplexity, load_gpt2_baseline,  # noqa: E402
                      load_scratch_model, strided_perplexity, trained_positions)

# What the README claims. These are the numbers being checked, so they live in
# one place and nowhere else.
CLAIM_WIKITEXT = 184.96
CLAIM_GPT2 = 59.69
CLAIM_RATIO = 3.10
CLAIM_FINEWEB = 38.89

# The window every number in the README was measured at. The checkpoint trained
# at context 128, so scoring wider would put half of each window on positional
# rows that never received gradient.
CONTEXT = 128

TOLERANCE = 0.01  # 1%, loose enough for float and library drift


class Report:
    """Collects rows, prints them aligned, remembers whether anything failed."""

    def __init__(self):
        self.rows = []
        self.failed = 0

    def check(self, name, measured, claimed, ok):
        self.rows.append((name, measured, claimed, bool(ok)))
        if not ok:
            self.failed += 1

    def info(self, name, measured):
        """A number worth printing that has nothing to compare against."""
        self.rows.append((name, measured, "", None))

    def print(self):
        if not self.rows:
            return
        w = max(len(r[0]) for r in self.rows)
        print()
        print(f"  {'':<{w}}  {'measured':>12}  {'README':>12}")
        print("  " + "-" * (w + 34))
        for name, measured, claimed, ok in self.rows:
            flag = "" if ok is None else ("PASS" if ok else "FAIL")
            print(f"  {name:<{w}}  {measured:>12}  {claimed:>12}   {flag}")


def close(measured, claimed):
    return abs(measured - claimed) <= TOLERANCE * abs(claimed)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="the released checkpoint")
    p.add_argument("--data-bin", default=None,
                   help="tokenized held-out FineWeb-Edu, to also check the "
                        "in-domain number")
    p.add_argument("--max-tokens", type=int, default=0,
                   help="score only the first N WikiText-2 tokens. Faster, but "
                        "the result is no longer comparable to the README")
    p.add_argument("--skip-baseline", action="store_true",
                   help="do not run GPT-2-small. Saves half the time and loses "
                        "the only thing that makes the number meaningful")
    return p.parse_args()


def load_wikitext(max_tokens):
    from datasets import load_dataset

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    tokens = torch.tensor(enc.encode_ordinary("\n\n".join(ds["text"])),
                          dtype=torch.long)
    if max_tokens and max_tokens < tokens.size(0):
        tokens = tokens[:max_tokens]
    return tokens


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 1)
    partial = bool(args.max_tokens)
    report = Report()

    forward, model_ctx, n_params, state = load_scratch_model(args.ckpt, device)
    trained = trained_positions(state)

    print(f"model    : from-scratch 134M, {n_params / 1e6:.2f}M parameters")
    print(f"device   : {device}")
    print(f"window   : {CONTEXT}, non-overlapping")
    print(f"context  : {trained} positional rows trained of {model_ctx} allocated")

    if CONTEXT > trained:
        raise SystemExit(
            f"the window ({CONTEXT}) exceeds the trained context ({trained}); "
            f"scoring here would understate the model")

    report.check("parameters", f"{n_params:,}", "134,077,440",
                 n_params == 134_077_440)
    report.check("trained context", str(trained), "128", trained == 128)

    # ---- WikiText-2, out of domain --------------------------------------
    print("\nloading WikiText-2 ...", flush=True)
    tokens = load_wikitext(args.max_tokens)
    print(f"         : {tokens.size(0):,} tokens"
          + (" (PARTIAL RUN)" if partial else " (full test set)"))

    print("scoring this model ...", flush=True)
    t0 = time.time()
    ppl, nll, scored = strided_perplexity(forward, tokens, CONTEXT, CONTEXT, device)
    print(f"         : {ppl:.2f} in {time.time() - t0:.0f}s over {scored:,} tokens")

    if partial:
        report.info("WikiText-2 perplexity", f"{ppl:.2f}")
    else:
        report.check("WikiText-2 perplexity", f"{ppl:.2f}", f"{CLAIM_WIKITEXT:.2f}",
                     close(ppl, CLAIM_WIKITEXT))

    # ---- The same harness, a different model -----------------------------
    if not args.skip_baseline:
        print("\nscoring GPT-2-small on the identical harness ...", flush=True)
        try:
            gpt2_forward, _, gpt2_params, _ = load_gpt2_baseline(device)
        except ImportError:
            print("         : needs `pip install transformers`, skipping")
        else:
            t0 = time.time()
            gpt2_ppl, _, _ = strided_perplexity(
                gpt2_forward, tokens, CONTEXT, CONTEXT, device)
            print(f"         : {gpt2_ppl:.2f} in {time.time() - t0:.0f}s")

            if partial:
                report.info("GPT-2-small perplexity", f"{gpt2_ppl:.2f}")
                report.info("ratio, GPT-2 is better by", f"{ppl / gpt2_ppl:.2f}x")
            else:
                report.check("GPT-2-small perplexity", f"{gpt2_ppl:.2f}",
                             f"{CLAIM_GPT2:.2f}", close(gpt2_ppl, CLAIM_GPT2))
                report.check("ratio, GPT-2 is better by", f"{ppl / gpt2_ppl:.2f}x",
                             f"{CLAIM_RATIO:.2f}x", close(ppl / gpt2_ppl, CLAIM_RATIO))

    # ---- FineWeb-Edu, in domain -----------------------------------------
    if args.data_bin:
        print(f"\nscoring held-out FineWeb-Edu from {args.data_bin} ...", flush=True)
        fw_ppl, _, fw_scored = bin_perplexity(
            forward, args.data_bin, CONTEXT, batch_size=8, device=device,
            max_batches=50)
        print(f"         : {fw_ppl:.2f} over {fw_scored:,} tokens")
        report.check("FineWeb-Edu perplexity", f"{fw_ppl:.2f}", f"{CLAIM_FINEWEB:.2f}",
                     close(fw_ppl, CLAIM_FINEWEB))

    report.print()

    if partial:
        print("\nPartial run: --max-tokens was set, so the perplexities above are")
        print("not the full-test-set numbers the README reports. Drop the flag to")
        print("check them properly.")

    checked = [r for r in report.rows if r[3] is not None]
    print(f"\n{len(checked) - report.failed}/{len(checked)} checks passed")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
