"""Turn a text dataset into a flat uint16 file of GPT-2 BPE token IDs.

This is the step that produced the 8B-token training corpus. The constraint was
a free Colab session: the full FineWeb-Edu snapshot is far larger than the
available disk and RAM, so the dataset is *streamed* rather than downloaded,
tokenized in chunks, and written straight into a pre-allocated memory-mapped
file. Peak RAM stays at roughly one chunk regardless of corpus size.

uint16 is deliberate: the GPT-2 vocabulary tops out at token ID 50256, which
fits in 16 bits. Storing IDs as int32 or int64 would double or quadruple a
16 GB file for no benefit.

Usage:
    # the 8B-token training corpus (long: this is the expensive step)
    python tokenize_data.py --out train.bin --max-tokens 8e9 --dump CC-MAIN-2024-10

    # a small held-out set from a DIFFERENT crawl, so it is disjoint from training
    python tokenize_data.py --out validation.bin --max-tokens 5e6 --dump CC-MAIN-2024-18

    # TinyStories instead of FineWeb-Edu (much smaller, useful for a quick run)
    python tokenize_data.py --dataset roneneldan/TinyStories --out train.bin --max-tokens 5e8
"""
import argparse
import os

import numpy as np
import tiktoken
from tqdm.auto import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu",
                   help="HuggingFace dataset id")
    p.add_argument("--dump", default="CC-MAIN-2024-10",
                   help="FineWeb-Edu crawl (dataset config). Ignored for other datasets. "
                        "Use a different crawl for validation so it is disjoint from training.")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="train.bin", help="output uint16 token file")
    p.add_argument("--max-tokens", type=float, default=8e9,
                   help="stop after writing this many tokens")
    p.add_argument("--min-doc-tokens", type=int, default=10,
                   help="skip documents shorter than this; they are mostly boilerplate")
    p.add_argument("--chunk-tokens", type=int, default=1_000_000,
                   help="buffer this many tokens before each write")
    return p.parse_args()


def main():
    args = parse_args()
    max_tokens = int(args.max_tokens)

    if os.path.exists(args.out):
        existing = np.memmap(args.out, dtype=np.uint16, mode="r")
        print(f"{args.out} already exists with {len(existing):,} tokens. "
              f"Delete it first to rebuild.")
        return

    from datasets import load_dataset

    # FineWeb-Edu needs a crawl name; most other datasets do not take one.
    name = args.dump if "fineweb" in args.dataset else None
    print(f"streaming {args.dataset}" + (f" [{name}]" if name else ""))
    ds = load_dataset(args.dataset, name=name, split=args.split, streaming=True)

    enc = tiktoken.get_encoding("gpt2")

    # Pre-allocate the full file, then truncate at the end. Growing a memmap
    # incrementally would mean repeated reallocation and copying.
    tmp = args.out + ".tmp"
    arr = np.memmap(tmp, dtype=np.uint16, mode="w+", shape=(max_tokens,))

    pos = 0
    buffer = []
    skipped = 0
    progress = tqdm(total=max_tokens, unit="tok", unit_scale=True, desc=args.out)

    for example in ds:
        # encode_ordinary ignores special tokens such as <|endoftext|>, so no
        # control token from the source text can leak into the corpus.
        ids = enc.encode_ordinary(example["text"])
        if len(ids) < args.min_doc_tokens:
            skipped += 1
            continue

        buffer.extend(ids)

        if len(buffer) >= args.chunk_tokens:
            take = min(len(buffer), max_tokens - pos)
            arr[pos:pos + take] = buffer[:take]
            pos += take
            progress.update(take)
            buffer = []
            if pos >= max_tokens:
                break

    # Flush whatever is left in the buffer
    if buffer and pos < max_tokens:
        take = min(len(buffer), max_tokens - pos)
        arr[pos:pos + take] = buffer[:take]
        pos += take
        progress.update(take)

    progress.close()
    arr.flush()
    del arr

    # Cut the file down to what was actually written. uint16 = 2 bytes per token.
    with open(tmp, "r+b") as f:
        f.truncate(pos * 2)
    os.rename(tmp, args.out)

    print(f"wrote {args.out}: {pos:,} tokens ({pos * 2 / 1e9:.2f} GB)")
    print(f"skipped {skipped:,} documents shorter than {args.min_doc_tokens} tokens")


if __name__ == "__main__":
    main()
