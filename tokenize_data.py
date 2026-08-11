"""Turn a streaming text dataset into a flat uint16 file of GPT-2 BPE token IDs.

This is the step that built the 8B-token training corpus. The constraint was a
free notebook session: the full FineWeb-Edu snapshot is far larger than the
available disk and RAM, so the dataset is *streamed* rather than downloaded,
tokenized in chunks, and written straight into a pre-allocated memory-mapped
file. Peak RAM stays at roughly one chunk regardless of corpus size.

uint16 is deliberate: the GPT-2 vocabulary tops out at token ID 50256, which
fits in 16 bits. Storing IDs as int32 or int64 would double or quadruple a 16 GB
file for no benefit.

`process_streaming_dataset` below is the original notebook function, unchanged.
Everything around it is a command-line wrapper so the dataset, crawl and token
budget are flags rather than being edited in two separate cells.

Usage:
    # the 8B-token training corpus
    python tokenize_data.py --out train.bin --max-tokens 8e9 --dump CC-MAIN-2024-10

    # a small held-out set from a DIFFERENT crawl, so it is disjoint from training
    python tokenize_data.py --out validation.bin --max-tokens 5e6 --dump CC-MAIN-2024-18

    # TinyStories instead of FineWeb-Edu
    python tokenize_data.py --dataset roneneldan/TinyStories --out train.bin --max-tokens 5e8
"""
import argparse
import gc
import os

import numpy as np
import tiktoken
from tqdm.auto import tqdm

enc = tiktoken.get_encoding("gpt2")


# ---------------------------------------------------------------------------
# Original notebook function, unchanged apart from taking the output filename as
# an argument instead of hard-coding "train.bin", so the same code can build the
# validation set too.
# ---------------------------------------------------------------------------

def process_streaming_dataset(dataset, out_path="train.bin",
                              max_tokens=10_000_000_000, chunk_size=10000):
    """
    Process streaming dataset in chunks to avoid memory issues
    """

    if not os.path.exists(out_path):
        print(f"Creating {out_path} file for up to {max_tokens:,} tokens...")

        # Pre-allocate memory-mapped file
        dtype = np.uint16
        arr = np.memmap(out_path, dtype=dtype, mode='w+', shape=(max_tokens,))

        current_pos = 0
        processed_examples = 0
        chunk_buffer = []

        print("Starting tokenization...")

        for example in tqdm(dataset, desc="Processing examples"):
            # Tokenize the text
            ids = enc.encode_ordinary(example['text'])

            # Skip very short examples (less than 10 tokens)
            if len(ids) < 10:
                continue

            chunk_buffer.extend(ids)
            processed_examples += 1

            # Process in chunks to manage memory
            if len(chunk_buffer) >= chunk_size:
                # Check if we have space
                if current_pos + len(chunk_buffer) >= max_tokens:
                    print(f"Reached maximum tokens limit: {max_tokens:,}")
                    break

                # Write chunk to file
                arr[current_pos:current_pos + len(chunk_buffer)] = chunk_buffer
                current_pos += len(chunk_buffer)
                chunk_buffer = []

                # Memory cleanup
                if processed_examples % 5000 == 0: #try 5000!
                    gc.collect()
                    print(f"Processed {processed_examples:,} examples, {current_pos:,} tokens")

        # Write remaining buffer
        if chunk_buffer and current_pos + len(chunk_buffer) < max_tokens:
            arr[current_pos:current_pos + len(chunk_buffer)] = chunk_buffer
            current_pos += len(chunk_buffer)

        # Resize array to actual size
        arr.flush()
        del arr

        # Trim the pre-allocated file down to the tokens actually written.
        #
        # CHANGED from the notebook: the original copied the whole array into a
        # second file ("train_final.bin") and renamed it, which needs twice the
        # disk. For an 8B-token corpus that is 32 GB instead of 16 GB, and a free
        # session does not have it. Truncating in place is equivalent.
        with open(out_path, "r+b") as f:
            f.truncate(current_pos * 2)  # uint16 = 2 bytes per token

        print(f"Tokenization complete! Final file size: {current_pos:,} tokens")
        return current_pos
    else:
        # File already exists, get its size
        existing_data = np.memmap(out_path, dtype=np.uint16, mode='r')
        print(f"Using existing {out_path} with {len(existing_data):,} tokens")
        return len(existing_data)


# ---------------------------------------------------------------------------
# Command-line wrapper
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu",
                   help="HuggingFace dataset id")
    p.add_argument("--dump", default="CC-MAIN-2024-10",
                   help="FineWeb-Edu crawl (dataset config). Ignored for other "
                        "datasets. Use a different crawl for validation so it is "
                        "disjoint from training.")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="train.bin", help="output uint16 token file")
    p.add_argument("--max-tokens", type=float, default=8e9)
    p.add_argument("--chunk-size", type=int, default=1_000_000,
                   help="buffer this many tokens before each write")
    return p.parse_args()


def main():
    args = parse_args()

    from datasets import load_dataset

    # FineWeb-Edu needs a crawl name; most other datasets do not take one.
    name = args.dump if "fineweb" in args.dataset else None
    print(f"Loading dataset with streaming: {args.dataset}"
          + (f" [{name}]" if name else ""))
    ds = load_dataset(args.dataset, name=name, split=args.split, streaming=True)
    print("Dataset loaded successfully with streaming!")

    total = process_streaming_dataset(
        ds, out_path=args.out,
        max_tokens=int(args.max_tokens),
        chunk_size=args.chunk_size,
    )
    print(f"Dataset processing complete! Total tokens: {total:,} "
          f"({total * 2 / 1e9:.2f} GB on disk)")


if __name__ == "__main__":
    main()
