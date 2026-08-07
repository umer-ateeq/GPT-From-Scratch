"""Batch sampling from a memory-mapped token file.

The corpus is a flat array of uint16 token IDs on disk, several gigabytes of it.
`np.memmap` lets the operating system page in only the slices actually touched,
so training reads from an 8B-token file while using almost no RAM.

A batch is `batch_size` random windows of `block_size` tokens. Targets are the
same windows shifted one position right, which is the whole of the next-token
prediction objective: predict token i+1 from tokens 0..i.

Change from the original notebook, made deliberately (see docs/AUDIT.md):
`get_batch` takes `batch_size` and `block_size` as explicit arguments. In the
notebook they were module-level globals read at call time, and a later cell
reassigned both. The sampler silently switched from the configured 64 x 256 to
32 x 128 while every printed summary still reported 64 x 256. Passing them as
arguments makes that class of bug impossible, and
tests/test_model.py asserts the returned shape follows the arguments.
"""
import numpy as np
import torch


def load_tokens(path):
    """Open a uint16 token file without reading it into memory."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def count_tokens(path):
    """How many tokens a .bin file holds."""
    return len(load_tokens(path))


def get_batch(path, batch_size, block_size, device):
    """Sample one batch of random windows from the token file at `path`.

    Returns (x, y), both int64 tensors of shape (batch_size, block_size), where
    y is x shifted forward by one token.

    The memmap is recreated on every call on purpose. Holding one open across
    thousands of iterations leaks memory, because the pages it touches are never
    released. This follows nanoGPT's approach.
    """
    data = np.memmap(path, dtype=np.uint16, mode="r")

    # Random start offsets. The -1 leaves room for the target window, which
    # extends one token further than the input window.
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))

    x = torch.stack([
        torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])

    if device.type == "cuda":
        # Pinned memory can be copied to the GPU asynchronously, so the transfer
        # overlaps with computation instead of blocking on it.
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "train.bin"
    n = count_tokens(path)
    print(f"{path}: {n:,} tokens ({n * 2 / 1e9:.2f} GB on disk as uint16)")

    x, y = get_batch(path, batch_size=2, block_size=8, device=torch.device("cpu"))
    print(f"x shape {tuple(x.shape)}, y shape {tuple(y.shape)}")
    print(f"x[0]: {x[0].tolist()}")
    print(f"y[0]: {y[0].tolist()}   <- x shifted by one")
