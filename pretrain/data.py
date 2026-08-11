"""Batch sampling from a memory-mapped token file.

The corpus is a flat array of uint16 token IDs on disk, several gigabytes of it.
np.memmap lets the operating system page in only the slices actually touched, so
training reads from an 8B-token file while using almost no RAM.

A batch is `batch_size` random windows of `block_size` tokens. Targets are the
same windows shifted one position right, which is the whole of the next-token
prediction objective: predict token i+1 from tokens 0..i.

Two details in the sampler are load-bearing:

  1. batch_size, block_size and the file path are ARGUMENTS, never module
     globals. A sampler that reads its shape from globals at call time can
     silently disagree with the configuration it was supposed to run at, and
     nothing raises. tests/test_model.py asserts the shape follows the arguments.
  2. torch.randint's upper bound is len(data) - block_size - 1 rather than
     len(data) - block_size, because the TARGET window reaches one token further
     than the input window and could otherwise run off the end of the file.
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
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    data = np.memmap(path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device.type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
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
