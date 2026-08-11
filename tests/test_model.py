"""Tests that need no checkpoint and no dataset, so anyone can run them for free.

Each one pins a property the rest of the repository's claims depend on: the
parameter count, that attention cannot see the future, that every parameter
receives gradient, and that the batch shape and learning rate come from the
arguments they are given.

    python -m pytest tests/ -v
"""
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from config import GPT_CONFIG_134M          # noqa: E402
from data import get_batch                  # noqa: E402
from model import GPTModel                  # noqa: E402
from train import calc_loss_batch, get_lr   # noqa: E402

# A tiny model keeps CI fast. Shape and wiring logic does not depend on width.
CONFIG_TINY = {
    "vocab_size": 512, "context_length": 32, "emb_dim": 64,
    "n_heads": 4, "n_layers": 2, "drop_rate": 0.0, "qkv_bias": False,
}


def test_parameter_count_is_exactly_134M():
    """The headline number. 134,077,440 is exact, not rounded."""
    model = GPTModel(GPT_CONFIG_134M)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 134_077_440

    # One causal mask buffer per layer, 256x256 each. Not trainable.
    buffers = sum(b.numel() for b in model.buffers())
    assert buffers == 256 * 256 * 8
    assert trainable + buffers == 134_601_728


def test_published_parameter_breakdown_sums_to_the_total():
    """The per-component table in the README must reconcile.

    A breakdown that does not add up to its own total is the cheapest possible
    thing for a reader to check and the most expensive one to get wrong, so the
    published figures are pinned here rather than maintained by hand.
    """
    model = GPTModel(GPT_CONFIG_134M)
    published = {
        "tok_emb": 38_597_376,
        "pos_emb": 196_608,
        "trf_blocks": 56_684_544,
        "final_norm": 1_536,
        "out_head": 38_597_376,
    }
    actual = {
        "tok_emb": model.tok_emb.weight.numel(),
        "pos_emb": model.pos_emb.weight.numel(),
        "trf_blocks": sum(p.numel() for p in model.trf_blocks.parameters()),
        "final_norm": sum(p.numel() for p in model.final_norm.parameters()),
        "out_head": model.out_head.weight.numel(),
    }
    assert actual == published
    assert sum(published.values()) == 134_077_440


def test_untied_output_head_is_a_separate_matrix():
    """Tying the head to the embedding would silently drop 38.6M parameters."""
    model = GPTModel(GPT_CONFIG_134M)
    assert model.out_head.weight.data_ptr() != model.tok_emb.weight.data_ptr()
    assert model.out_head.weight.shape == (50257, 768)
    assert model.out_head.bias is None


def test_forward_returns_one_logit_per_vocabulary_entry():
    model = GPTModel(CONFIG_TINY).eval()
    x = torch.randint(0, CONFIG_TINY["vocab_size"], (3, 16))
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (3, 16, CONFIG_TINY["vocab_size"])
    assert torch.isfinite(logits).all()


def test_attention_cannot_see_the_future():
    """The causal mask is the whole basis of next-token training.

    Changing the last token must not change any earlier position's logits. If it
    does, the model is reading the answer and every loss number is meaningless.
    """
    torch.manual_seed(0)
    model = GPTModel(CONFIG_TINY).eval()
    a = torch.randint(0, CONFIG_TINY["vocab_size"], (1, 12))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % CONFIG_TINY["vocab_size"]
    with torch.no_grad():
        la, lb = model(a), model(b)
    torch.testing.assert_close(la[:, :-1, :], lb[:, :-1, :])


def test_every_parameter_receives_gradient():
    """A parameter with no gradient is a silently dead part of the network."""
    model = GPTModel(CONFIG_TINY)
    x = torch.randint(0, CONFIG_TINY["vocab_size"], (2, 16))
    y = torch.randint(0, CONFIG_TINY["vocab_size"], (2, 16))
    calc_loss_batch(x, y, model, torch.device("cpu")).backward()
    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None]
    assert not dead, f"no gradient reached: {dead}"


def test_untrained_loss_is_close_to_random_guessing():
    """A sanity check on the loss itself.

    Before training, the model has no information, so cross-entropy should be
    about ln(vocab_size). A loss far from that at init means the loss is being
    computed over the wrong axis, which is an easy mistake to make when
    flattening (batch, tokens, vocab).
    """
    torch.manual_seed(0)
    model = GPTModel(CONFIG_TINY).eval()
    x = torch.randint(0, CONFIG_TINY["vocab_size"], (4, 16))
    y = torch.randint(0, CONFIG_TINY["vocab_size"], (4, 16))
    with torch.no_grad():
        loss = calc_loss_batch(x, y, model, torch.device("cpu")).item()
    assert abs(loss - math.log(CONFIG_TINY["vocab_size"])) < 0.5


def test_batch_shape_follows_arguments_not_globals(tmp_path):
    """The batch shape must follow the arguments, always.

    A sampler that reads its shape from module globals can silently run at a
    different shape than the configuration says, with nothing raised. Passing the
    shape in makes that unrepresentable; this pins it.
    """
    path = tmp_path / "toy.bin"
    np.arange(5000, dtype=np.uint16).tofile(path)
    device = torch.device("cpu")

    x, y = get_batch(str(path), batch_size=4, block_size=16, device=device)
    assert x.shape == (4, 16) and y.shape == (4, 16)

    # y is x shifted one token left: the next-token objective
    assert torch.equal(x[:, 1:], y[:, :-1])

    x2, _ = get_batch(str(path), batch_size=7, block_size=8, device=device)
    assert x2.shape == (7, 8)


def test_lr_warms_up_then_decays_and_never_climbs():
    """The whole schedule shape, not just its endpoints.

    A cosine floor typed independently of the peak can end up above it, which
    turns the decay into a climb and raises nothing. Deriving the floor from the
    peak makes that impossible; this asserts the resulting shape.
    """
    peak, floor, warmup, total = 6e-4, 6e-5, 100, 1000

    assert get_lr(0, peak, floor, warmup, total) == pytest.approx(peak / warmup)
    assert get_lr(warmup - 1, peak, floor, warmup, total) == pytest.approx(peak)

    after = [get_lr(s, peak, floor, warmup, total) for s in range(warmup, total + 1)]
    assert all(b <= a + 1e-12 for a, b in zip(after, after[1:])), "LR must not rise"
    assert min(after) >= floor - 1e-12
    assert max(after) <= peak + 1e-12
    assert after[-1] == pytest.approx(floor, rel=1e-6)

    # past the end of the schedule it clamps at the floor instead of diverging
    assert get_lr(total * 3, peak, floor, warmup, total) == pytest.approx(floor)


def test_training_run_leaves_a_complete_record(tmp_path):
    """Every run must leave a record behind.

    A run that records nothing cannot be audited afterwards, and a logged token
    count that drifts from what actually passed through the model makes every
    per-token figure unverifiable. Assert config, metrics and summary exist, and
    that the logged token count matches the real one.
    """
    import json

    train_bin, val_bin = tmp_path / "train.bin", tmp_path / "val.bin"
    rng = np.random.default_rng(0)
    rng.integers(0, 512, size=60_000, dtype=np.uint16).tofile(train_bin)
    rng.integers(0, 512, size=20_000, dtype=np.uint16).tofile(val_bin)

    repo = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pretrain")
    out_dir = tmp_path / "runs"
    result = subprocess.run(
        [sys.executable, "train.py",
         "--train-bin", str(train_bin), "--val-bin", str(val_bin),
         "--n-layers", "2", "--n-heads", "4", "--emb-dim", "64", "--context", "32",
         "--batch-size", "4", "--grad-accum", "2", "--train-tokens", "4096",
         "--warmup-steps", "2", "--eval-freq", "2", "--eval-iter", "2",
         "--dtype", "float32", "--out-dir", str(out_dir), "--run-name", "citest"],
        cwd=repo, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr

    run = out_dir / "citest"
    for name in ("config.json", "metrics.jsonl", "summary.json"):
        assert (run / name).exists(), f"{name} missing from the run record"

    config = json.loads((run / "config.json").read_text())
    assert config["args"]["seed"] == 123
    assert config["command"], "the launch command must be recorded"
    for key in ("python_version", "torch_version", "device_name"):
        assert config[key]

    summary = json.loads((run / "summary.json").read_text())
    # tokens_seen must come from the tensors, not from the config:
    # optimizer_steps x grad_accum micro-batches x batch_size x context
    assert summary["tokens_seen"] == summary["optimizer_steps"] * 2 * 4 * 32
    assert math.isfinite(summary["final_val_loss"])
    assert summary["final_val_perplexity"] == pytest.approx(
        math.exp(summary["final_val_loss"]), rel=1e-3)
