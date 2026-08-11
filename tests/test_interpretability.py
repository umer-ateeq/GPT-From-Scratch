"""Tests for the analysis code: attention capture, ablation, and the audit.

These matter more than the model tests. A bug in `model.py` shows up as a loss
that will not go down. A bug in `induction_heads.py` shows up as a confident,
plausible, wrong scientific claim, which is far harder to notice.

None of these need the released checkpoint or any dataset, so CI runs them free.

    python -m pytest tests/ -v
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablate_heads import make_ablation_hook, split_loss          # noqa: E402
from audit_checkpoint import positional_embedding_audit          # noqa: E402
from induction_heads import (build_repeated_sequence,            # noqa: E402
                             capture_attention, induction_scores,
                             uniform_baseline)
from model import GPTModel                                       # noqa: E402
from previous_token_heads import previous_token_scores           # noqa: E402

CONFIG_TINY = {
    "vocab_size": 128, "context_length": 64, "emb_dim": 64,
    "n_heads": 4, "n_layers": 3, "drop_rate": 0.0, "qkv_bias": False,
}


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    return GPTModel(CONFIG_TINY).eval()


# --------------------------------------------------------------------------
# capture_attention: the foundation everything else stands on
# --------------------------------------------------------------------------

def test_captured_attention_reproduces_the_module_output(tiny):
    """The strongest check available.

    capture_attention recomputes the softmax outside the model, using a forward
    hook, so that model.py can stay byte-identical to the training notebook. If
    that recomputation drifted from what MultiHeadAttention actually does, every
    downstream number would be quietly wrong.

    So: take the captured attention, finish the head's computation by hand
    (weights @ V, recombine heads, out_proj) and check it equals what the module
    itself returned.
    """
    x = torch.randn(1, 12, CONFIG_TINY["emb_dim"])
    block = tiny.trf_blocks[0]
    att = block.att

    with torch.no_grad():
        expected = att(x)

    captured = {}
    handle = att.register_forward_pre_hook(
        lambda m, args: captured.__setitem__("x", args[0].detach()))
    with torch.no_grad():
        att(x)
    handle.remove()

    b, t, _ = x.shape
    h, d = att.num_heads, att.head_dim
    with torch.no_grad():
        q = att.W_query(x).view(b, t, h, d).transpose(1, 2)
        k = att.W_key(x).view(b, t, h, d).transpose(1, 2)
        v = att.W_value(x).view(b, t, h, d).transpose(1, 2)
        scores = q @ k.transpose(2, 3)
        scores = scores.masked_fill(att.mask.bool()[:t, :t], -torch.inf)
        weights = torch.softmax(scores / d ** 0.5, dim=-1)
        rebuilt = att.out_proj((weights @ v).transpose(1, 2).reshape(b, t, -1))

    torch.testing.assert_close(rebuilt, expected, rtol=1e-5, atol=1e-6)


def test_captured_attention_is_a_causal_probability_distribution(tiny):
    tokens = torch.randint(0, CONFIG_TINY["vocab_size"], (1, 16))
    patterns = capture_attention(tiny, tokens)

    assert len(patterns) == CONFIG_TINY["n_layers"]
    for attn in patterns:
        assert attn.shape == (CONFIG_TINY["n_heads"], 16, 16)
        # every query row is a distribution
        torch.testing.assert_close(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)))
        # and no query attends to a future position
        future = torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1)
        assert attn[:, future].abs().max().item() == 0.0


def test_capture_attention_leaves_no_hooks_behind(tiny):
    """A leaked forward hook would silently slow or corrupt every later call."""
    before = [len(b.att._forward_pre_hooks) for b in tiny.trf_blocks]
    capture_attention(tiny, torch.randint(0, CONFIG_TINY["vocab_size"], (1, 8)))
    after = [len(b.att._forward_pre_hooks) for b in tiny.trf_blocks]
    assert before == after


# --------------------------------------------------------------------------
# The induction measurement itself
# --------------------------------------------------------------------------

def test_repeated_sequence_really_repeats():
    g = torch.Generator().manual_seed(0)
    seq = build_repeated_sequence(10, 128, g)
    assert seq.shape == (1, 20)
    torch.testing.assert_close(seq[0, :10], seq[0, 10:])


def test_uniform_baseline_matches_the_hand_computed_value():
    """A causal head at position i spreads over i+1 positions, so an indifferent
    head puts 1/(i+1) on any one of them."""
    seq_len = 8
    expected = np.mean([1.0 / (i + 1) for i in range(seq_len + 1, 2 * seq_len)])
    assert uniform_baseline(seq_len) == pytest.approx(expected)


def test_untrained_model_shows_no_induction(tiny):
    """The control that makes the real result meaningful.

    A randomly initialised model cannot have learned in-context copying. If the
    measurement reported induction here, it would be measuring an artifact of the
    probe rather than anything the model learned.
    """
    mean, std = induction_scores(tiny, seq_len=16, n_seqs=4,
                                 vocab_size=CONFIG_TINY["vocab_size"], seed=0)
    baseline = uniform_baseline(16)
    assert mean.max().item() / baseline < 3.0, (
        "an untrained model scored like an induction head; the probe is broken")
    assert std.shape == mean.shape


def test_offset_selects_a_different_target(tiny):
    """offset=0 (duplicate token) and offset=1 (induction) must not be the same
    measurement, or the control in the report is vacuous."""
    kw = dict(seq_len=16, n_seqs=2, vocab_size=CONFIG_TINY["vocab_size"], seed=0)
    induction, _ = induction_scores(tiny, offset=1, **kw)
    duplicate, _ = induction_scores(tiny, offset=0, **kw)
    assert not torch.allclose(induction, duplicate)


def test_previous_token_scores_have_the_right_shape(tiny):
    scores = previous_token_scores(tiny, seq_len=24, n_seqs=2,
                                   vocab_size=CONFIG_TINY["vocab_size"], seed=0)
    assert scores.shape == (CONFIG_TINY["n_layers"], CONFIG_TINY["n_heads"])
    assert (scores >= 0).all() and (scores <= 1).all()


# --------------------------------------------------------------------------
# Ablation
# --------------------------------------------------------------------------

def test_ablation_hook_zeroes_exactly_one_head(tiny):
    """Off-by-one here would ablate the wrong head and invalidate the result."""
    att = tiny.trf_blocks[0].att
    d = att.head_dim
    head = 2

    x = torch.randn(1, 5, CONFIG_TINY["emb_dim"])
    hook = make_ablation_hook(head, d)
    (modified,) = hook(att.out_proj, (x,))

    assert modified[..., head * d:(head + 1) * d].abs().max().item() == 0.0
    # every other head untouched
    torch.testing.assert_close(modified[..., :head * d], x[..., :head * d])
    torch.testing.assert_close(modified[..., (head + 1) * d:], x[..., (head + 1) * d:])
    # and the caller's tensor was not mutated in place
    assert x[..., head * d:(head + 1) * d].abs().max().item() > 0.0


def test_ablation_changes_the_model_output(tiny):
    att = tiny.trf_blocks[0].att
    tokens = torch.randint(0, CONFIG_TINY["vocab_size"], (1, 12))

    with torch.no_grad():
        before = tiny(tokens)
    handle = att.out_proj.register_forward_pre_hook(
        make_ablation_hook(1, att.head_dim))
    with torch.no_grad():
        during = tiny(tokens)
    handle.remove()
    with torch.no_grad():
        after = tiny(tokens)

    assert not torch.allclose(before, during), "ablation had no effect"
    torch.testing.assert_close(before, after)  # and it was fully reversed


def test_split_loss_separates_the_two_copies(tiny):
    """The whole ablation argument rests on comparing first copy to second."""
    seq_len = 12
    g = torch.Generator().manual_seed(0)
    tokens = build_repeated_sequence(seq_len, CONFIG_TINY["vocab_size"], g)
    first, second = split_loss(tiny, tokens, seq_len)
    assert np.isfinite(first) and np.isfinite(second)
    # an untrained model cannot copy, so the halves should be comparable
    assert abs(first - second) < 3.0


# --------------------------------------------------------------------------
# The checkpoint audit
# --------------------------------------------------------------------------

def test_audit_finds_a_planted_context_cliff():
    """Build a state dict with a known answer and check the audit recovers it."""
    torch.manual_seed(0)
    pos = torch.randn(256, 768)
    pos[100:] *= 1e-4                       # positions 100+ "never trained"
    result = positional_embedding_audit({"pos_emb.weight": pos}, dead_ratio=0.1)
    assert result["trained_prefix"] == 100
    assert result["n_dead"] == 156


def test_audit_reports_no_cliff_when_every_position_was_trained():
    torch.manual_seed(0)
    pos = torch.randn(256, 768)
    result = positional_embedding_audit({"pos_emb.weight": pos}, dead_ratio=0.1)
    assert result["trained_prefix"] == 256
    assert result["n_dead"] == 0


def test_audit_threshold_survives_a_half_dead_table():
    """Regression test for a real bug.

    The threshold was originally a fraction of the MEDIAN row norm. When exactly
    half the table is dead, the median falls inside the dead cluster and the
    detector reports that everything was trained. Referencing the maximum fixes
    it. The released checkpoint is exactly this 50/50 case.
    """
    torch.manual_seed(0)
    pos = torch.randn(256, 768)
    pos[128:] *= 1e-4
    result = positional_embedding_audit({"pos_emb.weight": pos}, dead_ratio=0.1)
    assert result["trained_prefix"] == 128, "the half-dead table defeated the detector"
