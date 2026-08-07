"""The 134M-parameter GPT, written from scratch in PyTorch.

Nothing here is imported from a transformer library. Attention, the feed-forward
block, layer normalization and the causal mask are all written out explicitly,
because the point of the project was to understand them rather than to call them.

Structure, bottom up:

    LayerNorm            normalize each token's feature vector, then rescale
    FeedForward          two linear layers with a ReLU between them, 4x wide
    MultiHeadAttention   causal self-attention over `n_heads` parallel heads
    TransformerBlock     pre-norm attention + pre-norm feed-forward, both residual
    GPTModel             embeddings -> N blocks -> final norm -> vocabulary logits

Architecture is verified against the released checkpoint by
tests/test_model.py, which asserts the exact parameter count of 134,077,440.
"""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer normalization, written out rather than using nn.LayerNorm.

    For each token independently: subtract the mean of its 768 features, divide
    by their standard deviation, then apply a learned per-feature scale and
    shift. This keeps activations in a stable range as they pass through 8
    blocks, which is what makes deep stacks trainable at all.

    `unbiased=False` divides the variance by N rather than N-1, matching GPT-2.
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # guards against division by zero
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
    """Position-wise feed-forward network: 768 -> 3072 -> ReLU -> 768.

    Applied to each token independently. The 4x widening is the standard
    transformer ratio, and this block holds roughly two thirds of the model's
    non-embedding parameters.
    """

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.ReLU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention.

    Each token builds a query, a key and a value. Attention scores are the dot
    products of queries against keys, scaled by 1/sqrt(head_dim) so the softmax
    does not saturate as head_dim grows. The causal mask sets every score for a
    future position to -inf, so after the softmax those positions carry zero
    weight and a token can only attend to itself and its past. That masking is
    what makes next-token prediction a valid training objective: without it the
    model could read the answer.

    The 768 dimensions are split into 12 heads of 64. The heads are computed in
    parallel as a single batched matmul by reshaping to
    (batch, heads, tokens, head_dim), not by looping.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # mixes the heads back together
        self.dropout = nn.Dropout(dropout)

        # Upper-triangular matrix of ones, excluding the diagonal. Registered as
        # a buffer so it moves with .to(device) and is saved in the checkpoint,
        # but receives no gradient.
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)        # (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Split the last dimension into heads:
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Put the head dimension next to the batch so every head is one matmul:
        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # (b, num_heads, num_tokens, num_tokens): how much each token attends to
        # each other token, per head
        attn_scores = queries @ keys.transpose(2, 3)

        # Crop the mask to this sequence length so shorter inputs still work
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values, then put the heads back on the last dimension
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class TransformerBlock(nn.Module):
    """One transformer block: attention, then feed-forward, both residual.

    Pre-norm: normalization happens *before* each sublayer rather than after, so
    the residual path from input to output is never normalized. That keeps
    gradients well behaved through a deep stack and is why this trains without a
    carefully tuned warmup, unlike the original post-norm transformer.
    """

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention sublayer
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Feed-forward sublayer
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """The full decoder-only language model.

    Token embeddings and learned positional embeddings are summed, passed
    through `n_layers` transformer blocks and a final normalization, then
    projected to one logit per vocabulary entry.

    The output head is *untied* from the input embedding: both are
    50257 x 768 = 38.6M parameters, and keeping them separate is what puts this
    model at 134M rather than the 96M a tied version would have.
    """

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # One positional vector per slot 0..seq_len-1, added to every sequence
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)

    def param_summary(self):
        """Parameter counts, separating weights from the non-trainable masks."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buffers = sum(b.numel() for b in self.buffers())
        return {"trainable": trainable, "buffers": buffers,
                "total": trainable + buffers}


def load_checkpoint(path, cfg, device="cpu"):
    """Build the model and load weights from a checkpoint file.

    Accepts both a bare state_dict (the format of the released checkpoint) and a
    dict with the weights under "model", so newer checkpoints saved by train.py
    load through the same function.
    """
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model = GPTModel(cfg)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


if __name__ == "__main__":
    from config import GPT_CONFIG_134M, describe

    model = GPTModel(GPT_CONFIG_134M)
    summary = model.param_summary()
    print(f"architecture : {describe()}")
    print(f"trainable    : {summary['trainable']:,} ({summary['trainable'] / 1e6:.2f}M)")
    print(f"mask buffers : {summary['buffers']:,}")
    print(f"total tensors: {summary['total']:,}")
