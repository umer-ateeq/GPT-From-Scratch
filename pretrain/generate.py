"""Sample text from a trained checkpoint.

Two decoding strategies, both written out rather than imported:

  greedy      always take the highest-probability token. Deterministic, and
              quickly falls into repetition loops, because the most likely
              continuation of a repeated phrase is more of the same phrase.
  temperature reshape the distribution before sampling. Below 1.0 sharpens it
  + top-k     toward the likely tokens, above 1.0 flattens it. top-k first
              discards everything outside the k most likely tokens, which stops
              the long tail of 50,000 near-zero-probability tokens from
              occasionally producing nonsense.

Context handling: the released checkpoint allocates 256 positional embeddings
but only positions 0-127 were ever trained (see interp/CHECKPOINT_AUDIT.md). Conditioning on
more than 128 tokens therefore indexes embedding rows that never received a
gradient, and quality degrades sharply. `--context 128` is the default for that
reason, and the sampler crops its conditioning window rather than letting it grow.

Usage:
    python generate.py --ckpt weights.pth --prompt "Photosynthesis is"
    python generate.py --ckpt weights.pth --temperature 0.8 --top-k 50
    python generate.py --ckpt weights.pth --greedy
"""
import argparse

import tiktoken
import torch

from config import GPT_CONFIG_134M
from model import load_checkpoint

# Prompts spanning the domains FineWeb-Edu actually covers: biology, history,
# maths, physics. Fixed in the source so the sample set cannot be cherry-picked
# after seeing the output.
DEFAULT_PROMPTS = [
    "The mitochondria is",
    "In 1969, the first",
    "Photosynthesis is the process by which",
    "The main causes of the French Revolution were",
    "To solve a quadratic equation, you",
    "Water boils at",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default=None,
                   help="a single prompt; omit to run the built-in prompt set")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--greedy", action="store_true",
                   help="argmax decoding; ignores temperature and top-k")
    p.add_argument("--context", type=int, default=128,
                   help="conditioning window. 128 is the trained context.")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", default=None, help="write the samples to a markdown file")
    return p.parse_args()


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_checkpoint(args.ckpt, GPT_CONFIG_134M, device)
    tokenizer = tiktoken.get_encoding("gpt2")
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    temperature = 0.0 if args.greedy else args.temperature
    top_k = None if args.greedy else args.top_k

    results = []
    for prompt in prompts:
        ids = text_to_token_ids(prompt, tokenizer).to(device)
        out = generate(model, ids, args.max_new_tokens, args.context,
                       temperature=temperature, top_k=top_k)
        text = token_ids_to_text(out, tokenizer)
        results.append((prompt, text))
        print(f"\n--- {prompt!r} ---\n{text}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# Sample completions\n\n")
            f.write(f"From `{args.ckpt}` on {device.type.upper()}. "
                    f"{'Greedy decoding' if args.greedy else f'Temperature {args.temperature}, top-k {args.top_k}'}, "
                    f"{args.max_new_tokens} new tokens, conditioning context "
                    f"{args.context}, seed {args.seed}.\n\n")
            f.write("Reproduce:\n\n```bash\npython generate.py --ckpt "
                    f"{args.ckpt} --out SAMPLES.md\n```\n\n")
            f.write("Every prompt in `DEFAULT_PROMPTS` appears below, in source "
                    "order, from one seeded run. Nothing is selected after the "
                    "fact.\n\n")
            for prompt, text in results:
                f.write(f"### {prompt!r}\n\n```\n{text}\n```\n\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
