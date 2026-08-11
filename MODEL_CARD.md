---
license: mit
language:
  - en
library_name: pytorch
pipeline_tag: text-generation
tags:
  - gpt
  - from-scratch
  - pretraining
  - educational
datasets:
  - HuggingFaceFW/fineweb-edu
---

# ZeroToGPT-134M

A 134M-parameter decoder-only transformer, written from scratch in PyTorch and
pretrained on FineWeb-Edu on a single free-tier notebook GPU.

This is a **learning-scale research artifact, not a useful assistant.** It has
no instruction tuning, no alignment, and it saw roughly 1.2B tokens, which is
about 1% of what a modern small model gets. It writes locally fluent English
and is frequently wrong about facts. It is published because the training code,
the run logs, and a full audit of its own flaws are published with it.

Code, audit, and reproduction instructions:
https://github.com/umer-ateeq/GPT-From-Scratch

## Architecture

| Component | Choice |
|---|---|
| Type | decoder-only transformer, GPT-2 style |
| Parameters | 134,077,440 trainable (+524,288 causal-mask buffers) |
| Layers | 8 |
| Attention heads | 12 (head dim 64) |
| Model width | 768 |
| FFN | 4x expansion, ReLU |
| Normalization | custom LayerNorm with learned scale/shift, pre-norm |
| Positional encoding | learned embeddings |
| Vocabulary | 50257, GPT-2 BPE via `tiktoken` |
| Output head | untied from the input embedding |
| Context length | 256 configured, **128 actually trained** (see below) |

Every row above except the head count was recovered from the checkpoint's
tensor shapes by `audit_checkpoint.py`, not copied from a config file. The head
count is not recoverable from shapes, since every projection is d_model x d_model;
12 is the run-time choice the notebook records.

## Read this before evaluating it: the context is 128, not 256

The config says 256. The model was trained at 128 because a notebook cell
rebound the globals the batch sampler read. The positional embedding table
proves it: rows 0-127 have mean norm 0.2175, rows 128-255 have mean norm
0.002346, a 93x cliff exactly at the boundary, because untrained rows received
no gradient and were decayed toward zero by AdamW weight decay.

**Evaluate and generate at context 128.** At 256 you are scoring the model on
positional rows it has never seen, and perplexity roughly doubles as a result.
The full story, with four bugs and the proof for each, is in
[AUDIT.md](AUDIT.md).

## Training

| | |
|---|---|
| Corpus | FineWeb-Edu, `CC-MAIN-2024-10`, tokenized to an 8B-token uint16 memmap |
| Tokens seen | **~1.0-1.2B**, agreed by two independent methods (the weight-decay clock and the notebook's loop counts). See AUDIT.md |
| Batch | 32 rows x 128 tokens = 4,096 tokens per optimizer step. Gradient accumulation was configured but never ran (AUDIT.md bug 5) |
| Optimizer | AdamW, lr 4e-4 flat, weight decay 0.1 |
| Precision | mixed precision (fp16 autocast) |
| Hardware | a single free-tier notebook GPU (16 GB) |

The learning rate was **flat**, not cosine-decayed. The notebook built a warmup
plus cosine schedule but bound it to an optimizer it then discarded, so the
schedule never reached the weights. This is Bug 2 in the audit. The
`train.py` in the repo implements warmup and cosine decay correctly and logs the
per-step learning rate, but this checkpoint predates it.

## Evaluation

Perplexity via the standard strided sliding window, overlap tokens masked out of
the loss so nothing is double counted. Reproduce any row with the command next
to it.

| Benchmark | Context | Perplexity | Command |
|---|---|---|---|
| FineWeb-Edu held out, `CC-MAIN-2024-18` | 128 | **38.89** | `python evaluate.py --ckpt weights8b_300epoch.pth --data-bin val.bin --context 128` |
| FineWeb-Edu held out, `CC-MAIN-2024-18` | 256 | 74.70 | same, `--context 256`. Outside the trained regime |
| TinyStories validation | 128 | 35.41 | `python evaluate.py --ckpt ... --data-bin tinystories_val.bin --context 128` |
| WikiText-2 raw test, full | 128 | **184.96** | `python evaluate.py --ckpt weights8b_300epoch.pth --mode wikitext --max-length 128` |
| WikiText-2, GPT-2-small on the identical harness | 128 | 59.69 | `python evaluate.py --model gpt2 --mode wikitext --max-length 128` |

The held-out set is a different Common Crawl snapshot than the training data:
distribution-matched but fully disjoint.

**This model is 3.10x worse than GPT-2-small on WikiText-2**, measured by pushing
GPT-2-small through the identical scoring function on the same 285,396 tokens.
That is the expected outcome: GPT-2-small saw roughly 8B tokens of WebText
against this model's 1.23B of filtered educational web text, and WikiText-2 is
encyclopedic prose far from FineWeb-Edu. The same comparison validates the
harness, since GPT-2-small's published ~29.4 at context 1024 degrading to 59.69
at context 128 is the right direction and magnitude. Full detail in
[RESULTS.md](RESULTS.md).

## Usage

```python
import tiktoken, torch
from model import GPTModel   # from the GitHub repo, also uploaded here

CONFIG = {"vocab_size": 50257, "context_length": 256, "emb_dim": 768,
          "n_heads": 12, "n_layers": 8, "drop_rate": 0.1, "qkv_bias": False}

model = GPTModel(CONFIG)
model.load_state_dict(torch.load("weights8b_300epoch.pth", map_location="cpu",
                                 weights_only=True))
model.eval()

enc = tiktoken.get_encoding("gpt2")
ids = torch.tensor([enc.encode_ordinary("Photosynthesis is the process by which")])

with torch.no_grad():
    for _ in range(50):
        logits = model(ids[:, -128:])[:, -1, :] / 0.8   # keep within trained context
        kth = logits.topk(50).values[:, -1:]
        logits = logits.masked_fill(logits < kth, -float("inf"))
        ids = torch.cat([ids, torch.multinomial(torch.softmax(logits, -1), 1)], dim=1)

print(enc.decode(ids[0].tolist()))
```

Or use the repo's script, which handles the context window for you:

```bash
python generate.py --ckpt weights8b_300epoch.pth --prompt "Photosynthesis is"
```

## Limitations and intended use

- **Not an assistant.** No instruction tuning, no RLHF, no safety tuning. It
  completes text and nothing else.
- **Undertrained.** ~1.23B tokens at 134M parameters is far below
  compute-optimal. Expect confident factual errors.
- **128-token effective context.** Longer prompts silently degrade.
- **Inherits FineWeb-Edu's biases.** The corpus is filtered educational web
  text, English only, and carries whatever biases the filtering left behind.
- **Intended use:** studying transformer pretraining, reproducing the training
  pipeline, and as a baseline for the architecture ablations described in the

## Attribution

Model code started from Sebastian Raschka's
[Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch).
Memory-mapped data loading and parts of the training loop follow Andrej
Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). Training data is
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).

## Citation

```bibtex
@misc{siddiqui2026zerotogpt,
  author = {Umer Ateeq Siddiqui},
  title  = {ZeroToGPT-134M: pretraining a GPT from scratch on a free-tier GPU,
            with a full audit of the run},
  year   = {2026},
  url    = {https://github.com/umer-ateeq/GPT-From-Scratch}
}
```
