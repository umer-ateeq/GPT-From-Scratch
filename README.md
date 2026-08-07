# GPT Pretraining From Scratch

[![tests](https://github.com/umer-ateeq/gpt-pretraining-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/umer-ateeq/gpt-pretraining-from-scratch/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)

A 134-million-parameter GPT, written from scratch in PyTorch and pretrained on
educational web text using a single free-tier notebook GPU. No transformer
library: attention, the causal mask, layer normalization and the training loop
are all implemented directly.

Then I audited my own checkpoint and found four bugs in the run that produced it,
one of which had made my reported training scale four times too large. That audit
is the most useful thing in this repository.

**Every number below was produced by a script in this repo, and the command that
produces it sits next to it. Where something has not been measured, this repo
says "not measured" instead of estimating.**

---

## Results

| | |
|---|---|
| Parameters | **134,077,440** trainable |
| Architecture | 8 layers, 12 heads, 768 wide, GPT-2 BPE (50257) |
| Corpus built | **8B tokens**, FineWeb-Edu, streamed and tokenized to a uint16 memmap |
| Tokens consumed | **1.23B**, about 46% of the Chinchilla-optimal budget for this size |
| Held-out perplexity | **38.89** at context 128, on a disjoint Common Crawl snapshot |
| WikiText-2 perplexity | **184.96**, against **59.69** for GPT-2-small on the identical harness |
| Hardware | one free-tier 16 GB notebook GPU |
| Throughput | **not measured** (see [docs/RESULTS.md](docs/RESULTS.md)) |

This model **loses to GPT-2-small by 3.1x** on WikiText-2. That is the expected
result and it is stated up front: GPT-2-small saw roughly 8B tokens against this
model's 1.23B. The comparison is here because a perplexity number without a
reference measured the same way is not a claim anyone can check.

Full detail, including the two protocols and why in-domain and out-of-domain
differ by 4.8x: **[docs/RESULTS.md](docs/RESULTS.md)**.

## Verify any of it in three commands

```bash
pip install -r requirements.txt

# 1. Recover the architecture and the true training context from the weights alone
python audit_checkpoint.py --ckpt weights8b_300epoch.pth

# 2. WikiText-2, then the same benchmark on GPT-2-small through the identical code path
python evaluate.py --ckpt weights8b_300epoch.pth --mode wikitext --max-length 128
python evaluate.py --model gpt2 --mode wikitext --max-length 128

# 3. The tests, each guarding a bug the audit found. No checkpoint or dataset needed.
python -m pytest tests/ -v
```

Command 3 needs nothing but this repo. Commands 1 and 2 need the checkpoint,
which is 538 MB and therefore published separately rather than committed, since
GitHub caps files at 100 MB.

## The audit

The config said context 256 and 4.9B tokens. The weights say context **128** and
**1.23B**.

A notebook cell had rebound the globals that the batch sampler read, eighteen
cells after they were defined, so every batch was 32 x 128 while every printed
summary still said 64 x 256. The positional embedding table records this
permanently: a row only receives gradient when a batch is long enough to reach
it, and AdamW's weight decay shrinks whatever it touches, so untrained rows decay
toward zero.

```
       positions   mean norm
            0-31      0.2132     <- trained
           32-63      0.1886
           64-95      0.2196
          96-127      0.2487
         128-159      0.0024     <- never received gradient
         160-191      0.0024
         192-223      0.0024
         224-255      0.0023
```

A **93x cliff exactly at position 128**. The source bug and the weights agree,
independently.

Three more bugs turned up in the same audit, including a warmup-plus-cosine
schedule bound to an optimizer the training cell then replaced, so it never
touched the weights at all. All four, with the proof for each and the code
change that now prevents it: **[docs/AUDIT.md](docs/AUDIT.md)**.

## Corpus size is not training scale

Two numbers get conflated in projects like this, and doing so is how a run gets
overstated by several multiples. Both are reported separately throughout:

- **Corpus: 8B tokens.** What `tokenize_data.py` streamed and wrote to disk.
- **Consumed: 1.23B tokens.** What actually passed through the model, sampled as
  random windows, roughly 15% of the corpus.

At 9.17 tokens per parameter this run reached about **46% of the
Chinchilla-optimal budget** for a 134M model. Undertrained by roughly half, and
knowingly so, because the budget was one free notebook GPU.

## The code

```
config.py            all hyperparameters in one place, which is the single change
                     that would have prevented the worst bug in the original run
model.py             LayerNorm, FeedForward, MultiHeadAttention, TransformerBlock,
                     GPTModel. Written out, no transformer library
data.py              memory-mapped batch sampling from a uint16 token file
tokenize_data.py     stream a HuggingFace dataset into that token file
train.py             the training loop: mixed precision, gradient accumulation,
                     warmup + cosine decay, clipping, resumable, fully logged
generate.py          greedy and temperature/top-k sampling
evaluate.py          perplexity, two protocols, GPT-2-small as an in-harness baseline
audit_checkpoint.py  recover architecture and true training context from raw weights
tests/               9 tests, each mapped to a real failure mode
notebooks/           the original Colab notebook, unedited, cited as evidence by the audit
```

Start with [model.py](model.py) to read the transformer, or
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the same thing in prose with a
data-flow diagram.

## How it was trained

Three techniques, all present because of the 16 GB limit:

**Gradient accumulation.** 32 micro-batches of 32 sequences accumulate before one
optimizer step, giving the gradient quality of a 1024-sequence batch at the
memory cost of 32. This is the main reason the run fits.

**Mixed precision (fp16).** Halves activation memory and uses the GPU's tensor
cores. A gradient scaler multiplies the loss before backward so small gradients
do not underflow to zero in fp16, then unscales before the optimizer step.

**Memory-mapped data.** The corpus is a flat uint16 array on disk. `np.memmap`
pages in only the windows actually sampled, so training reads from an 8B-token
file without loading it.

Reproduce:

```bash
# 1. build a corpus (the full 8B is ~16 GB; start smaller)
python tokenize_data.py --out train.bin      --max-tokens 1e9 --dump CC-MAIN-2024-10
python tokenize_data.py --out validation.bin --max-tokens 5e6 --dump CC-MAIN-2024-18

# 2. train (logs config, seed, git commit, throughput and loss curves to runs/)
python train.py --train-bin train.bin --val-bin validation.bin \
    --train-tokens 1e9 --lr 4e-4 --batch-size 32 --run-name baseline

# 3. evaluate
python evaluate.py --ckpt runs/baseline/ckpt_best.pt --mode bin \
    --data-bin validation.bin --context 128
```

The validation set is built from a **different Common Crawl snapshot** than
training, so it is distribution-matched but genuinely disjoint.

## What the model actually produces

[SAMPLES.md](SAMPLES.md) has six unedited completions from one seeded run, with
the prompts fixed in source so they cannot be chosen after seeing the output.

> The main causes of the French Revolution were the war in France, the wars in
> France, the war in France and the French Revolution in France...

Locally fluent, factually wrong, and looping. That is what 184.96 out-of-domain
perplexity predicts, and it is shown rather than hidden.

## What this is not

- **Not an assistant.** No instruction tuning, no RLHF, no safety tuning. It
  completes text.
- **Not competitive.** It loses to GPT-2-small, a 2019 model, by 3.1x on
  WikiText-2.
- **Not fully measured.** Throughput and the LR/batch sweeps have not been run.
  [docs/RESULTS.md](docs/RESULTS.md) lists every gap.

It is a from-scratch implementation, trained end to end under a real constraint,
measured honestly against a baseline, and audited by its author.

## Documentation

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | every component, why it is there, parameter budget |
| [docs/AUDIT.md](docs/AUDIT.md) | the four bugs, the proof for each, what prevents them now |
| [docs/RESULTS.md](docs/RESULTS.md) | every measurement with its command, and what is unmeasured |
| [docs/CHANGES_FROM_NOTEBOOK.md](docs/CHANGES_FROM_NOTEBOOK.md) | every line-level difference between the notebook and this package |

## Attribution

The model implementation follows Sebastian Raschka's
[Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch).
The memory-mapped batch sampler and parts of the training loop follow Andrej
Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). Training data is
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
Evaluation follows the strided-window protocol from the GPT-2 paper.

MIT licensed, see [LICENSE](LICENSE).
