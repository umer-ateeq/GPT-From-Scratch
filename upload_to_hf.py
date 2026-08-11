"""Publish the checkpoint and model card to the Hugging Face Hub.

The checkpoint is 538 MB. GitHub's hard per-file limit is 100 MB, and Git LFS on
a free account gives 1 GB of bandwidth per month, which a handful of clones would
exhaust. So the weights live on the Hub and this repo links to them.

Authentication: this script never takes a token as an argument, so it cannot end
up in your shell history or in a log. Log in once, out of band:

    huggingface-cli login

or set HF_TOKEN in your environment. Then:

    python upload_to_hf.py --repo-id <your-username>/zerotogpt-134m --dry-run
    python upload_to_hf.py --repo-id <your-username>/zerotogpt-134m

--dry-run prints exactly what would be uploaded and exits without touching the
network.
"""
import argparse
import os
import sys

FILES = [
    ("weights8b_300epoch.pth", "weights8b_300epoch.pth"),
    ("MODEL_CARD.md", "README.md"),   # the Hub renders README.md as the model card
    ("model.py", "model.py"),  # so the checkpoint can be loaded standalone
    ("docs/AUDIT.md", "AUDIT.md"),
    ("docs/RESULTS.md", "RESULTS.md"),
    ("docs/INDUCTION_HEADS.md", "INDUCTION_HEADS.md"),
    ("config.py", "config.py"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True,
                   help="target Hub repo, e.g. umer-ateeq/zerotogpt-134m")
    p.add_argument("--ckpt", default="weights8b_300epoch.pth",
                   help="path to the checkpoint; it does not have to sit in this folder")
    p.add_argument("--private", action="store_true", help="create the repo private")
    p.add_argument("--dry-run", action="store_true",
                   help="print the upload plan and exit without connecting")
    return p.parse_args()


def main():
    args = parse_args()

    files = [(args.ckpt, os.path.basename(args.ckpt))] + FILES[1:]
    missing = [src for src, _ in files if not os.path.exists(src)]
    plan = [(src, dst) for src, dst in files if os.path.exists(src)]

    print(f"target repo : {args.repo_id} ({'private' if args.private else 'public'})")
    print("upload plan :")
    for src, dst in plan:
        size = os.path.getsize(src)
        arrow = f"{src} -> {dst}" if src != dst else src
        print(f"  {arrow:52s} {size / 1e6:8.1f} MB")
    if missing:
        print("missing (skipped):")
        for src in missing:
            print(f"  {src}")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return

    if not (os.environ.get("HF_TOKEN") or os.path.exists(
            os.path.expanduser("~/.cache/huggingface/token"))):
        sys.exit("No Hugging Face credentials found. Run `huggingface-cli login` "
                 "or set HF_TOKEN, then re-run. Do not pass a token on the "
                 "command line.")

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model",
                    private=args.private, exist_ok=True)
    for src, dst in plan:
        print(f"uploading {src} ...", flush=True)
        api.upload_file(path_or_fileobj=src, path_in_repo=dst,
                        repo_id=args.repo_id, repo_type="model")
    print(f"\ndone: https://huggingface.co/{args.repo_id}")
    print("Now update the checkpoint link at the top of README.md to point here.")


if __name__ == "__main__":
    main()
