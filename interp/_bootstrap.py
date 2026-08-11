"""Put `pretrain/` on sys.path so these scripts can import the model they analyse.

The interpretability scripts need `model.py` and `config.py`, which live in
`pretrain/`. Importing this module first makes `python interp/<script>.py` work
from anywhere without installing the repo as a package.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

for _path in (_HERE, os.path.join(_ROOT, "pretrain")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
