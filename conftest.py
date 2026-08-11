"""Make `pretrain/` and `interp/` importable during tests.

pytest imports this automatically before collecting, so the test files can say
`from model import GPTModel` without any path juggling of their own.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

for _name in ("pretrain", "interp"):
    _path = os.path.join(_ROOT, _name)
    if _path not in sys.path:
        sys.path.insert(0, _path)
