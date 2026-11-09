"""
Module run as the process a call of a function decorated with `in_own_process` runs in.

It is started as ``python -m geommicgen._worker``, and serves one call.
"""

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen._process import serve

if __name__ == "__main__":
    serve()
