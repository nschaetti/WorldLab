"""Compatibility entry point for the WorldLab board experiment."""

import sys

from worldlab.experiments.board import autoencoder as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
