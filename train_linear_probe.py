"""Compatibility entry point for the WorldLab board experiment."""

import sys

from worldlab.experiments.board import probe as _implementation

if __name__ == "__main__":
    _implementation.cli()
else:
    sys.modules[__name__] = _implementation
