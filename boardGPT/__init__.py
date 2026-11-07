"""Compatibility namespace for older BoardGPT scripts and serialized models."""

import sys
from importlib import import_module, util
from importlib.abc import Loader, MetaPathFinder

from worldlab import board as _board


class _LegacyLoader(Loader):
    def __init__(self, target):
        self.target = target

    def exec_module(self, module):
        # Keep a single module object and one set of class identities.
        sys.modules[module.__name__] = import_module(self.target)

    def get_code(self, fullname):
        # runpy uses this for old "python -m boardGPT.…" entry points.
        return compile(
            f"from runpy import run_module\nrun_module({self.target!r}, run_name='__main__', alter_sys=True)",
            f"<compatibility alias for {fullname}>",
            "exec",
        )


class _LegacyFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("boardGPT."):
            return None
        suffix = fullname.removeprefix("boardGPT.")
        if suffix == "nn" or suffix.startswith("nn."):
            canonical = "worldlab.nn.board" + suffix[2:]
        elif suffix in {"models.gptboard", "models.game_autoencoder"}:
            canonical = "worldlab.nn.board." + suffix.split(".")[1]
        else:
            canonical = "worldlab.board." + suffix
        try:
            spec = util.find_spec(canonical)
        except ModuleNotFoundError as exc:
            if not exc.name or not (canonical == exc.name or canonical.startswith(exc.name + ".")):
                raise
            return None
        if spec is None:
            return None
        return util.spec_from_loader(fullname, _LegacyLoader(canonical), is_package=spec.submodule_search_locations is not None)


sys.meta_path.insert(0, _LegacyFinder())
sys.modules[__name__] = _board
