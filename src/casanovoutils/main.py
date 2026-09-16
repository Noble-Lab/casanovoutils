"""
Top-level CLI entry point for casanovoutils.

Builds a nested command dict from each submodule's COMMANDS constant and
exposes them as a single ``casanovoutils`` CLI via ``fire``.

Submodules are auto-detected: any module with a ``COMMANDS`` constant is
included. The CLI key is taken from a ``CLI_NAME`` constant if present,
otherwise the module name with a trailing ``utils`` suffix stripped.
"""

import importlib
import pkgutil

import fire

from .types import Commands


def main() -> None:
    """
    Entry point for the ``casanovoutils`` CLI.

    Scans all submodules of this package for a ``COMMANDS`` constant and
    builds a nested command dict to pass to ``fire.Fire``.  The top-level
    key for each submodule is taken from its ``CLI_NAME`` constant if one
    exists, otherwise the bare module name is used.

    To expose a new group of commands, add a ``COMMANDS`` dict to any
    submodule.
    """
    package = importlib.import_module(__package__)
    commands: Commands = {}

    for module_info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{__package__}.{module_info.name}")
        if not hasattr(module, "COMMANDS"):
            continue

        commands[module_info.name] = module.COMMANDS

    fire.Fire(commands)


if __name__ == "__main__":
    main()
