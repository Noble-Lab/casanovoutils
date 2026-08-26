"""
Top-level CLI entry point for casanovoutils.

Builds a nested command dict from each submodule's COMMANDS constant and
exposes them as a single ``casanovoutils`` CLI via ``fire``.

Submodules are auto-detected: any module with a ``COMMANDS`` constant is
included. The CLI key is taken from a ``CLI_NAME`` constant if present,
otherwise the module name with a trailing ``utils`` suffix stripped.
"""

import functools
import importlib
import inspect
import pkgutil
import typing

import fire

from .types import Commands


def parse_bool(name: str, value: typing.Any) -> typing.Any:
    """
    Parse a command line value for a ``bool`` parameter.

    ``fire`` only recognizes ``True``/``False`` literals, so ``--flag=false``
    arrives as the truthy string ``"false"``. Non-strings pass through.
    """
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"--{name} expects true or false, got {value!r}")


def strict_booleans(commands: Commands) -> Commands:
    """Wrap commands (recursively) so ``bool`` parameters are parsed by name."""
    if isinstance(commands, dict):
        return {key: strict_booleans(value) for key, value in commands.items()}

    hints = typing.get_type_hints(commands)
    names = {n for n, hint in hints.items() if hint is bool and n != "return"}
    if not names:
        return commands

    signature = inspect.signature(commands)

    @functools.wraps(commands)
    def wrapper(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        for name in names & bound.arguments.keys():
            bound.arguments[name] = parse_bool(name, bound.arguments[name])
        return commands(*bound.args, **bound.kwargs)

    return wrapper


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

    fire.Fire(strict_booleans(commands))


if __name__ == "__main__":
    main()
