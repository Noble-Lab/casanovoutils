"""
Tests for casanovoutils.main.

Verifies that main() discovers submodules with COMMANDS, uses CLI_NAME when
present, skips modules without COMMANDS, and delegates to fire.Fire.
"""

import importlib
import types
from unittest.mock import MagicMock, call, patch

import pytest

import casanovoutils.main as main_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_module(name: str, commands=None, cli_name=None) -> types.ModuleType:
    """Return a minimal fake module."""
    mod = types.ModuleType(name)
    if commands is not None:
        mod.COMMANDS = commands
    if cli_name is not None:
        mod.CLI_NAME = cli_name
    return mod


def _make_module_info(name: str):
    """Return a pkgutil.ModuleInfo-like object."""
    mi = MagicMock()
    mi.name = name
    return mi


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def _patch_discovery(module_map: dict):
    """
    Context-manager that patches pkgutil.iter_modules and importlib.import_module
    so that main() sees exactly the modules in *module_map*
    (``{module_name: module_object}``), plus a fake package object.
    """
    fake_package = types.ModuleType("casanovoutils")
    fake_package.__path__ = []

    module_infos = [_make_module_info(name) for name in module_map]

    _real_import = importlib.import_module

    def fake_import(name):
        # top-level package import (no dot)
        if "." not in name:
            return fake_package
        _, mod_name = name.rsplit(".", 1)
        if mod_name not in module_map:
            return _real_import(name)
        return module_map[mod_name]

    return (
        patch("casanovoutils.main.pkgutil.iter_modules", return_value=module_infos),
        patch("casanovoutils.main.importlib.import_module", side_effect=fake_import),
    )


def test_commands_from_modules_with_commands():
    cmd_a = {"run": lambda: None}
    cmd_b = {"show": lambda: None}
    modules = {
        "mod_a": _make_module("mod_a", commands=cmd_a),
        "mod_b": _make_module("mod_b", commands=cmd_b),
    }
    iter_patch, import_patch = _patch_discovery(modules)
    with iter_patch, import_patch, patch("casanovoutils.main.fire.Fire") as mock_fire:
        main_mod.main()

    fired = mock_fire.call_args[0][0]
    assert fired == {"mod_a": cmd_a, "mod_b": cmd_b}


def test_modules_without_commands_are_skipped():
    modules = {
        "has_cmd": _make_module("has_cmd", commands={"go": lambda: None}),
        "no_cmd": _make_module("no_cmd"),  # no COMMANDS attribute
    }
    iter_patch, import_patch = _patch_discovery(modules)
    with iter_patch, import_patch, patch("casanovoutils.main.fire.Fire") as mock_fire:
        main_mod.main()

    fired = mock_fire.call_args[0][0]
    assert "no_cmd" not in fired
    assert "has_cmd" in fired


def test_module_name_used_as_key_when_cli_name_absent():
    cmd = {"run": lambda: None}
    modules = {
        "somemodule": _make_module("somemodule", commands=cmd),
    }
    iter_patch, import_patch = _patch_discovery(modules)
    with iter_patch, import_patch, patch("casanovoutils.main.fire.Fire") as mock_fire:
        main_mod.main()

    fired = mock_fire.call_args[0][0]
    assert "somemodule" in fired


def test_empty_package_fires_empty_dict():
    iter_patch, import_patch = _patch_discovery({})
    with iter_patch, import_patch, patch("casanovoutils.main.fire.Fire") as mock_fire:
        main_mod.main()

    assert mock_fire.call_args[0][0] == {}


def test_fire_called_exactly_once():
    modules = {
        "mod": _make_module("mod", commands={"x": lambda: None}),
    }
    iter_patch, import_patch = _patch_discovery(modules)
    with iter_patch, import_patch, patch("casanovoutils.main.fire.Fire") as mock_fire:
        main_mod.main()

    assert mock_fire.call_count == 1


def test_commands_dict_passed_directly_to_fire():
    cmd = {"go": lambda: None}
    modules = {"m": _make_module("m", commands=cmd)}
    iter_patch, import_patch = _patch_discovery(modules)
    with iter_patch, import_patch, patch("casanovoutils.main.fire.Fire") as mock_fire:
        main_mod.main()

    assert mock_fire.call_args == call({"m": cmd})


# ---------------------------------------------------------------------------
# Integration: real package
# ---------------------------------------------------------------------------


def test_real_package_includes_known_modules():
    """Smoke test: main() with the real package finds at least the known submodules."""
    with patch("casanovoutils.main.fire.Fire") as mock_fire:
        main_mod.main()

    fired = mock_fire.call_args[0][0]
    for expected in ("mgfutils", "denovoutils", "residues"):
        assert expected in fired, f"Expected key {expected!r} in commands dict"


def test_real_package_commands_are_dicts_or_callables():
    with patch("casanovoutils.main.fire.Fire") as mock_fire:
        main_mod.main()

    fired = mock_fire.call_args[0][0]
    for key, value in fired.items():
        assert isinstance(value, dict) or callable(
            value
        ), f"COMMANDS for {key!r} must be a dict or callable, got {type(value)}"


# ---------------------------------------------------------------------------
# strict_booleans
# ---------------------------------------------------------------------------


def _probe(aa_level: bool = False, name: str = "x"):
    """Stand-in command with one boolean and one string parameter."""
    return aa_level, name


@pytest.mark.parametrize(
    "value, expected",
    [
        ("false", False),
        ("False", False),
        ("true", True),
        ("True", True),
        (False, False),
        (True, True),
    ],
)
def test_strict_booleans_parses_bool_parameters(value, expected):
    """fire leaves --flag=false a truthy string; it must reach the command as a bool."""
    wrapped = main_mod.strict_booleans(_probe)
    assert wrapped(aa_level=value) == (expected, "x")


def test_strict_booleans_leaves_non_bool_parameters_alone():
    wrapped = main_mod.strict_booleans(_probe)
    assert wrapped(name="false") == (False, "false")


def test_strict_booleans_rejects_non_boolean_strings():
    wrapped = main_mod.strict_booleans(_probe)
    with pytest.raises(ValueError, match="expects true or false"):
        wrapped(aa_level="maybe")


def test_strict_booleans_recurses_into_nested_commands():
    wrapped = main_mod.strict_booleans({"group": {"probe": _probe}})
    assert wrapped["group"]["probe"](aa_level="false") == (False, "x")


def test_main_wraps_commands_before_firing():
    """The real get_pc_df must reach fire wrapped, since it takes --aa_level."""
    from casanovoutils import preccov

    with patch("casanovoutils.main.fire.Fire") as mock_fire:
        main_mod.main()

    get_pc_df = mock_fire.call_args[0][0]["preccov"]["get_pc_df"]
    assert get_pc_df is not preccov.get_prec_cov_df
    assert get_pc_df.__wrapped__ is preccov.get_prec_cov_df
