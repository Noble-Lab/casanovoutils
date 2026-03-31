"""
Utilities for loading and distributing the amino acid residue mass vocabulary.
"""

import pathlib
import shutil
from os import PathLike
from typing import Optional

import fire
import yaml

from .types import Commands


def get_residues(residues_path: Optional[PathLike] = None) -> dict[str, float]:
    """
    Load a mapping of amino acid residue names to masses from a YAML file.

    If ``residues_path`` is not provided, the function loads a default
    ``residues.yaml`` file located in the same directory as this module.

    Parameters
    ----------
    residues_path : PathLike, optional
        Path to a YAML file containing residue mass information.
        If ``None`` (default), the bundled ``residues.yaml`` file is used.

    Returns
    -------
    dict[str, float]
        A dictionary mapping residue identifiers (typically one-letter or
        multi-character amino acid codes) to their corresponding masses.
    """
    if residues_path is None:
        residues_path = pathlib.Path(__file__).parent / "residues.yaml"
    with open(residues_path) as f:
        return yaml.safe_load(f)


def dump_residues(destination_path: PathLike) -> None:
    """
    Copy the default ``residues.yaml`` file included with this package to a
    specified destination.

    Parameters
    ----------
    destination_path : PathLike
        Path to copy the YAML file to. May be a directory or a file path.

    Returns
    -------
    None
    """
    residues_path = pathlib.Path(__file__).parent / "residues.yaml"
    shutil.copy(residues_path, destination_path)


COMMANDS: Commands = dump_residues


def main() -> None:
    fire.Fire(COMMANDS)


if __name__ == "__main__":
    main()
