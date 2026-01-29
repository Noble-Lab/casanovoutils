import pathlib
import shutil
from os import PathLike
from typing import Any, Optional

import fire
import pyteomics.mgf
import tqdm
import yaml

PyteomicsSpectrum = list[dict[str, Any]]


def get_pep_dict_mgf(mgf_files: PathLike) -> dict[str, PyteomicsSpectrum]:
    """
    Read a MGF file and group spectra by peptide sequence.

    Parameters
    ----------
    mgf_files : PathLike
        Path to an MGF file.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        A mapping from peptide sequence (str) to a list of corresponding
        pyteomics MGF spectrum dictionaries.
    """
    mgf_iter = pyteomics.mgf.read(mgf_files, use_index=False)
    mgf_iter = tqdm.tqdm(mgf_iter, desc=f"Reading mgf file", unit="psm")

    out = {}
    for curr in mgf_iter:
        seq = curr["params"]["seq"]
        if curr["params"]["seq"] not in out:
            out[seq] = []
        out[seq].append(curr)

    return out


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


def main() -> None:
    """CLI Entry"""
    fire.Fire(
        {
            "dump-residues": dump_residues,
        }
    )
