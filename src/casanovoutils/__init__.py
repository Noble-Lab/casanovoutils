import pathlib
import shutil
from os import PathLike
from typing import Any, Optional

import fire
import pyteomics.mgf
import tqdm
import yaml

PyteomicsSpectrum = list[dict[str, Any]]


def get_pep_dict_mgf(
    mgf_file: PathLike | pyteomics.mgf.MGF,
) -> dict[str, PyteomicsSpectrum]:
    """
    Read spectra from an MGF file  and group them by peptide sequence.

    This function iterates through all spectra in an MGF source and builds a
    dictionary mapping each peptide sequence to the list of spectra (PSMs)
    annotated with that sequence.

    Parameters
    ----------
    mgf_file : PathLike or pyteomics.mgf.MGFBase
        Either:
        - A filesystem path to an MGF file, which will be opened using
          ``pyteomics.mgf.read(..., use_index=False)``, or
        - An existing Pyteomics MGF reader/iterator (e.g., ``MGF`` or
          ``IndexedMGF``), in which case it is used directly.

    Returns
    -------
    dict[str, list[PyteomicsSpectrum]]
        A dictionary mapping peptide sequence strings (taken from
        ``spectrum["params"]["seq"]``) to a list of Pyteomics spectrum
        dictionaries corresponding to that sequence. Each value in the list is
        one spectrum/PSM entry as yielded by Pyteomics.
    """
    if not isinstance(mgf_file, pyteomics.mgf.MGFBase):
        mgf_iter = pyteomics.mgf.read(mgf_file, use_index=False)

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
