import pathlib
import shutil
from os import PathLike
from typing import Any, Iterable, Optional

import numpy as np
import tqdm
import fire
import pyteomics.mgf
import pyteomics.mztab
import casanovo.denovo.evaluate
import pandas as pd
import yaml


def get_pep_dict_mgf(mgf_files: Iterable[PathLike]) -> dict[str, list[dict[str, Any]]]:
    """
    Read one or more MGF files and group spectra by peptide sequence.

    Parameters
    ----------
    mgf_files : Iterable[PathLike]
        Iterable of paths to MGF files. Each file is parsed into PSMs.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        A mapping from peptide sequence (str) to a list of corresponding
        pyteomics MGF spectrum dictionaries.
    """
    mgf_iter = tqdm.tqdm(
        pyteomics.mgf.from_iterable(mgf_files), desc=f"Reading mgf files", unit="psm"
    )

    out = {}
    for curr in mgf_iter:
        seq = curr["params"]["seq"]
        if curr["params"]["seq"] not in out:
            out[seq] = []
        out[seq].append(curr)

    return out


def prec_cov(
    scores: np.ndarray, is_correct: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the precision-coverage curve and its area-under-curve (AUPC) for a
    set of scored predictions.

    Parameters
    ----------
    scores : np.ndarray
        1D array of prediction scores, where higher values indicate greater
        confidence.
    is_correct : np.ndarray
        1D boolean or binary array indicating whether each prediction is correct
        (1/True) or incorrect (0/False). Must be the same length as ``scores``.

    Returns
    -------
    precision : np.ndarray
        Precision values at each coverage step after sorting by score. Length
        ``N``.
    coverage : np.ndarray
        Coverage values, normalized to the range [0, 1], where ``coverage[i]``
        is the fraction of samples included up to index ``i`` in the ranked
        list.
    aupc : float
        Area under the precision-coverage curve.
    """
    sort_idx = np.argsort(scores)[::-1]
    is_correct = is_correct[sort_idx]
    total_coverage = np.arange(1, len(is_correct) + 1)
    total_precision = np.cumsum(is_correct)

    precision = total_precision / total_coverage
    coverage = total_coverage / total_coverage[-1]
    aupc = np.trapz(precision, coverage)
    return precision, coverage, aupc


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


def get_aa_matches(
    mztab_path: PathLike | pd.DataFrame,
    pred_col: str = "sequence",
    ground_truth_col: Optional[str] = None,
    ground_truth_mgf: Optional[PathLike] = None,
    residues_path: Optional[PathLike] = None,
) -> tuple[tuple[np.ndarray, bool], np.ndarray, np.ndarray]:
    """
    Extract predicted peptide sequences, ground-truth sequences, and amino-acid
    match information for a batch of PSMs.

    This function loads peptide-spectrum matches (PSMs) from either an mzTab
    file or a pre-existing pandas DataFrame, retrieves predicted peptide
    sequences, obtains ground-truth sequences from an mzTab column or an MGF
    file, aligns the predictions to the correct spectrum indices when needed,
    and computes amino-acid-level correctness using Casanovo's
    ``aa_match_batch`` evaluator.

    Parameters
    ----------
    mztab_path : PathLike or pandas.DataFrame
        Path to an mzTab file containing a ``spectrum_match_table`` section, or
        an already-loaded DataFrame containing equivalent columns.
    pred_col : str, optional
        Name of the column containing predicted peptide sequences.
        Defaults to ``"sequence"``.
    ground_truth_col : str, optional
        Name of a column in the mzTab file containing ground-truth peptide
        sequences. Mutually exclusive with ``ground_truth_mgf``.
    ground_truth_mgf : PathLike, optional
        Path to an MGF file containing ground-truth peptide sequences encoded
        using ``SEQ=<peptide>`` lines. If provided, the ground truth is read
        from the MGF file and aligned to mzTab PSM rows via ``spectra_ref``.
    residues_path : PathLike, optional
        Path to a residue-mass YAML file used by ``get_residues()`` to
        construct an amino-acid mass dictionary for Casanovo's evaluator.

    Returns
    -------
    aa_matches : tuple[np.ndarray, bool]
        Per-PSM amino-acid match results produced by
        ``casanovo.denovo.evaluate.aa_match_batch``.
        For each PSM:
            - ``aa_matches[i][0]`` is a 1D array of per-AA correctness
              values (e.g., 1 = correct, 0 = incorrect).
            - ``aa_matches[i][1]`` is a Boolean peptide-level correctness flag.
    ground_truth : np.ndarray
        Array of ground-truth peptide sequences aligned to the PSMs. If the
        ground truth originates from an MGF file, this array is ordered by
        spectrum index and matches the mzTab rows via ``spectra_ref``.
    pred : np.ndarray
        Array of predicted peptide sequences aligned to the PSMs. If an MGF
        ground-truth file is used, predictions are re-ordered so that
        ``pred[i]`` corresponds to ``ground_truth[i]`` for the same spectrum.

    Raises
    ------
    ValueError
        If neither ``ground_truth_col`` nor ``ground_truth_mgf`` is provided.
    """
    if not isinstance(mztab_path, pd.DataFrame):
        psm_df = pyteomics.mztab.MzTab(mztab_path).spectrum_match_table
    else:
        psm_df = mztab_path

    pred = psm_df[pred_col].to_numpy()
    if ground_truth_col is None and ground_truth_mgf is None:
        raise ValueError(
            "Either a ground truth mztab column or ground truth mgf file must be"
            " provided"
        )
    elif ground_truth_col is not None:
        ground_truth = psm_df[ground_truth_col].to_numpy()
    else:
        ground_truth = []
        with open(ground_truth_mgf) as f:
            for line in tqdm.tqdm(f, desc=f"Reading mgf file: {ground_truth_mgf}", unit="lines"):
                if line.startswith("SEQ="):
                    ground_truth.append(line.removeprefix("SEQ=").strip())

        spectra_idx = (
            psm_df["spectra_ref"].str[len("ms_run[1]:index=") :].apply(int).to_numpy()
        )

        ground_truth = np.array(ground_truth)
        pred_aligned = np.full_like(ground_truth, None, dtype=object)
        pred_aligned[spectra_idx] = pred
        pred = pred_aligned

    aa_dict = get_residues(residues_path)
    aa_matches, _, _ = casanovo.denovo.evaluate.aa_match_batch(
        tqdm.tqdm(pred, desc="Checking peptides", unit="peptide"),
        ground_truth,
        aa_dict,
    )

    return aa_matches, ground_truth, pred


def main() -> None:
    """CLI Entry"""
    fire.Fire(
        {
            "dump-residues": dump_residues,
        }
    )
