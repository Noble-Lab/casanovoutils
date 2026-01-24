from os import PathLike

import pyteomics.mztab
import pandas as pd
import numpy as np
import tqdm


def get_ground_truth(
    mztab_path: PathLike | pd.DataFrame, mgf_path: PathLike, replace_i_l: bool = False
) -> tuple[tuple[np.ndarray, bool], np.ndarray, np.ndarray]:
    """
    Align MzTab PSM predictions to MGF-provided ground-truth sequences.

    This helper reads peptide-spectrum match (PSM) predictions from an MzTab
    file (or an already-loaded PSM DataFrame), reads the ground-truth peptide
    sequence for each spectrum from an MGF file (via ``SEQ=...`` lines), and
    produces a per-spectrum table containing:

    - ``ground_truth``: ground-truth peptide sequence from the MGF (SEQ=)
    - ``predicted``: predicted peptide sequence from the MzTab PSM table
    - ``pep_score``: score used to rank PSMs (from ``search_engine_score[1]``)
    - ``pep_correct``: boolean exact-match correctness label

    The MzTab rows are aligned to the MGF spectrum order using the integer
    spectrum index parsed from the ``spectra_ref`` column, which is assumed to
    have the form ``"ms_run[1]:index=<INT>"``. Only spectra referenced by the
    MzTab receive predictions/scores; all others remain at default fill values.

    Parameters
    ----------
    mztab_path : PathLike or pandas.DataFrame
        Either a path to an MzTab file readable by ``pyteomics.mztab.MzTab``, or
        a DataFrame representing the spectrum match table with at least the
        columns:

        - ``spectra_ref``
        - ``sequence``
        - ``search_engine_score[1]``
    mgf_path : PathLike
        Path to an MGF file containing ground-truth sequences.
    replace_i_l : bool, default=False
        If True, treat isoleucine (I) and leucine (L) as equivalent by replacing
        ``"I"`` with ``"L"`` in the ground-truth sequences prior to computing
        correctness.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per spectrum (i.e., per ``SEQ=`` line in the MGF),
        containing the columns:

        - ``ground_truth`` (str)
        - ``predicted`` (str; empty string when missing)
        - ``pep_score`` (float; -1.0 when missing)
        - ``pep_correct`` (bool)
    """
    if not isinstance(mztab_path, pd.DataFrame):
        psm_df = pyteomics.mztab.MzTab(mztab_path).spectrum_match_table
    else:
        psm_df = mztab_path

    ground_truth = []
    with open(mgf_path) as f:
        for line in tqdm.tqdm(f, desc=f"Reading mgf file: {mgf_path}", unit="lines"):
            if line.startswith("SEQ="):
                ground_truth.append(line.removeprefix("SEQ=").strip())

    spectra_idx = (
        psm_df["spectra_ref"].str[len("ms_run[1]:index=") :].apply(int).to_numpy()
    )

    predictions_df = pd.DataFrame({"ground_truth": ground_truth})

    for old, new in [
        ("sequence", "predicted"),
        ("search_engine_score[1]", "pep_score"),
    ]:
        predictions_df[new] = "" if new != "pep_score" else -1.0
        predictions_df[new].iloc[spectra_idx] = psm_df[old]

    if replace_i_l:
        predictions_df["ground_truth"] = predictions_df["ground_truth"].str.replace(
            "I", "L"
        )

    predictions_df["pep_correct"] = (
        predictions_df["ground_truth"] == predictions_df["predicted"]
    )

    return predictions_df


def prec_cov(
    scores: np.ndarray,
    is_correct: np.ndarray,
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
