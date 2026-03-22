from os import PathLike

import numpy as np
import pandas as pd
import pyteomics.mztab
import tqdm

MIN_PEP_SCORE = -1.0


def get_ground_truth(
    mztab_path: PathLike | pd.DataFrame, mgf_path: list[PathLike], replace_i_l: bool = False
) -> tuple[tuple[np.ndarray, bool], np.ndarray, np.ndarray]:
    """
    Align MzTab PSM predictions to MGF-provided ground-truth sequences.

    This helper reads peptide-spectrum match (PSM) predictions from an MzTab
    file (or an already-loaded PSM DataFrame), reads the ground-truth peptide
    sequence for each spectrum from an MGF file (via ``SEQ=...`` lines), and
    returns a DataFrame indexed by MGF spectrum order with MzTab-derived fields
    inserted at the corresponding spectrum indices.

    The MzTab rows are aligned to the MGF spectrum order using the integer
    spectrum index parsed from the ``spectra_ref`` column, which is assumed to
    have the form::

        "ms_run[1]:index=<INT>"

    Only spectra referenced by the MzTab receive prediction values; all other
    rows are filled with dtype-appropriate missing values (``pd.NA`` for most
    columns, or column-specific defaults).

    Parameters
    ----------
    mztab_path : PathLike or pandas.DataFrame
        Either:

        - A path to an MzTab file readable by ``pyteomics.mztab.MzTab``, or
        - A DataFrame representing the spectrum match table.

        The PSM table must contain at least:

        - ``spectra_ref`` — used to determine spectrum index
        - ``sequence`` — predicted peptide sequence

        Any additional columns present in the PSM table are also propagated
        into the output DataFrame.
    mgf_path : PathLike
        Path to an MGF file containing ground-truth peptide sequences encoded as
        ``SEQ=<PEPTIDE>`` lines. Each such line defines one spectrum entry in
        order of appearance.
    replace_i_l : bool, default=False
        If True, treat isoleucine (I) and leucine (L) as equivalent by replacing
        ``"I"`` with ``"L"`` in the ground-truth sequences prior to computing
        correctness.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per spectrum (i.e., per ``SEQ=`` line in the
        MGF). It always contains:

        - ``ground_truth`` (string dtype)
            Ground-truth peptide sequence from the MGF.

        - All columns from the MzTab PSM table
            These are inserted at the spectrum indices specified by
            ``spectra_ref``. Rows without a corresponding PSM retain missing
            values.

        - ``pep_correct`` (boolean)
            Exact-match correctness label computed as::

                ground_truth == sequence

            after optional I/L replacement.
    """
    if not isinstance(mztab_path, pd.DataFrame):
        psm_df = pyteomics.mztab.MzTab(mztab_path).spectrum_match_table
    else:
        psm_df = mztab_path

    ground_truth = []
    if isinstance(mgf_path, str): 
        mgf_path = [mgf_path]

    for path in mgf_path:
        gt = []
        with open(path) as f:
            for line in tqdm.tqdm(f, desc=f"Reading mgf file: {path}", unit="lines"):
                if line.startswith("SEQ="):
                    gt.append(line.removeprefix("SEQ=").strip())
        ground_truth.append(gt)

    overall_spectra_idx = []
    overall_run_masks = []  
    for i in range(len(ground_truth)):
        run_mask = psm_df["spectra_ref"].str.startswith(f"ms_run[{i + 1}]")
        overall_run_masks.append(run_mask) 
        spectra_idx = (
            psm_df[run_mask]["spectra_ref"].str[len(f"ms_run[{i + 1}]:index="):].apply(int).to_numpy()
        )
        overall_spectra_idx.append(spectra_idx)

    predictions_df = pd.DataFrame({"ground_truth": [item for gt in ground_truth for item in gt]})

    total_spectra = sum(len(gt) for gt in ground_truth)  
    lengths = [len(gt) for gt in ground_truth]
    offsets = [0] + [sum(lengths[:i]) for i in range(1, len(lengths))]

    for col in psm_df.columns:
        if col == "sequence":
            predictions_df[col] = pd.Series(
                "", index=range(total_spectra), dtype="string"  
            )
        elif col == "search_engine_score[1]":
            predictions_df[col] = pd.Series(
                MIN_PEP_SCORE, index=range(total_spectra), dtype="float64"  
            )
        else:
            predictions_df[col] = None

        col_idx = predictions_df.columns.get_loc(col)
        for i, (spectra_idx, run_mask) in enumerate(zip(overall_spectra_idx, overall_run_masks)): 
            global_idx = spectra_idx + offsets[i] 
            predictions_df.iloc[global_idx, col_idx] = psm_df[run_mask][col].values

    if replace_i_l:
        predictions_df["ground_truth"] = predictions_df["ground_truth"].str.replace(
            "I", "L"
        )

    predictions_df["pep_correct"] = (
        predictions_df["ground_truth"] == predictions_df["sequence"]
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
    aupc = np.trapezoid(precision, coverage)
    return precision, coverage, aupc
