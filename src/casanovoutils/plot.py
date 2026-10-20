"""Plot annotated spectra and mirror plots from Casanovo mzTab results."""

import logging

import matplotlib.pyplot as plt
import polars as pl
import spectrum_utils.plot as sup
import spectrum_utils.spectrum as sus

from .constants import Constants
from .denovoutils import DfPath, get_mztab_df
from .types import Commands
from .visualize_errors import _load_mgf_peaks, _make_spectrum


def _spectrum_at_index(peak_file: str, index: int) -> dict:
    """
    Return the pyteomics spectrum dict at a zero-based MGF position.

    The spectrum is read by position (``use_index=False``) so the lookup
    works on MGF files without ``TITLE=`` lines, such as those derived
    from MassIVE-KB.

    Parameters
    ----------
    peak_file : str
        Path to the MGF peak file.
    index : int
        Zero-based index of the spectrum, matching the ``index=N`` value
        in the mzTab ``spectra_ref`` column.
    """
    peaks = _load_mgf_peaks(peak_file, indices={index})
    if index not in peaks:
        raise ValueError(f"No spectrum at index {index} in {peak_file}")
    return peaks[index]


def _peptide_at_index(mztab_df: pl.DataFrame, index: int) -> str:
    """
    Return the ProForma peptide for the spectrum at a zero-based index.

    Parameters
    ----------
    mztab_df : pl.DataFrame
        The mzTab spectrum match table, as returned by ``get_mztab_df``.
    index : int
        Zero-based spectrum index to look up in ``spectra_ref``.

    Returns
    -------
    str
        The ProForma peptide string for the matching PSM.
    """
    column = (
        Constants.proforma_column
        if Constants.proforma_column in mztab_df.columns
        else "mztab_sequence"
    )
    matches = mztab_df.filter(
        pl.col("mztab_spectra_ref").str.ends_with(f"index={index}")
    )
    if matches.is_empty():
        raise ValueError(f"No PSM found for spectrum index {index}")
    if matches.height > 1:
        logging.warning(
            "Multiple PSMs found for spectrum index %d; using the first.", index
        )
    return matches[column][0]


def _annotated_spectrum(
    spectrum_dict: dict,
    peptide: str,
    identifier: str,
    fragment_tol: float,
    fragment_tol_mode: str,
    ion_types: str,
    max_charge: int | None,
    max_isotope: int,
    remove_precursor_tol: float | None,
    remove_precursor_tol_mode: str,
) -> sus.MsmsSpectrum:
    """Build an annotated spectrum from a pyteomics spectrum dict."""
    spectrum = _make_spectrum(spectrum_dict, identifier)
    if remove_precursor_tol is not None:
        spectrum.remove_precursor_peak(remove_precursor_tol, remove_precursor_tol_mode)
    spectrum.annotate_proforma(
        peptide,
        fragment_tol,
        fragment_tol_mode,
        ion_types=ion_types,
        max_isotope=max_isotope,
        max_ion_charge=max_charge,
    )
    return spectrum


def spectrum(
    mztab: DfPath,
    peak_file: str,
    index: int,
    out: str = "spectrum.png",
    fragment_tol: float = 0.5,
    fragment_tol_mode: str = "Da",
    ion_types: str = "by",
    max_charge: int | None = None,
    max_isotope: int = 0,
    title: str | None = None,
    remove_precursor_tol: float | None = None,
    remove_precursor_tol_mode: str = "Da",
) -> None:
    """
    Plot a single annotated spectrum from a Casanovo mzTab result.

    Parameters
    ----------
    mztab : DfPath
        Path to the Casanovo mzTab file.
    peak_file : str
        Path to the MGF peak file the spectra were sequenced from.
    index : int
        Zero-based index of the spectrum to plot (the ``index=N`` value in
        the mzTab ``spectra_ref`` column).
    out : str
        Output image path.
    fragment_tol : float
        Fragment mass tolerance for annotation.
    fragment_tol_mode : str
        Fragment mass tolerance mode, ``"Da"`` or ``"ppm"``.
    ion_types : str
        Fragment ion types to annotate, e.g. ``"by"``.
    max_charge : int or None
        Maximum fragment ion charge to annotate. ``None`` uses the
        spectrum_utils default (the precursor charge).
    max_isotope : int
        Maximum isotope number to annotate for each fragment ion.
    title : str or None
        Optional title for the plot.
    remove_precursor_tol : float or None
        If set, remove the precursor peak within this tolerance (in
        ``remove_precursor_tol_mode`` units) before annotating. ``None``
        keeps the precursor peak.
    remove_precursor_tol_mode : str
        Tolerance mode for precursor peak removal, ``"Da"`` or ``"ppm"``.
    """
    peptide = _peptide_at_index(get_mztab_df(mztab), index)
    spec = _annotated_spectrum(
        _spectrum_at_index(peak_file, index),
        peptide,
        str(index),
        fragment_tol,
        fragment_tol_mode,
        ion_types,
        max_charge,
        max_isotope,
        remove_precursor_tol,
        remove_precursor_tol_mode,
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sup.spectrum(spec, ax=ax)
    if title is not None:
        ax.set_title(title)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", out)


def mirror(
    mztab: DfPath,
    peak_file: str,
    index: int,
    ground_truth: str | None = None,
    out: str = "mirror.png",
    fragment_tol: float = 0.5,
    fragment_tol_mode: str = "Da",
    ion_types: str = "by",
    max_charge: int | None = None,
    max_isotope: int = 0,
    title: str | None = None,
    remove_precursor_tol: float | None = None,
    remove_precursor_tol_mode: str = "Da",
) -> None:
    """
    Plot a predicted peptide against a ground truth as a mirror plot.

    The predicted peptide (from ``mztab``) is annotated on the top panel
    and the ground truth on the bottom, using the same peaks from
    ``peak_file``.

    Parameters
    ----------
    mztab : DfPath
        Path to the Casanovo mzTab file with the predicted peptides.
    peak_file : str
        Path to the MGF peak file.
    index : int
        Zero-based index of the spectrum to plot.
    ground_truth : str or None
        Ground truth peptide for the bottom panel, as a ProForma string.
        When ``None``, it is read from the spectrum's ``SEQ=`` field in
        the MGF.
    out : str
        Output image path.
    fragment_tol : float
        Fragment mass tolerance for annotation.
    fragment_tol_mode : str
        Fragment mass tolerance mode, ``"Da"`` or ``"ppm"``.
    ion_types : str
        Fragment ion types to annotate, e.g. ``"by"``.
    max_charge : int or None
        Maximum fragment ion charge to annotate. ``None`` uses the
        spectrum_utils default (the precursor charge).
    max_isotope : int
        Maximum isotope number to annotate for each fragment ion.
    title : str or None
        Optional title for the plot. A note that the top spectrum is the
        prediction and the bottom is the ground truth is always appended.
    remove_precursor_tol : float or None
        If set, remove the precursor peak within this tolerance (in
        ``remove_precursor_tol_mode`` units) from both panels before
        annotating. ``None`` keeps the precursor peak.
    remove_precursor_tol_mode : str
        Tolerance mode for precursor peak removal, ``"Da"`` or ``"ppm"``.
    """
    predicted = _peptide_at_index(get_mztab_df(mztab), index)
    spectrum_dict = _spectrum_at_index(peak_file, index)

    if ground_truth is None:
        seq = spectrum_dict["params"].get("seq")
        if not seq:
            raise ValueError(
                f"No SEQ= field for spectrum index {index}; pass "
                "ground_truth explicitly."
            )
        ground_truth = str(seq)

    top = _annotated_spectrum(
        spectrum_dict,
        predicted,
        f"{index}-predicted",
        fragment_tol,
        fragment_tol_mode,
        ion_types,
        max_charge,
        max_isotope,
        remove_precursor_tol,
        remove_precursor_tol_mode,
    )
    bottom = _annotated_spectrum(
        spectrum_dict,
        ground_truth,
        f"{index}-ground-truth",
        fragment_tol,
        fragment_tol_mode,
        ion_types,
        max_charge,
        max_isotope,
        remove_precursor_tol,
        remove_precursor_tol_mode,
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sup.mirror(top, bottom, ax=ax)
    gt_label = "Top: predicted, bottom: ground truth"
    ax.set_title(f"{title}\n{gt_label}" if title else gt_label)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", out)


COMMANDS: Commands = {"spectrum": spectrum, "mirror": mirror}
