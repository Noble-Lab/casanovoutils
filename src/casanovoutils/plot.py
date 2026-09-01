"""Plot annotated spectra and mirror plots from Casanovo mzTab results."""

import sys
import warnings

import matplotlib.pyplot as plt
import polars as pl
import pyteomics.mgf
import pyteomics.mzml
import spectrum_utils.plot as sup
import spectrum_utils.spectrum as sus

from .constants import Constants
from .denovoutils import DfPath, get_mztab_df
from .types import Commands


def _read_spectrum(peak_file: str, index: int) -> sus.MsmsSpectrum:
    """
    Read the spectrum at a zero-based index from an MGF or mzML file.

    Parameters
    ----------
    peak_file : str
        Path to the MGF or mzML peak file.
    index : int
        Zero-based index of the spectrum, matching the ``index=N`` value
        in the mzTab ``spectra_ref`` column.

    Returns
    -------
    sus.MsmsSpectrum
        The spectrum with its m/z, intensity, and precursor information.
    """
    peak_file = str(peak_file)
    if peak_file.lower().endswith((".mzml", ".mzml.gz")):
        with pyteomics.mzml.MzML(peak_file) as reader:
            spectrum = reader[index]
        ion = spectrum["precursorList"]["precursor"][0]["selectedIonList"][
            "selectedIon"
        ][0]
        precursor_mz = float(ion["selected ion m/z"])
        charge = ion.get("charge state")
    else:
        with pyteomics.mgf.IndexedMGF(peak_file) as reader:
            spectrum = reader[index]
        params = spectrum["params"]
        pepmass = params["pepmass"]
        # pepmass is a (m/z, intensity) tuple, but the intensity may be
        # absent; only index into it when there are multiple values.
        precursor_mz = float(
            pepmass[0] if isinstance(pepmass, (list, tuple)) else pepmass
        )
        mgf_charge = params.get("charge")
        charge = mgf_charge[0] if mgf_charge else None

    if charge is None:
        warnings.warn(
            f"No precursor charge for spectrum index {index}; defaulting to 0."
        )
        precursor_charge = 0
    else:
        precursor_charge = int(charge)
    return sus.MsmsSpectrum(
        str(index),
        precursor_mz,
        precursor_charge,
        spectrum["m/z array"],
        spectrum["intensity array"],
    )


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
        warnings.warn(
            f"Multiple PSMs found for spectrum index {index}; using the first."
        )
    return matches[column][0]


def _annotated_spectrum(
    mztab: DfPath,
    peak_file: str,
    index: int,
    fragment_tol: float,
    fragment_tol_mode: str,
    ion_types: str,
    max_charge: int | None,
    max_isotope: int,
) -> sus.MsmsSpectrum:
    """Build the annotated spectrum for a PSM in an mzTab file."""
    peptide = _peptide_at_index(get_mztab_df(mztab), index)
    spectrum = _read_spectrum(peak_file, index)
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
) -> None:
    """
    Plot a single annotated spectrum from a Casanovo mzTab result.

    Parameters
    ----------
    mztab : DfPath
        Path to the Casanovo mzTab file.
    peak_file : str
        Path to the MGF or mzML peak file the spectra were sequenced from.
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
    """
    spec = _annotated_spectrum(
        mztab,
        peak_file,
        index,
        fragment_tol,
        fragment_tol_mode,
        ion_types,
        max_charge,
        max_isotope,
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sup.spectrum(spec, ax=ax)
    if title is not None:
        ax.set_title(title)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}", file=sys.stderr)


def mirror(
    mztab: DfPath,
    peak_file: str,
    index: int,
    ground_truth_mztab: DfPath,
    out: str = "mirror.png",
    fragment_tol: float = 0.5,
    fragment_tol_mode: str = "Da",
    ion_types: str = "by",
    max_charge: int | None = None,
    max_isotope: int = 0,
    title: str | None = None,
) -> None:
    """
    Plot a mirror plot of a predicted versus a ground truth annotation.

    The predicted peptide (from ``mztab``) is annotated on the top spectrum
    and the ground truth peptide (from ``ground_truth_mztab``) on the
    bottom, using the same peaks from ``peak_file``.

    Parameters
    ----------
    mztab : DfPath
        Path to the Casanovo mzTab file with the predicted peptides.
    peak_file : str
        Path to the MGF or mzML peak file.
    index : int
        Zero-based index of the spectrum to plot.
    ground_truth_mztab : DfPath
        Path to an mzTab file with the ground truth peptides.
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
    """
    top = _annotated_spectrum(
        mztab,
        peak_file,
        index,
        fragment_tol,
        fragment_tol_mode,
        ion_types,
        max_charge,
        max_isotope,
    )
    bottom = _annotated_spectrum(
        ground_truth_mztab,
        peak_file,
        index,
        fragment_tol,
        fragment_tol_mode,
        ion_types,
        max_charge,
        max_isotope,
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sup.mirror(top, bottom, ax=ax)
    gt_label = "Top: predicted, bottom: ground truth"
    ax.set_title(f"{title}\n{gt_label}" if title else gt_label)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}", file=sys.stderr)


COMMANDS: Commands = {"spectrum": spectrum, "mirror": mirror}
