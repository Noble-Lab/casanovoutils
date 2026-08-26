"""
Visualize the top-k incorrectly predicted spectra from a Casanovo run.

For each of the k highest-scoring incorrect predictions, a mirror plot is
produced: the top panel annotates the spectrum with the *predicted* ProForma
sequence; the bottom (mirrored) panel annotates the same spectrum with the
*ground truth* ProForma sequence.  A text header on each figure shows key
mzTab fields (score, charge, precursor m/z, Δm/z in Da and ppm, scan number).
"""

import logging
import pathlib
import sys
from os import PathLike
from typing import Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyteomics.mgf
import tqdm

from . import configure_logging
from .constants import Constants
from .denovoutils import get_ground_truth_df
from .preccov import (
    calc_precision_coverage,
    fill_null_columns,
    tokenize_and_parse_scores,
)
from .types import Commands

try:
    from spectrum_utils.spectrum import MsmsSpectrum
    import spectrum_utils.plot as sup
except ImportError as e:  # pragma: no cover
    print(f"Error: failed to import spectrum_utils: {e}", file=sys.stderr)
    print("Try: pip install --upgrade spectrum_utils", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_mgf_peaks(mgf_file: PathLike) -> dict[int, dict]:
    """Return a dict mapping 0-based spectrum index → pyteomics spectrum dict."""
    result: dict[int, dict] = {}
    with pyteomics.mgf.read(str(mgf_file), use_index=False) as reader:
        for i, spectrum in enumerate(
            tqdm.tqdm(reader, desc="Loading MGF peaks", unit="spectrum")
        ):
            result[i] = spectrum
    return result


def _delta_mz(exp_mz: float, calc_mz: float) -> tuple[float, float]:
    """Return (delta_Da, delta_ppm) between experimental and calculated m/z."""
    delta_da = exp_mz - calc_mz
    delta_ppm = delta_da / calc_mz * 1e6 if calc_mz else float("nan")
    return delta_da, delta_ppm


def _make_spectrum(spectrum_dict: dict, identifier: str) -> MsmsSpectrum:
    """Build a spectrum_utils MsmsSpectrum from a pyteomics spectrum dict."""
    params = spectrum_dict["params"]
    mz_array = np.asarray(spectrum_dict["m/z array"], dtype=float)
    intensity_array = np.asarray(spectrum_dict["intensity array"], dtype=float)

    pepmass = params.get("pepmass", (0.0,))
    precursor_mz = (
        float(pepmass[0]) if isinstance(pepmass, (tuple, list)) else float(pepmass)
    )

    charge_raw = params.get("charge", [2])
    if isinstance(charge_raw, list):
        charge = int(charge_raw[0]) if charge_raw else 2
    else:
        charge = int(charge_raw)

    return MsmsSpectrum(
        identifier=identifier,
        precursor_mz=precursor_mz,
        precursor_charge=charge,
        mz=mz_array,
        intensity=intensity_array,
    )


def _make_mirror_plot(
    spectrum_dict: dict,
    predicted_seq: str,
    ground_truth_seq: str,
    score: float,
    exp_mz: float,
    calc_mz: float,
    charge: int,
    scan: str,
    rank: int,
    fragment_tol: float,
    fragment_tol_mode: str,
    ion_types: str,
    neutral_losses: bool,
) -> plt.Figure:
    """Create a mirror plot: predicted on top panel, ground truth on bottom."""
    identifier = f"rank{rank:04d}"

    spec_top = _make_spectrum(spectrum_dict, identifier)
    spec_bottom = _make_spectrum(spectrum_dict, identifier)

    for spec in (spec_top, spec_bottom):
        spec.filter_intensity(min_intensity=0.0)
        spec.scale_intensity(scaling="root")

    nl_arg: bool | dict = (
        {"NH3": -17.02655, "H2O": -18.01056} if neutral_losses else False
    )

    try:
        spec_top.annotate_proforma(
            predicted_seq,
            fragment_tol_mass=fragment_tol,
            fragment_tol_mode=fragment_tol_mode,
            ion_types=ion_types,
            neutral_losses=nl_arg,
        )
    except Exception as exc:
        logging.warning("Could not annotate predicted seq %r: %s", predicted_seq, exc)

    try:
        spec_bottom.annotate_proforma(
            ground_truth_seq,
            fragment_tol_mass=fragment_tol,
            fragment_tol_mode=fragment_tol_mode,
            ion_types=ion_types,
            neutral_losses=nl_arg,
        )
    except Exception as exc:
        logging.warning(
            "Could not annotate ground truth seq %r: %s", ground_truth_seq, exc
        )

    delta_da, delta_ppm = _delta_mz(exp_mz, calc_mz)

    fig, ax = plt.subplots(figsize=(12, 6))
    sup.mirror(spec_top, spec_bottom, ax=ax)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("m/z", fontsize=10)
    ax.set_ylabel("Intensity (normalized)", fontsize=10)

    title_lines = [
        f"Rank {rank}  |  score = {score:.4f}  |  scan = {scan}",
        (
            f"z = {charge}   exp m/z = {exp_mz:.4f}   calc m/z = {calc_mz:.4f}"
            f"   \u0394m/z = {delta_da:+.4f} Da  ({delta_ppm:+.1f} ppm)"
        ),
        f"Predicted (top):       {predicted_seq}",
        f"Ground truth (bottom): {ground_truth_seq}",
    ]
    fig.suptitle(
        "\n".join(title_lines),
        fontsize=8,
        ha="left",
        x=0.01,
        y=1.02,
        va="bottom",
        fontfamily="monospace",
    )
    plt.tight_layout()
    return fig


def _parse_mgf_idx(spectra_ref: str) -> Optional[int]:
    """Extract the 0-based spectrum index from a spectra_ref string."""
    if "index=" in spectra_ref:
        try:
            return int(spectra_ref.split("index=")[-1])
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Public command
# ---------------------------------------------------------------------------


def visualize_errors(
    mgf_file: PathLike,
    mztab_file: PathLike,
    output_dir: PathLike = "visualize_errors",
    k: int = 10,
    fragment_tol: float = 0.05,
    fragment_tol_mode: str = "Da",
    ion_types: str = "by",
    neutral_losses: bool = False,
    distinct_il: bool = False,
    overwrite: bool = False,
    residues_path: Optional[PathLike] = None,
) -> None:
    """
    Plot the top-k incorrectly predicted spectra from a Casanovo run.

    Loads an annotated MGF file and a Casanovo mzTab file, joins them,
    identifies incorrect predictions using mass-based matching, sorts by
    descending Casanovo score, and writes one mirror plot per spectrum to
    *output_dir*.  Each plot shows the predicted sequence annotated on the
    top panel and the ground truth sequence on the mirrored bottom panel,
    with a header block displaying score, charge, precursor m/z, Δm/z in
    Da and ppm, and scan number.

    By default isoleucine (I) and leucine (L) are treated as equivalent
    when deciding whether a prediction is correct (they have the same
    monoisotopic mass and cannot be distinguished by standard CID/HCD
    fragmentation).  Pass ``--distinct_il`` to treat them as distinct
    amino acids instead.

    Parameters
    ----------
    mgf_file : PathLike
        Annotated MGF file (must have ``SEQ=`` fields in ProForma notation).
    mztab_file : PathLike
        Casanovo mzTab output file.
    output_dir : PathLike, optional
        Directory for output PNG files and the log file (default
        ``"visualize_errors"``).
    k : int, optional
        Maximum number of spectra to plot (default 10).
    fragment_tol : float, optional
        Fragment ion mass tolerance for b/y annotation (default 0.05).
    fragment_tol_mode : str, optional
        Tolerance unit: ``"Da"`` or ``"ppm"`` (default ``"Da"``).
    ion_types : str, optional
        Ion series to annotate, e.g. ``"by"`` (default) or ``"abcxyz"``.
    neutral_losses : bool, optional
        If ``True``, annotate NH3 and H2O neutral losses (default ``False``).
    distinct_il : bool, optional
        If ``True``, treat isoleucine (I) and leucine (L) as distinct amino
        acids when deciding correctness.  By default (``False``) they are
        considered equivalent, which is the standard practice for CID/HCD
        data because the two residues have identical monoisotopic masses.
    overwrite : bool, optional
        If ``False`` (default), raise ``FileExistsError`` if *output_dir*
        already contains PNG files from a previous run.
    residues_path : PathLike, optional
        Path to a custom residue mass YAML file.  If ``None``, the bundled
        ``residues.yaml`` is used.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_pngs = list(output_dir.glob("rank_*.png"))
    if existing_pngs and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains {len(existing_pngs)} rank_*.png "
            "file(s) from a previous run. Use --overwrite to replace them."
        )

    log_path = output_dir / "visualize_errors.log"
    file_handler = configure_logging(log_path)

    try:
        logging.info(
            "visualize_errors: mgf=%s  mztab=%s  k=%d  distinct_il=%s",
            mgf_file,
            mztab_file,
            k,
            distinct_il,
        )

        # ── 1. Build merged ground-truth + prediction DataFrame ──────────────
        pc_df = get_ground_truth_df(mgf_file, mztab_file)

        pred_col = Constants.get_pred_sequence_column(pc_df)
        logging.debug("Predicted sequence column: %s", pred_col)

        aa_col = Constants.get_aa_scores_column(pc_df)
        if aa_col != Constants.aa_scores_column:
            if Constants.aa_scores_column in pc_df.columns:
                pc_df = pc_df.drop(Constants.aa_scores_column)
            pc_df = pc_df.rename({aa_col: Constants.aa_scores_column})

        pc_df = fill_null_columns(pc_df, pred_col)
        pc_df = tokenize_and_parse_scores(
            pc_df,
            pred_col,
            residues_path,
            replace_isoleucine_with_leucine=not distinct_il,
        )

        # ── 2. Mass-based correctness ─────────────────────────────────────────
        pc_df = calc_precision_coverage(pc_df, Constants.pep_score_column)

        n_total = len(pc_df)
        n_wrong = int((~pc_df["pc_is_correct"]).sum())
        logging.info("%d / %d predictions are incorrect", n_wrong, n_total)

        # ── 3. Top-k incorrect by descending score ───────────────────────────
        wrong_df = (
            pc_df.filter(~pl.col("pc_is_correct"))
            .sort(Constants.pep_score_column, descending=True)
            .head(k)
        )

        if len(wrong_df) == 0:
            logging.info("No incorrect predictions — nothing to plot.")
            return

        logging.info("Plotting top %d of %d incorrect spectra", len(wrong_df), n_wrong)

        # ── 4. Load raw peak arrays indexed by MGF position ─────────────────
        mgf_peaks = _load_mgf_peaks(mgf_file)

        # ── 5. One mirror plot per spectrum ──────────────────────────────────
        spectra_ref_col = "mztab_spectra_ref"
        has_ref = spectra_ref_col in wrong_df.columns

        plotted = 0
        for rank, row in enumerate(wrong_df.iter_rows(named=True), start=1):
            predicted_seq = row.get(pred_col) or ""
            ground_truth_seq = row.get(Constants.ground_truth_sequence_column) or ""
            score = float(row.get(Constants.pep_score_column) or 0.0)
            charge = int(row.get("mztab_charge") or 2)
            exp_mz = float(row.get("mztab_exp_mass_to_charge") or 0.0)
            calc_mz = float(row.get("mztab_calc_mass_to_charge") or 0.0)

            scan_raw = row.get("mztab_opt_global_cv_MS:1003057_scan_number")
            if scan_raw is None:
                scan_raw = row.get("mgf_scan") or row.get("mgf_scans")
            scan = str(scan_raw) if scan_raw is not None else "?"
            # Strip the "ms_run[N]:scan=" prefix that Casanovo writes so that
            # the filename contains only the bare integer.  Colons and brackets
            # are invalid (or trigger NTFS alternate-data-stream behaviour) on
            # Windows, so we must not include them in the output path.
            if "scan=" in scan:
                scan = scan.split("scan=")[-1]

            mgf_idx: Optional[int] = None
            if has_ref:
                ref_str = str(row.get(spectra_ref_col) or "")
                mgf_idx = _parse_mgf_idx(ref_str)

            if mgf_idx is None or mgf_idx not in mgf_peaks:
                logging.warning(
                    "Rank %d: cannot find MGF peaks (ref=%s) — skipping",
                    rank,
                    ref_str if has_ref else "?",
                )
                continue

            fig = _make_mirror_plot(
                spectrum_dict=mgf_peaks[mgf_idx],
                predicted_seq=predicted_seq,
                ground_truth_seq=ground_truth_seq,
                score=score,
                exp_mz=exp_mz,
                calc_mz=calc_mz,
                charge=charge,
                scan=scan,
                rank=rank,
                fragment_tol=fragment_tol,
                fragment_tol_mode=fragment_tol_mode,
                ion_types=ion_types,
                neutral_losses=neutral_losses,
            )

            out_path = output_dir / f"rank_{rank:04d}_scan_{scan}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logging.info("Saved %s", out_path)
            plotted += 1

        logging.info("Done. %d plots written to %s", plotted, output_dir)

    finally:
        if file_handler is not None:
            logging.root.removeHandler(file_handler)
            file_handler.close()


COMMANDS: Commands = visualize_errors
