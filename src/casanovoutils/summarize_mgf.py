#!/usr/bin/env python
"""MGF file visualization tools.

fragment-coverage:
    Calculate the proportion of total fragment ion intensity explained by
    b- and y-ions (including neutral losses of NH3 and H2O) for each spectrum
    in an annotated MGF file.

    The peptide annotation is read from the SEQ= field (ProForma notation).

    Outputs:
      1. A TSV with columns: scan, filename, sequence, charge, n_peaks,
         n_matched, proportion_matched
      2. A histogram (PNG) of the proportion distribution.

charge-distribution:
    Compute the global charge state distribution from an MGF file.

    Outputs:
      1. A TSV with columns: charge, count
      2. A bar chart (PNG) of the distribution.

peak-counts:
    Count the number of peaks per spectrum in an MGF file.

    Outputs:
      1. A TSV with columns: n_peaks, count
      2. A histogram (PNG) of the distribution.

peptide-lengths:
    Measure peptide lengths for annotated spectra in an MGF file.
    Requires SEQ= field in ProForma notation.

    Outputs:
      1. A TSV with columns: length, count
      2. A histogram (PNG) of the distribution.

summarize-mgf:
    Produce a self-contained HTML summary of an MGF file with basic statistics
    and embedded histograms for charge state distribution, peaks per spectrum,
    peptide lengths, and fragment ion coverage.

Requires: pyteomics, spectrum_utils, numpy, matplotlib
"""

import csv
import html as html_mod
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from os import PathLike
from typing import Iterable

import fire

# isort: off
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# isort: on
import numpy as np

from .denovoutils import get_mgf_psms_df, process_spectrum

try:
    from lark.exceptions import LarkError
    from pyteomics import mgf, proforma as pyteomics_proforma
    from spectrum_utils.spectrum import MsmsSpectrum
except ImportError as e:
    print(f"Error: failed to import required packages: {e}", file=sys.stderr)
    print("Try: pip install --upgrade spectrum_utils numba pyteomics", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------


def _make_charge_fig(counts: dict) -> plt.Figure:
    """Create a bar chart figure for charge state distribution."""
    charges = sorted(counts)
    vals = [counts[c] for c in charges]
    total = sum(vals)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(c) for c in charges], vals, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Charge state")
    ax.set_ylabel("Number of spectra")
    ax.set_title(f"Charge state distribution (n={total:,})")
    fig.tight_layout()
    return fig


def _make_histogram_fig(
    values, xlabel: str, title: str, integer_bins: bool = False
) -> plt.Figure:
    """Create a histogram figure."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if integer_bins and len(values) > 0:
        arr = np.asarray(values)
        bins = np.arange(arr.min(), arr.max() + 2) - 0.5
        ax.hist(values, bins=bins, edgecolor="black", linewidth=0.5)
    else:
        ax.hist(values, bins=50, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of spectra")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _make_histogram_fig_from_counter(
    counter: Counter, xlabel: str, title: str, integer_bins: bool = False
) -> plt.Figure:
    """Create a histogram figure from a Counter without expanding to raw values."""
    keys = np.array(sorted(counter.keys()), dtype=float)
    weights = np.array([counter[k] for k in sorted(counter.keys())], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    if integer_bins and len(keys) <= 200:
        ax.bar(keys, weights, edgecolor="black", linewidth=0.5)
    else:
        n_bins = min(50, len(keys))
        bin_counts, bin_edges = np.histogram(keys, bins=n_bins, weights=weights)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = (bin_edges[1] - bin_edges[0]) * 0.9
        ax.bar(centers, bin_counts, width=width, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of spectra")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _make_histogram_fig_from_bins(
    bin_counts: np.ndarray, bin_edges: np.ndarray, xlabel: str, title: str
) -> plt.Figure:
    """Create a histogram figure from pre-binned data."""
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = (bin_edges[1] - bin_edges[0]) * 0.9
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(centers, bin_counts, width=width, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of spectra")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _counter_stats(counter: Counter) -> dict | None:
    """Compute min, max, median, mean from a Counter of numeric values."""
    if not counter:
        return None
    total = sum(counter.values())
    min_val = min(counter)
    max_val = max(counter)
    mean_val = sum(k * v for k, v in counter.items()) / total
    mid = total / 2.0
    cumsum = 0
    median_val = min_val
    for k in sorted(counter):
        cumsum += counter[k]
        if cumsum >= mid:
            median_val = k
            break
    return {"min": min_val, "max": max_val, "median": median_val, "mean": mean_val}


def _median_from_bins(bin_counts: np.ndarray, bin_edges: np.ndarray) -> float:
    """Estimate median from pre-binned histogram data using bin-centre interpolation."""
    total = bin_counts.sum()
    if total == 0:
        return 0.0
    mid = total / 2.0
    cumsum = 0
    for i, count in enumerate(bin_counts):
        cumsum += count
        if cumsum >= mid:
            return float((bin_edges[i] + bin_edges[i + 1]) / 2)
    return float((bin_edges[-2] + bin_edges[-1]) / 2)


# ---------------------------------------------------------------------------
# Charge parsing helper
# ---------------------------------------------------------------------------


def _parse_single_charge(charge_raw) -> int | None:
    """Return the charge as an int, or None if ambiguous or malformed.

    Accepts a scalar value or a single-element list as returned by pyteomics.
    Returns None for empty lists, multi-element lists, and values that cannot
    be converted to int.
    """
    try:
        if isinstance(charge_raw, list):
            if len(charge_raw) != 1:
                return None
            return int(charge_raw[0])
        return int(charge_raw)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Charge distribution
# ---------------------------------------------------------------------------


def count_charge_states(spectra: Iterable) -> tuple[dict[int, int], int]:
    """Count precursor charge states across spectra.

    Parameters
    ----------
    spectra : iterable
        Iterable of pyteomics spectrum dicts.

    Returns
    -------
    counts : dict[int, int]
        Mapping of charge state to count.
    n_skipped : int
        Number of spectra skipped (missing, empty, or multiple charge states).
    """
    counts: Counter[int] = Counter()
    n_skipped = 0
    for spectrum in spectra:
        charge = _parse_single_charge(spectrum["params"].get("charge", []))
        if charge is None:
            n_skipped += 1
            continue
        counts[charge] += 1
    return counts, n_skipped


def charge_distribution(
    mgf_file: PathLike,
    output_tsv: PathLike = "charge_distribution.tsv",
    output_plot: PathLike = "charge_distribution.png",
) -> None:
    """Charge state distribution for an MGF file.

    Parameters
    ----------
    mgf_file : PathLike
        Input MGF file.
    output_tsv : PathLike
        Output TSV path (default: charge_distribution.tsv).
    output_plot : PathLike
        Output bar chart path (default: charge_distribution.png).
    """
    with mgf.MGF(mgf_file) as reader:
        counts, n_skipped = count_charge_states(reader)

    total = sum(counts.values())
    print(f"Processed {total + n_skipped} spectra total.", file=sys.stderr)
    if n_skipped:
        print(
            f"  Warning: {n_skipped} spectra with multiple or missing charge"
            " states were skipped.",
            file=sys.stderr,
        )
    for charge in sorted(counts):
        print(f"  charge {charge}: {counts[charge]}", file=sys.stderr)

    if not counts:
        print(
            "  Warning: no spectra with valid charge states found.",
            file=sys.stderr,
        )

    # -- TSV output (sorted by charge) ----------------------------------------
    with open(output_tsv, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["charge", "count"])
        for charge in sorted(counts):
            w.writerow([charge, counts[charge]])
    print(f"Wrote {output_tsv}", file=sys.stderr)

    # -- Bar chart ------------------------------------------------------------
    fig = _make_charge_fig(counts)
    fig.savefig(output_plot, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_plot}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Peak counts
# ---------------------------------------------------------------------------


def count_peaks(spectra: Iterable) -> list[int]:
    """Return the number of peaks for each spectrum.

    Parameters
    ----------
    spectra : Iterable
        Iterable of pyteomics spectrum dicts.

    Returns
    -------
    list[int]
        Peak count for each spectrum, in input order.
    """
    return [len(s["m/z array"]) for s in spectra]


def peak_counts(
    mgf_file: PathLike,
    output_tsv: PathLike = "peak_counts.tsv",
    output_plot: PathLike = "peak_counts.png",
) -> None:
    """Number of peaks per spectrum in an MGF file.

    Parameters
    ----------
    mgf_file : PathLike
        Input MGF file.
    output_tsv : PathLike
        Output TSV path (default: peak_counts.tsv).
    output_plot : PathLike
        Output histogram path (default: peak_counts.png).
    """
    df = get_mgf_psms_df(mgf_file)
    counts_map = Counter(df["mgf_n_peaks"].to_list())
    total = len(df)
    print(f"Processed {total} spectra total.", file=sys.stderr)

    # -- TSV output (sorted by n_peaks) ----------------------------------------
    with open(output_tsv, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["n_peaks", "count"])
        for n_peaks in sorted(counts_map):
            w.writerow([n_peaks, counts_map[n_peaks]])
    print(f"Wrote {output_tsv}", file=sys.stderr)

    # -- Histogram ------------------------------------------------------------
    if counts_map:
        arr = np.array(df["mgf_n_peaks"].to_list())
        title = f"Peaks per spectrum (n={total:,}, median={int(np.median(arr))})"
        fig = _make_histogram_fig(
            arr,
            xlabel="Number of peaks",
            title=title,
        )
        fig.savefig(output_plot, dpi=150)
        plt.close(fig)
        print(f"Wrote {output_plot}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Peptide lengths
# ---------------------------------------------------------------------------


def measure_peptide_lengths(spectra: Iterable) -> tuple[list[int], int]:
    """Return (lengths, n_skipped) for annotated spectra.

    Spectra without SEQ= or with invalid ProForma sequences are counted
    as skipped.
    """
    lengths = []
    n_skipped = 0
    for spectrum in spectra:
        seq = spectrum["params"].get("seq", "")
        if not seq:
            n_skipped += 1
            continue
        try:
            parsed_seq, _ = pyteomics_proforma.parse(seq)
            lengths.append(len(parsed_seq))
        except Exception:
            n_skipped += 1
            continue
    return lengths, n_skipped


def peptide_lengths(
    mgf_file: PathLike,
    output_tsv: PathLike = "peptide_lengths.tsv",
    output_plot: PathLike = "peptide_lengths.png",
) -> None:
    """Peptide length distribution for annotated spectra in an MGF file.

    Parameters
    ----------
    mgf_file : PathLike
        Input MGF file.
    output_tsv : PathLike
        Output TSV path (default: peptide_lengths.tsv).
    output_plot : PathLike
        Output histogram path (default: peptide_lengths.png).
    """
    with mgf.MGF(mgf_file) as reader:
        lengths, n_skipped = measure_peptide_lengths(reader)

    total = len(lengths) + n_skipped
    print(f"Processed {total} spectra total.", file=sys.stderr)
    if n_skipped:
        print(
            f"  Warning: {n_skipped} spectra without SEQ= or with invalid"
            " ProForma sequences were skipped.",
            file=sys.stderr,
        )

    counts_map = Counter(lengths)

    # -- TSV output (sorted by length) ----------------------------------------
    with open(output_tsv, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["length", "count"])
        for length in sorted(counts_map):
            w.writerow([length, counts_map[length]])
    print(f"Wrote {output_tsv}", file=sys.stderr)

    # -- Histogram ------------------------------------------------------------
    if lengths:
        arr = np.array(lengths)
        title = (
            f"Peptide length distribution (n={len(lengths):,},"
            f" median={int(np.median(arr))})"
        )
        fig = _make_histogram_fig(
            lengths,
            xlabel="Peptide length (residues)",
            title=title,
            integer_bins=True,
        )
        fig.savefig(output_plot, dpi=150)
        plt.close(fig)
        print(f"Wrote {output_plot}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Fragment coverage — multiprocessing worker
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 500


def _annotate_chunk(args):
    """Annotate a chunk of spectra for fragment coverage.

    Defined at module level so it is picklable for multiprocessing spawn.

    Parameters
    ----------
    args : tuple
        (chunk, tolerance, tolerance_unit, max_charge, neutral_losses)
        where *chunk* is a list of
        (scan, filename, seq, charge, precursor_mz, obs_mz, obs_int).

    Returns
    -------
    list of tagged tuples:
        ("ok",         scan, filename, seq, charge, n_peaks, n_matched, prop)
        ("lark_error", scan, seq, err_str)
        ("error",      scan, seq, err_str)
    """
    chunk, tolerance, tolerance_unit, max_charge, neutral_losses = args
    results = []
    for scan, filename, seq, charge, precursor_mz, obs_mz, obs_int in chunk:
        max_ion_charge = charge if max_charge == "max" else None
        try:
            spectrum = MsmsSpectrum(
                identifier=scan,
                precursor_mz=precursor_mz,
                precursor_charge=charge,
                mz=obs_mz,
                intensity=obs_int.astype(np.float32),
            )
            spectrum.annotate_proforma(
                seq,
                fragment_tol_mass=tolerance,
                fragment_tol_mode=tolerance_unit,
                ion_types="by",
                max_isotope=1,
                max_ion_charge=max_ion_charge,
                neutral_losses=neutral_losses,
            )
        except LarkError as e:
            results.append(("lark_error", scan, seq, str(e)))
            continue
        except Exception as e:
            results.append(("error", scan, seq, str(e)))
            continue

        n_peaks = len(obs_mz)
        n_matched = sum(
            1 for ann in spectrum.annotation if len(ann.fragment_annotations) > 0
        )
        matched_intensity = sum(
            spectrum.intensity[i]
            for i, ann in enumerate(spectrum.annotation)
            if len(ann.fragment_annotations) > 0
        )
        total_intensity = spectrum.intensity.sum()
        prop = (
            float(matched_intensity / total_intensity) if total_intensity > 0 else 0.0
        )
        results.append(("ok", scan, filename, seq, charge, n_peaks, n_matched, prop))
    return results


# ---------------------------------------------------------------------------
# Fragment coverage — core computation helper
# ---------------------------------------------------------------------------


def _compute_coverage_results(
    spectra,
    tolerance: float = 10.0,
    tolerance_unit: str = "ppm",
    workers: int = 1,
    max_charge: str = "1less",
    neutral_losses: bool = True,
) -> tuple[list, int]:
    """Compute fragment ion coverage for all spectra.

    Returns (results, n_skipped).
    results is a list of (scan, filename, seq, charge, n_peaks, n_matched, prop).

    Parameters
    ----------
    spectra : iterable
        Iterable of pyteomics spectrum dicts.
    tolerance : float
        Fragment mass tolerance.
    tolerance_unit : str
        'ppm' or 'Da'.
    workers : int
        Number of worker processes. 1 = sequential (default).
    max_charge : str
        Maximum charge for fragment ions: 'max' (precursor charge) or
        '1less' (precursor charge minus one, the spectrum_utils default).
    neutral_losses : bool
        Whether to include neutral losses in annotation.
    """
    if not isinstance(workers, int) or workers < 1:
        raise ValueError(f"workers must be a positive integer, got {workers!r}")

    if workers == 1:
        # Sequential path
        results = []
        n_skipped = 0
        count = 0
        for spectrum_data in spectra:
            count += 1
            if count % 10000 == 0:
                print(f"  {count} spectra processed ...", file=sys.stderr)

            params = spectrum_data["params"]
            scan = str(params.get("scans", params.get("scan", f"idx_{count}")))
            filename = str(params.get("filename", ""))
            seq = params.get("seq", "")
            charge = _parse_single_charge(params.get("charge", []))
            if charge is None:
                n_skipped += 1
                continue
            pepmass_raw = params.get("pepmass", (0.0,))
            precursor_mz = float(
                pepmass_raw[0]
                if isinstance(pepmass_raw, (list, tuple))
                else pepmass_raw
            )

            if not seq:
                n_skipped += 1
                continue

            obs_mz = spectrum_data["m/z array"]
            obs_int = spectrum_data["intensity array"]
            max_ion_charge = charge if max_charge == "max" else None

            try:
                spectrum = MsmsSpectrum(
                    identifier=scan,
                    precursor_mz=precursor_mz,
                    precursor_charge=charge,
                    mz=obs_mz,
                    intensity=obs_int.astype(np.float32),
                )
                spectrum.annotate_proforma(
                    seq,
                    fragment_tol_mass=tolerance,
                    fragment_tol_mode=tolerance_unit,
                    ion_types="by",
                    max_isotope=1,
                    max_ion_charge=max_ion_charge,
                    neutral_losses=neutral_losses,
                )
            except LarkError as e:
                raise RuntimeError(
                    f"Failed to parse sequence {seq!r} as ProForma (scan {scan}).\n"
                    f"Modifications must use bracket notation, e.g. 'M[+15.995]' "
                    f"not 'M+15.995'.\nParser error: {e}"
                ) from None
            except Exception as e:
                print(f"  Skipping scan {scan} ({seq!r}): {e}", file=sys.stderr)
                n_skipped += 1
                continue

            n_peaks = len(obs_int)
            n_matched = sum(
                1 for ann in spectrum.annotation if len(ann.fragment_annotations) > 0
            )
            matched_intensity = sum(
                spectrum.intensity[i]
                for i, ann in enumerate(spectrum.annotation)
                if len(ann.fragment_annotations) > 0
            )
            total_intensity = spectrum.intensity.sum()
            prop = (
                float(matched_intensity / total_intensity)
                if total_intensity > 0
                else 0.0
            )
            results.append((scan, filename, seq, charge, n_peaks, n_matched, prop))

        return results, n_skipped

    # Parallel path
    results = []
    n_skipped = 0
    count = 0
    chunk: list = []
    futures = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for spectrum_data in spectra:
            count += 1
            if count % 10000 == 0:
                print(f"  {count} spectra read ...", file=sys.stderr)

            params = spectrum_data["params"]
            seq = params.get("seq", "")
            if not seq:
                n_skipped += 1
                continue

            scan = str(params.get("scans", params.get("scan", f"idx_{count}")))
            filename = str(params.get("filename", ""))
            charge = _parse_single_charge(params.get("charge", []))
            if charge is None:
                n_skipped += 1
                continue
            pepmass_raw = params.get("pepmass", (0.0,))
            precursor_mz = float(
                pepmass_raw[0]
                if isinstance(pepmass_raw, (list, tuple))
                else pepmass_raw
            )
            obs_mz = spectrum_data["m/z array"]
            obs_int = spectrum_data["intensity array"]

            chunk.append((scan, filename, seq, charge, precursor_mz, obs_mz, obs_int))
            if len(chunk) >= _CHUNK_SIZE:
                futures.append(
                    executor.submit(
                        _annotate_chunk,
                        (chunk, tolerance, tolerance_unit, max_charge, neutral_losses),
                    )
                )
                chunk = []

        if chunk:
            futures.append(
                executor.submit(
                    _annotate_chunk,
                    (chunk, tolerance, tolerance_unit, max_charge, neutral_losses),
                )
            )

        n_scored = 0
        for future in futures:
            for item in future.result():
                if item[0] == "ok":
                    _, scan, filename, seq, charge, n_peaks, n_matched, prop = item
                    results.append(
                        (scan, filename, seq, charge, n_peaks, n_matched, prop)
                    )
                    n_scored += 1
                    if n_scored % 10000 == 0:
                        print(f"  {n_scored} spectra scored ...", file=sys.stderr)
                elif item[0] == "lark_error":
                    _, scan, seq, err_str = item
                    raise RuntimeError(
                        f"Failed to parse sequence {seq!r} as ProForma (scan {scan}).\n"
                        f"Modifications must use bracket notation, e.g. 'M[+15.995]' "
                        f"not 'M+15.995'.\nParser error: {err_str}"
                    )
                else:
                    _, scan, seq, err_str = item
                    print(
                        f"  Skipping scan {scan} ({seq!r}): {err_str}", file=sys.stderr
                    )
                    n_skipped += 1

    return results, n_skipped


# ---------------------------------------------------------------------------
# Fragment coverage — main function
# ---------------------------------------------------------------------------


def fragment_coverage(
    mgf_file,
    tolerance=0.05,
    tolerance_unit="Da",
    output_tsv="fragment_coverage.tsv",
    output_full_tsv="fragment_coverage.full.tsv",
    output_plot="fragment_coverage.png",
    workers=1,
    max_charge="1less",
    neutral_losses=True,
):
    """Fragment ion intensity coverage for annotated MGF spectra.

    Parameters
    ----------
    mgf_file : str
        Annotated MGF file (with SEQ= in ProForma notation).
    tolerance : float
        Mass tolerance (default: 10).
    tolerance_unit : str
        Tolerance unit: 'ppm' or 'Da' (default: ppm).
    output_tsv : str
        Output TSV path (default: fragment_coverage.tsv).
    output_full_tsv : str
        Output per-spectrum TSV path (default: fragment_coverage.full.tsv).
    output_plot : str
        Output histogram path (default: fragment_coverage.png).
    workers : int
        Number of parallel worker processes (default: 1).
    max_charge : str
        Maximum charge state for fragment ions: 'max' (precursor charge)
        or '1less' (precursor charge minus one, default).
    neutral_losses : bool
        Include neutral losses in annotation (default: True).
    """
    if tolerance_unit not in ("ppm", "Da"):
        raise ValueError(
            f"tolerance_unit must be 'ppm' or 'Da', got '{tolerance_unit}'"
        )
    if max_charge not in ("max", "1less"):
        raise ValueError(f"max_charge must be 'max' or '1less', got '{max_charge}'")

    with mgf.MGF(mgf_file) as reader:
        results, n_skipped = _compute_coverage_results(
            reader,
            tolerance,
            tolerance_unit,
            workers=workers,
            max_charge=max_charge,
            neutral_losses=neutral_losses,
        )

    count = len(results) + n_skipped
    print(f"Processed {count} spectra total.", file=sys.stderr)
    print(f"  {len(results)} scored, {n_skipped} skipped.", file=sys.stderr)

    # -- Full per-spectrum TSV (in input order) --------------------------------
    with open(output_full_tsv, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["scan", "peptide", "charge", "coverage"])
        for scan, _, seq, charge, _, _, prop in results:
            w.writerow([scan, seq, charge, f"{prop:.6f}"])
    print(f"Wrote {output_full_tsv}", file=sys.stderr)

    # -- TSV output (sorted by proportion_matched) ----------------------------
    results.sort(key=lambda r: r[6])
    with open(output_tsv, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(
            [
                "scan",
                "filename",
                "sequence",
                "charge",
                "n_peaks",
                "n_matched",
                "proportion_matched",
            ]
        )
        for scan, filename, seq, charge, n_peaks, n_matched, prop in results:
            w.writerow([scan, filename, seq, charge, n_peaks, n_matched, f"{prop:.6f}"])
    print(f"Wrote {output_tsv}", file=sys.stderr)

    # -- Histogram -----------------------------------------------------------
    if results:
        proportions = np.array([r[6] for r in results])
        fig = _make_histogram_fig(
            proportions,
            xlabel="Proportion of intensity matched by b/y ions",
            title=(
                f"Fragment ion coverage  (n={len(proportions):,}, "
                f"median={np.median(proportions):.3f})"
            ),
        )
        fig.savefig(output_plot, dpi=150)
        plt.close(fig)
        print(f"Wrote {output_plot}", file=sys.stderr)


# ---------------------------------------------------------------------------
# MGF summary — HTML builder
# ---------------------------------------------------------------------------


def _build_summary_html(
    mgf_file: str,
    total_spectra: int,
    n_with_charge: int,
    n_with_seq: int,
    charge_png: str,
    charge_counts: dict,
    peaks_png: str,
    peaks_stats: dict | None,
    lengths_png: str | None,
    lengths_stats: dict | None,
    coverage_png: str | None,
    coverage_stats: dict | None,
    mod_counts: Counter | None = None,
    tolerance: float = 0.05,
    tolerance_unit: str = "Da",
    max_charge: str = "1less",
    neutral_losses: bool = True,
) -> str:
    """Build the HTML string for the MGF summary (links to external PNG/TSV)."""
    css = """
    body { font-family: sans-serif; margin: 2em; color: #222; }
    h1 { font-size: 1.6em; }
    h2 { font-size: 1.2em; margin-top: 2em; border-bottom: 1px solid #ccc; padding-bottom: 0.3em; }
    table { border-collapse: collapse; margin: 0.8em 0; font-size: 0.9em; }
    th, td { border: 1px solid #bbb; padding: 0.3em 0.8em; text-align: left; }
    th { background: #f0f0f0; }
    img { max-width: 700px; display: block; margin: 0.8em 0; }
    .note { color: #666; font-size: 0.85em; }
    """

    def _table(header_row, data_rows):
        ths = "".join(f"<th>{html_mod.escape(str(h))}</th>" for h in header_row)
        trs = "".join(
            "<tr>"
            + "".join(f"<td>{html_mod.escape(str(v))}</td>" for v in row)
            + "</tr>"
            for row in data_rows
        )
        return f"<table><tr>{ths}</tr>{trs}</table>"

    def _img(path):
        return f'<img src="{path}" alt="plot">'

    overview = _table(
        ["Metric", "Value"],
        [
            ("Total spectra", f"{total_spectra:,}"),
            ("Spectra with charge", f"{n_with_charge:,}"),
            ("Spectra with SEQ=", f"{n_with_seq:,}"),
        ],
    )

    charge_table = _table(
        ["Charge", "Count"],
        [(c, f"{charge_counts[c]:,}") for c in sorted(charge_counts)],
    )

    if peaks_stats:
        peaks_html = _table(
            ["Metric", "Value"],
            [
                ("Min peaks", f"{int(peaks_stats['min']):,}"),
                ("Max peaks", f"{int(peaks_stats['max']):,}"),
                ("Median peaks", f"{int(peaks_stats['median']):,}"),
                ("Mean peaks", f"{peaks_stats['mean']:.1f}"),
            ],
        )
    else:
        peaks_html = "<p class='note'>No data.</p>"

    lengths_section = ""
    if lengths_png is not None:
        if lengths_stats:
            lengths_html = _table(
                ["Metric", "Value"],
                [
                    ("Min length", f"{int(lengths_stats['min']):,}"),
                    ("Max length", f"{int(lengths_stats['max']):,}"),
                    ("Median length", f"{int(lengths_stats['median']):,}"),
                    ("Mean length", f"{lengths_stats['mean']:.1f}"),
                ],
            )
        else:
            lengths_html = "<p class='note'>No annotated spectra.</p>"
        lengths_section = f"<h2>Peptide Lengths</h2>{lengths_html}{_img(lengths_png)}"

    coverage_section = ""
    if coverage_png is not None:
        max_charge_label = (
            "precursor charge" if max_charge == "max" else "precursor charge - 1"
        )
        coverage_params = _table(
            ["Parameter", "Value"],
            [
                ("Ion types", "b, y"),
                ("Tolerance", f"{tolerance} {tolerance_unit}"),
                ("Max fragment charge", max_charge_label),
                ("Neutral losses", "yes" if neutral_losses else "no"),
            ],
        )
        if coverage_stats:
            coverage_stats_html = _table(
                ["Metric", "Value"],
                [
                    ("Spectra scored", f"{coverage_stats['n_scored']:,}"),
                    ("Min coverage", f"{coverage_stats['min']:.3f}"),
                    ("Max coverage", f"{coverage_stats['max']:.3f}"),
                    ("Median coverage", f"{coverage_stats['median']:.3f}"),
                    ("Mean coverage", f"{coverage_stats['mean']:.3f}"),
                ],
            )
        else:
            coverage_stats_html = "<p class='note'>No annotated spectra.</p>"
        coverage_section = (
            f"<h2>Fragment Ion Coverage</h2>"
            f"{coverage_params}"
            f"{coverage_stats_html}"
            f"{_img(coverage_png)}"
        )

    if mod_counts:
        mod_rows = sorted(mod_counts.items(), key=lambda x: -x[1])
        mods_section = "<h2>Amino Acid Modifications</h2>" + _table(
            ["Residue", "Modification", "Count"],
            [(residue, mod, f"{count:,}") for (residue, mod), count in mod_rows],
        )
    else:
        mods_section = ""

    charge_img = _img(charge_png) if charge_png else ""
    peaks_img = _img(peaks_png) if peaks_png else ""

    safe_name = html_mod.escape(os.path.basename(mgf_file))
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MGF Summary: {safe_name}</title>
<style>{css}</style>
</head>
<body>
<h1>MGF Summary: {safe_name}</h1>
<h2>Overview</h2>
{overview}
{mods_section}
<h2>Charge State Distribution</h2>
{charge_table}
{charge_img}
<h2>Peaks per Spectrum</h2>
{peaks_html}
{peaks_img}
{lengths_section}
{coverage_section}
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# MGF summary — stderr/log tee
# ---------------------------------------------------------------------------


class _Tee:
    """Write to multiple streams simultaneously."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        return False

    def fileno(self):
        raise AttributeError("_Tee does not support fileno()")


# ---------------------------------------------------------------------------
# MGF summary — main function
# ---------------------------------------------------------------------------


def summarize_mgf(
    mgf_file: PathLike,
    output_root: PathLike = "mgf_summary",
    tolerance: float = 0.05,
    tolerance_unit: str = "Da",
    workers: int = 1,
    max_charge: str = "1less",
    neutral_losses: bool = True,
) -> None:
    """Produce a self-contained HTML summary of an MGF file.

    Parameters
    ----------
    mgf_file : PathLike
        Input MGF file.
    output_root : PathLike
        Output directory name; the HTML file inside will share this basename
        (default: mgf_summary).
    tolerance : float
        Fragment mass tolerance for coverage calculation (default: 10).
    tolerance_unit : str
        Tolerance unit: 'ppm' or 'Da' (default: ppm).
    workers : int
        Number of parallel worker processes for coverage annotation (default: 1).
    max_charge : str
        Maximum charge state for fragment ions: 'max' (precursor charge)
        or '1less' (precursor charge minus one, default).
    neutral_losses : bool
        Include neutral losses in annotation (default: True).
    """
    if tolerance_unit not in ("ppm", "Da"):
        raise ValueError(
            f"tolerance_unit must be 'ppm' or 'Da', got '{tolerance_unit}'"
        )
    if max_charge not in ("max", "1less"):
        raise ValueError(f"max_charge must be 'max' or '1less', got '{max_charge}'")

    os.makedirs(output_root, exist_ok=True)

    stem = os.path.basename(output_root)
    log_path = os.path.join(output_root, stem + ".log")
    _real_stderr = sys.stderr

    with open(log_path, "w", encoding="utf-8") as log_fh:
        sys.stderr = _Tee(_real_stderr, log_fh)
        try:
            # ----------------------------------------------------------------
            # Pass 1: Fast streaming pass for charge/peak/length statistics.
            # No annotation — just counting.
            # ----------------------------------------------------------------
            total_spectra = 0
            n_with_seq = 0
            charge_counts: Counter[int] = Counter()
            peak_counts_counter: Counter[int] = Counter()
            length_counts: Counter[int] = Counter()

            mod_counts: Counter[tuple[str, str]] = Counter()

            n_parse_errors = 0
            print("Reading spectra ...", file=sys.stderr)
            with mgf.MGF(mgf_file) as reader:
                for spectrum_data in reader:
                    total_spectra += 1
                    if total_spectra % 1000 == 0:
                        print(
                            f"  {total_spectra:,} spectra loaded ...",
                            file=sys.stderr,
                        )

                    params = process_spectrum(spectrum_data)

                    # Charge distribution
                    charge = _parse_single_charge(params.get("charge", []))
                    if charge is not None:
                        charge_counts[charge] += 1

                    # Peak counts
                    n_peaks = params["n_peaks"]
                    peak_counts_counter[n_peaks] += 1

                    # Sequence-based analyses (length + modifications — no annotation)
                    seq = params.get("seq", "")
                    if not seq:
                        continue

                    n_with_seq += 1
                    try:
                        parsed_seq, props = pyteomics_proforma.parse(seq)
                        length_counts[len(parsed_seq)] += 1
                        for residue, mods in parsed_seq:
                            for mod in mods or []:
                                mod_counts[(residue, str(mod))] += 1
                        for mod in props.get("n_term") or []:
                            mod_counts[("N-term", str(mod))] += 1
                        for mod in props.get("c_term") or []:
                            mod_counts[("C-term", str(mod))] += 1
                    except Exception:
                        n_parse_errors += 1

            n_with_charge = sum(charge_counts.values())
            print(f"  {total_spectra:,} spectra loaded.", file=sys.stderr)
            if n_parse_errors:
                print(
                    f"  Warning: {n_parse_errors} spectra with unparseable"
                    " SEQ= skipped.",
                    file=sys.stderr,
                )
            print("Computing charge distribution ...", file=sys.stderr)
            print(f"  Spectra with charge: {n_with_charge:,}", file=sys.stderr)
            print("Counting peaks per spectrum ...", file=sys.stderr)
            print("Measuring peptide lengths ...", file=sys.stderr)
            print(f"  Spectra with SEQ=: {n_with_seq:,}", file=sys.stderr)

            # ----------------------------------------------------------------
            # Pass 2: Fragment coverage annotation (sequential or parallel).
            # Reads the MGF file a second time.
            # ----------------------------------------------------------------
            print("Computing fragment ion coverage ...", file=sys.stderr)
            with mgf.MGF(mgf_file) as reader:
                cov_results, n_cov_skipped = _compute_coverage_results(
                    reader,
                    tolerance,
                    tolerance_unit,
                    workers=workers,
                    max_charge=max_charge,
                    neutral_losses=neutral_losses,
                )

            cov_n = len(cov_results)
            if n_cov_skipped:
                print(
                    f"  {n_cov_skipped} annotated spectra skipped (errors).",
                    file=sys.stderr,
                )
            print(f"  {cov_n:,} spectra scored.", file=sys.stderr)

            # Build pre-binned histogram from coverage results
            _N_COV_BINS = 50
            _cov_bin_edges = np.linspace(0.0, 1.0, _N_COV_BINS + 1)
            cov_bin_counts = np.zeros(_N_COV_BINS, dtype=np.int64)
            cov_min = 1.0
            cov_max = 0.0
            cov_sum = 0.0
            for _, _, _, _, _, _, prop in cov_results:
                bin_idx = min(int(prop * _N_COV_BINS), _N_COV_BINS - 1)
                cov_bin_counts[bin_idx] += 1
                if prop < cov_min:
                    cov_min = prop
                if prop > cov_max:
                    cov_max = prop
                cov_sum += prop

            # -- Compute summary stats ----------------------------------------
            peaks_stats = _counter_stats(peak_counts_counter)
            lengths_stats = _counter_stats(length_counts) if length_counts else None
            coverage_stats = None
            if cov_n > 0:
                coverage_stats = {
                    "n_scored": cov_n,
                    "min": cov_min,
                    "max": cov_max,
                    "median": _median_from_bins(cov_bin_counts, _cov_bin_edges),
                    "mean": cov_sum / cov_n,
                }

            # -- Write TSV files ----------------------------------------------
            print("Writing TSV files ...", file=sys.stderr)

            def _write_tsv(path, header, rows):
                with open(path, "w", newline="") as fh:
                    w = csv.writer(fh, delimiter="\t")
                    w.writerow(header)
                    for row in rows:
                        w.writerow(row)
                print(f"  Wrote {path}", file=sys.stderr)

            _write_tsv(
                os.path.join(output_root, "charge_distribution.tsv"),
                ["charge", "count"],
                [(c, charge_counts[c]) for c in sorted(charge_counts)],
            )
            _write_tsv(
                os.path.join(output_root, "peak_counts.tsv"),
                ["n_peaks", "count"],
                [(n, peak_counts_counter[n]) for n in sorted(peak_counts_counter)],
            )
            if length_counts:
                _write_tsv(
                    os.path.join(output_root, "peptide_lengths.tsv"),
                    ["length", "count"],
                    [(ln, length_counts[ln]) for ln in sorted(length_counts)],
                )
            if cov_n > 0:
                cov_tsv_path = os.path.join(output_root, "fragment_coverage.tsv")
                _write_tsv(
                    cov_tsv_path,
                    [
                        "scan",
                        "filename",
                        "sequence",
                        "charge",
                        "n_peaks",
                        "n_matched",
                        "proportion_matched",
                    ],
                    [
                        (scan, filename, seq, charge, n_peaks, n_matched, f"{prop:.6f}")
                        for scan, filename, seq, charge, n_peaks, n_matched, prop in cov_results
                    ],
                )

            # -- Save PNG files -----------------------------------------------
            print("Saving figures ...", file=sys.stderr)

            def _save_fig(fig, rel_name):
                path = os.path.join(output_root, rel_name)
                fig.savefig(path, dpi=150)
                plt.close(fig)
                print(f"  Wrote {path}", file=sys.stderr)
                return rel_name

            charge_png = ""
            if charge_counts:
                charge_png = _save_fig(
                    _make_charge_fig(charge_counts), "charge_distribution.png"
                )

            peaks_png = ""
            if peak_counts_counter and peaks_stats:
                peaks_png = _save_fig(
                    _make_histogram_fig_from_counter(
                        peak_counts_counter,
                        xlabel="Number of peaks",
                        title=(
                            f"Peaks per spectrum (n={total_spectra:,},"
                            f" median={int(peaks_stats['median'])})"
                        ),
                    ),
                    "peak_counts.png",
                )

            lengths_png: str | None = None
            if length_counts and lengths_stats:
                n_lengths = sum(length_counts.values())
                lengths_png = _save_fig(
                    _make_histogram_fig_from_counter(
                        length_counts,
                        xlabel="Peptide length (residues)",
                        title=(
                            f"Peptide length distribution (n={n_lengths:,},"
                            f" median={int(lengths_stats['median'])})"
                        ),
                        integer_bins=True,
                    ),
                    "peptide_lengths.png",
                )

            coverage_png: str | None = None
            if cov_n > 0 and coverage_stats:
                coverage_png = _save_fig(
                    _make_histogram_fig_from_bins(
                        cov_bin_counts,
                        _cov_bin_edges,
                        xlabel="Proportion of intensity matched by b/y ions",
                        title=(
                            f"Fragment ion coverage (n={cov_n:,},"
                            f" median={coverage_stats['median']:.3f})"
                        ),
                    ),
                    "fragment_coverage.png",
                )

            # -- Build and write HTML -----------------------------------------
            print("Writing HTML ...", file=sys.stderr)
            html = _build_summary_html(
                mgf_file=str(mgf_file),
                total_spectra=total_spectra,
                n_with_charge=n_with_charge,
                n_with_seq=n_with_seq,
                charge_png=charge_png,
                charge_counts=charge_counts,
                peaks_png=peaks_png,
                peaks_stats=peaks_stats,
                lengths_png=lengths_png,
                lengths_stats=lengths_stats,
                coverage_png=coverage_png,
                coverage_stats=coverage_stats,
                mod_counts=mod_counts,
                tolerance=tolerance,
                tolerance_unit=tolerance_unit,
                max_charge=max_charge,
                neutral_losses=neutral_losses,
            )

            out_path = os.path.join(output_root, stem + ".html")
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(html)
            print(f"Wrote {out_path}", file=sys.stderr)
            print(f"Wrote {log_path}", file=sys.stderr)

        finally:
            sys.stderr = _real_stderr


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


COMMANDS = {
    "summarize": summarize_mgf,
    "charge-distribution": charge_distribution,
    "fragment-coverage": fragment_coverage,
    "peak-counts": peak_counts,
    "peptide-lengths": peptide_lengths,
}


def main() -> None:
    fire.Fire(COMMANDS)
