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
import os
import sys
from collections import Counter
from os import PathLike
from typing import Iterable

import fire
# isort: off
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# isort: on
import numpy as np
try:
    from lark.exceptions import LarkError
    from pyteomics import mgf, proforma as pyteomics_proforma
    from spectrum_utils.spectrum import MsmsSpectrum
except Exception as e:
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
        charge_raw = spectrum["params"].get("charge", [])
        if isinstance(charge_raw, list):
            if len(charge_raw) != 1:
                n_skipped += 1
                continue
            charge = int(charge_raw[0])
        else:
            charge = int(charge_raw)
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
    """Return number of peaks per spectrum."""
    return [len(spectrum["m/z array"]) for spectrum in spectra]


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
    with mgf.MGF(mgf_file) as reader:
        counts_list = count_peaks(reader)

    counts_map = Counter(counts_list)
    total = len(counts_list)
    print(f"Processed {total} spectra total.", file=sys.stderr)

    # -- TSV output (sorted by n_peaks) ----------------------------------------
    with open(output_tsv, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["n_peaks", "count"])
        for n_peaks in sorted(counts_map):
            w.writerow([n_peaks, counts_map[n_peaks]])
    print(f"Wrote {output_tsv}", file=sys.stderr)

    # -- Histogram ------------------------------------------------------------
    if counts_list:
        arr = np.array(counts_list)
        title = (
            f"Peaks per spectrum (n={total:,}, median={int(np.median(arr))})"
        )
        fig = _make_histogram_fig(
            counts_list,
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
            f"  Warning: {n_skipped} spectra without SEQ= or with unknown"
            " modifications were skipped.",
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
# Fragment coverage — core computation helper
# ---------------------------------------------------------------------------


def _compute_coverage_results(
    spectra,
    tolerance: float = 10.0,
    tolerance_unit: str = "ppm",
) -> tuple[list, int]:
    """Compute fragment ion coverage for all spectra.

    Returns (results, n_skipped).
    results is a list of (scan, filename, seq, charge, n_peaks, n_matched, prop).
    """
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
        charge_raw = params.get("charge", [2])
        charge = (
            int(charge_raw[0]) if isinstance(charge_raw, list) else int(charge_raw)
        )
        pepmass_raw = params.get("pepmass", (0.0,))
        precursor_mz = float(
            pepmass_raw[0] if isinstance(pepmass_raw, (list, tuple)) else pepmass_raw
        )

        if not seq:
            n_skipped += 1
            continue

        obs_mz = spectrum_data["m/z array"]
        obs_int = spectrum_data["intensity array"]

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
                neutral_losses=True,
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
            1 for ann in spectrum.annotation
            if len(ann.fragment_annotations) > 0
        )
        matched_intensity = sum(
            spectrum.intensity[i]
            for i, ann in enumerate(spectrum.annotation)
            if len(ann.fragment_annotations) > 0
        )
        total_intensity = spectrum.intensity.sum()
        prop = float(matched_intensity / total_intensity) if total_intensity > 0 else 0.0

        results.append((scan, filename, seq, charge, n_peaks, n_matched, prop))

    return results, n_skipped


# ---------------------------------------------------------------------------
# Fragment coverage — main function
# ---------------------------------------------------------------------------


def fragment_coverage(
    mgf_file,
    tolerance=10.0,
    tolerance_unit="ppm",
    output_tsv="fragment_coverage.tsv",
    output_full_tsv="fragment_coverage.full.tsv",
    output_plot="fragment_coverage.png",
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
    """
    if tolerance_unit not in ("ppm", "Da"):
        raise ValueError(
            f"tolerance_unit must be 'ppm' or 'Da', got '{tolerance_unit}'"
        )

    with mgf.MGF(mgf_file) as reader:
        results, n_skipped = _compute_coverage_results(
            reader, tolerance, tolerance_unit
        )

    count = len(results) + n_skipped
    print(f"Processed {count} spectra total.", file=sys.stderr)
    print(f"  {len(results)} scored, {n_skipped} skipped.", file=sys.stderr)

    # -- Full per-spectrum TSV (in input order) --------------------------------
    with open(output_full_tsv, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["scan", "peptide", "charge", "coverage"])
        for scan, filename, seq, charge, n_peaks, n_matched, prop in results:
            w.writerow([scan, seq, charge, f"{prop:.6f}"])
    print(f"Wrote {output_full_tsv}", file=sys.stderr)

    # -- TSV output (sorted by proportion_matched) ----------------------------
    results.sort(key=lambda r: r[6])
    with open(output_tsv, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["scan", "filename", "sequence", "charge", "n_peaks",
                     "n_matched", "proportion_matched"])
        for scan, filename, seq, charge, n_peaks, n_matched, prop in results:
            w.writerow([scan, filename, seq, charge, n_peaks, n_matched,
                        f"{prop:.6f}"])
    print(f"Wrote {output_tsv}", file=sys.stderr)

    # -- Histogram -----------------------------------------------------------
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
    peak_counts_list: list,
    lengths_png: str | None,
    lengths: list,
    coverage_png: str | None,
    coverage_results: list,
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
        ths = "".join(f"<th>{h}</th>" for h in header_row)
        trs = "".join(
            "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
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

    if peak_counts_list:
        arr = np.array(peak_counts_list)
        peaks_stats = _table(
            ["Metric", "Value"],
            [
                ("Min peaks", f"{int(arr.min()):,}"),
                ("Max peaks", f"{int(arr.max()):,}"),
                ("Median peaks", f"{int(np.median(arr)):,}"),
                ("Mean peaks", f"{arr.mean():.1f}"),
            ],
        )
    else:
        peaks_stats = "<p class='note'>No data.</p>"

    lengths_section = ""
    if lengths_png is not None:
        if lengths:
            arr = np.array(lengths)
            lengths_stats = _table(
                ["Metric", "Value"],
                [
                    ("Min length", f"{int(arr.min()):,}"),
                    ("Max length", f"{int(arr.max()):,}"),
                    ("Median length", f"{int(np.median(arr)):,}"),
                    ("Mean length", f"{arr.mean():.1f}"),
                ],
            )
        else:
            lengths_stats = "<p class='note'>No annotated spectra.</p>"
        lengths_section = (
            f"<h2>Peptide Lengths</h2>{lengths_stats}{_img(lengths_png)}"
        )

    coverage_section = ""
    if coverage_png is not None:
        if coverage_results:
            proportions = np.array([r[6] for r in coverage_results])
            coverage_stats = _table(
                ["Metric", "Value"],
                [
                    ("Spectra scored", f"{len(proportions):,}"),
                    ("Min coverage", f"{proportions.min():.3f}"),
                    ("Max coverage", f"{proportions.max():.3f}"),
                    ("Median coverage", f"{np.median(proportions):.3f}"),
                    ("Mean coverage", f"{proportions.mean():.3f}"),
                ],
            )
        else:
            coverage_stats = "<p class='note'>No annotated spectra.</p>"
        coverage_section = (
            f"<h2>Fragment Ion Coverage</h2>{coverage_stats}{_img(coverage_png)}"
        )

    charge_img = _img(charge_png) if charge_png else ""
    peaks_img = _img(peaks_png) if peaks_png else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MGF Summary: {os.path.basename(mgf_file)}</title>
<style>{css}</style>
</head>
<body>
<h1>MGF Summary: {os.path.basename(mgf_file)}</h1>
<h2>Overview</h2>
{overview}
<h2>Charge State Distribution</h2>
{charge_table}
{charge_img}
<h2>Peaks per Spectrum</h2>
{peaks_stats}
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


# ---------------------------------------------------------------------------
# MGF summary — main function
# ---------------------------------------------------------------------------


def summarize_mgf(
    mgf_file: PathLike,
    output_root: PathLike = "mgf_summary",
    tolerance: float = 10.0,
    tolerance_unit: str = "ppm",
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
    """
    if tolerance_unit not in ("ppm", "Da"):
        raise ValueError(
            f"tolerance_unit must be 'ppm' or 'Da', got '{tolerance_unit}'"
        )

    os.makedirs(output_root, exist_ok=True)

    stem = os.path.basename(output_root)
    log_path = os.path.join(output_root, stem + ".log")
    _real_stderr = sys.stderr

    with open(log_path, "w", encoding="utf-8") as log_fh:
        sys.stderr = _Tee(_real_stderr, log_fh)
        try:
            print("Reading spectra ...", file=sys.stderr)
            spectra = list(mgf.MGF(mgf_file))
            total_spectra = len(spectra)
            print(f"  {total_spectra:,} spectra loaded.", file=sys.stderr)

            # -- Charge distribution ------------------------------------------
            print("Computing charge distribution ...", file=sys.stderr)
            charge_counts, _ = count_charge_states(spectra)
            n_with_charge = sum(charge_counts.values())
            print(f"  Spectra with charge: {n_with_charge:,}", file=sys.stderr)

            # -- Peak counts --------------------------------------------------
            print("Counting peaks per spectrum ...", file=sys.stderr)
            peak_counts_list = count_peaks(spectra)

            # -- Peptide lengths ----------------------------------------------
            print("Measuring peptide lengths ...", file=sys.stderr)
            lengths, n_len_skipped = measure_peptide_lengths(spectra)
            n_with_seq = sum(1 for s in spectra if s["params"].get("seq", ""))
            print(f"  Spectra with SEQ=: {n_with_seq:,}", file=sys.stderr)
            if n_len_skipped:
                print(
                    f"  {n_len_skipped} spectra skipped (no SEQ= or unknown"
                    " modification).",
                    file=sys.stderr,
                )

            # -- Fragment coverage (only if sequences present) ----------------
            coverage_results_list: list = []
            if n_with_seq > 0:
                print("Computing fragment ion coverage ...", file=sys.stderr)
                coverage_results_list, _ = _compute_coverage_results(
                    spectra, tolerance, tolerance_unit
                )
                print(
                    f"  {len(coverage_results_list):,} spectra scored.",
                    file=sys.stderr,
                )

            # -- Write TSV files ----------------------------------------------
            print("Writing TSV files ...", file=sys.stderr)

            def _write_tsv(path, header, rows):
                with open(path, "w", newline="") as fh:
                    w = csv.writer(fh, delimiter="\t")
                    w.writerow(header)
                    for row in rows:
                        w.writerow(row)
                print(f"  Wrote {path}", file=sys.stderr)

            charge_tsv = os.path.join(output_root, "charge_distribution.tsv")
            _write_tsv(
                charge_tsv,
                ["charge", "count"],
                [(c, charge_counts[c]) for c in sorted(charge_counts)],
            )

            peaks_counts_map = Counter(peak_counts_list)
            peaks_tsv = os.path.join(output_root, "peak_counts.tsv")
            _write_tsv(
                peaks_tsv,
                ["n_peaks", "count"],
                [(n, peaks_counts_map[n]) for n in sorted(peaks_counts_map)],
            )

            lengths_png: str | None = None
            if lengths:
                lengths_counts_map = Counter(lengths)
                lengths_tsv = os.path.join(output_root, "peptide_lengths.tsv")
                _write_tsv(
                    lengths_tsv,
                    ["length", "count"],
                    [(ln, lengths_counts_map[ln]) for ln in sorted(lengths_counts_map)],
                )

            coverage_png: str | None = None
            if coverage_results_list:
                coverage_results_list.sort(key=lambda r: r[6])
                coverage_tsv = os.path.join(output_root, "fragment_coverage.tsv")
                _write_tsv(
                    coverage_tsv,
                    ["scan", "filename", "sequence", "charge", "n_peaks",
                     "n_matched", "proportion_matched"],
                    [
                        (sc, fn, sq, ch, np_, nm, f"{pr:.6f}")
                        for sc, fn, sq, ch, np_, nm, pr in coverage_results_list
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
            if peak_counts_list:
                arr = np.array(peak_counts_list)
                peaks_png = _save_fig(
                    _make_histogram_fig(
                        peak_counts_list,
                        xlabel="Number of peaks",
                        title=(
                            f"Peaks per spectrum (n={total_spectra:,},"
                            f" median={int(np.median(arr))})"
                        ),
                    ),
                    "peak_counts.png",
                )

            if lengths:
                arr = np.array(lengths)
                lengths_png = _save_fig(
                    _make_histogram_fig(
                        lengths,
                        xlabel="Peptide length (residues)",
                        title=(
                            f"Peptide length distribution (n={len(lengths):,},"
                            f" median={int(np.median(arr))})"
                        ),
                        integer_bins=True,
                    ),
                    "peptide_lengths.png",
                )

            if coverage_results_list:
                proportions = np.array([r[6] for r in coverage_results_list])
                coverage_png = _save_fig(
                    _make_histogram_fig(
                        proportions,
                        xlabel="Proportion of intensity matched by b/y ions",
                        title=(
                            f"Fragment ion coverage (n={len(proportions):,},"
                            f" median={np.median(proportions):.3f})"
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
                peak_counts_list=peak_counts_list,
                lengths_png=lengths_png,
                lengths=lengths,
                coverage_png=coverage_png,
                coverage_results=coverage_results_list,
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


def charge_distribution_main() -> None:
    fire.Fire(charge_distribution)


def fragment_coverage_main() -> None:
    fire.Fire(fragment_coverage)


def peak_counts_main() -> None:
    fire.Fire(peak_counts)


def peptide_lengths_main() -> None:
    fire.Fire(peptide_lengths)


def summarize_mgf_main() -> None:
    fire.Fire(summarize_mgf)
