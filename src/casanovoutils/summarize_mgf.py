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

Requires: pyteomics, numpy, matplotlib
"""

import base64
import csv
import io
import os
import re
import sys
from os import PathLike
from typing import Iterable

import fire
# isort: off
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# isort: on
import numpy as np
from pyteomics import mgf, mass

# ---------------------------------------------------------------------------
# Fragment coverage — physical constants
# ---------------------------------------------------------------------------
PROTON_MASS = 1.00727646677
H2O_MASS = 18.01056468

# Standard amino acid residue masses (monoisotopic)
AA_MASS = mass.std_aa_mass.copy()

# Modification delta masses (monoisotopic, from Unimod)
MOD_MASS = {
    "Acetyl": 42.010565,
    "Ammonia-loss": -17.026549,
    "Carbamidomethyl": 57.021464,
    "Carbamyl": 43.005814,
    "Deamidated": 0.984016,
    "Oxidation": 15.994915,
}

# Neutral losses applied to every b/y ion (Da)
NEUTRAL_LOSSES = [0.0, 17.026549, 18.010565]  # none, NH3, H2O

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


def _fig_to_base64(fig) -> str:
    """Encode a matplotlib figure as a base64 PNG string and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# Fragment coverage — ProForma parsing
# ---------------------------------------------------------------------------


def parse_proforma(seq_str):
    """Parse a ProForma sequence string.

    Returns (n_term_mod_mass, list_of_residue_masses) or None if the
    sequence contains an unrecognised modification or amino acid.
    """
    n_term_mod = 0.0

    # N-terminal modification: [Mod]-
    m = re.match(r"^\[([^\]]+)\]-", seq_str)
    if m:
        mod = m.group(1)
        if mod not in MOD_MASS:
            return None
        n_term_mod = MOD_MASS[mod]
        seq_str = seq_str[m.end():]

    residue_masses = []
    i = 0
    while i < len(seq_str):
        aa = seq_str[i]
        if aa not in AA_MASS:
            return None
        rm = AA_MASS[aa]
        i += 1
        # Collect any bracketed modifications on this residue
        while i < len(seq_str) and seq_str[i] == "[":
            j = seq_str.index("]", i)
            mod = seq_str[i + 1 : j]
            if mod not in MOD_MASS:
                return None
            rm += MOD_MASS[mod]
            i = j + 1
        residue_masses.append(rm)

    return n_term_mod, residue_masses


# ---------------------------------------------------------------------------
# Fragment coverage — theoretical fragment computation
# ---------------------------------------------------------------------------


def theoretical_mzs(n_term_mod, residue_masses, precursor_charge):
    """Return sorted array of theoretical b/y m/z values incl. neutral losses.

    Fragment charge states range from 1 to max(precursor_charge - 1, 1).
    """
    n = len(residue_masses)
    if n < 2:
        return np.array([])

    cum = np.cumsum(residue_masses)
    max_z = max(precursor_charge - 1, 1)

    mzs = []
    for i in range(1, n):
        # b_i neutral mass = sum of first i residues + N-term mod
        b_neutral = cum[i - 1] + n_term_mod
        # y_(n-i) neutral mass = sum of last (n-i) residues + H2O
        y_neutral = cum[-1] - cum[i - 1] + H2O_MASS

        for nl in NEUTRAL_LOSSES:
            bn = b_neutral - nl
            yn = y_neutral - nl
            for z in range(1, max_z + 1):
                if bn > 0:
                    mzs.append((bn + z * PROTON_MASS) / z)
                if yn > 0:
                    mzs.append((yn + z * PROTON_MASS) / z)

    return np.sort(mzs)


# ---------------------------------------------------------------------------
# Fragment coverage — peak matching
# ---------------------------------------------------------------------------


def matched_proportion(obs_mz, obs_int, theo, tolerance, tol_unit="ppm"):
    """Fraction of observed intensity matched by any theoretical m/z.

    Returns (n_matched_peaks, proportion).
    """
    total = obs_int.sum()
    if total == 0 or len(theo) == 0:
        return 0, 0.0

    order = np.argsort(obs_mz)
    mz_s = obs_mz[order]
    int_s = obs_int[order]
    matched = np.zeros(len(mz_s), dtype=bool)

    for t in theo:
        tol = t * tolerance * 1e-6 if tol_unit == "ppm" else tolerance
        lo = np.searchsorted(mz_s, t - tol, side="left")
        hi = np.searchsorted(mz_s, t + tol, side="right")
        matched[lo:hi] = True

    return int(matched.sum()), int_s[matched].sum() / total


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
    counts = {}
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
        counts[charge] = counts.get(charge, 0) + 1
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

    counts_map: dict[int, int] = {}
    for n in counts_list:
        counts_map[n] = counts_map.get(n, 0) + 1

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
    """Return (lengths, n_skipped) using parse_proforma for residue count.

    Spectra without SEQ= or with unrecognised modifications are counted
    as skipped.
    """
    lengths = []
    n_skipped = 0
    for spectrum in spectra:
        seq = spectrum["params"].get("seq", "")
        if not seq:
            n_skipped += 1
            continue
        parsed = parse_proforma(seq)
        if parsed is None:
            n_skipped += 1
            continue
        _, residue_masses = parsed
        lengths.append(len(residue_masses))
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

    counts_map: dict[int, int] = {}
    for length in lengths:
        counts_map[length] = counts_map.get(length, 0) + 1

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
) -> tuple[list, int, set]:
    """Compute fragment ion coverage for all spectra.

    Returns (results, n_skipped, unknown_mods).
    results is a list of (scan, filename, seq, charge, n_peaks, n_matched, prop).
    """
    results = []
    n_skipped = 0
    unknown_mods: set = set()

    count = 0
    for spectrum in spectra:
        count += 1
        if count % 10000 == 0:
            print(f"  {count} spectra processed ...", file=sys.stderr)

        params = spectrum["params"]
        scan = str(params.get("scans", params.get("scan", f"idx_{count}")))
        filename = str(params.get("filename", ""))
        seq = params.get("seq", "")
        charge_raw = params.get("charge", [2])
        charge = (
            int(charge_raw[0]) if isinstance(charge_raw, list) else int(charge_raw)
        )

        if not seq:
            n_skipped += 1
            continue

        parsed = parse_proforma(seq)
        if parsed is None:
            for mod in re.findall(r"\[([^\]]+)\]", seq):
                if mod not in MOD_MASS:
                    unknown_mods.add(mod)
            n_skipped += 1
            continue

        n_term_mod, res_masses = parsed
        theo = theoretical_mzs(n_term_mod, res_masses, charge)
        obs_mz = spectrum["m/z array"]
        obs_int = spectrum["intensity array"]
        n_peaks = len(obs_mz)
        n_matched, prop = matched_proportion(
            obs_mz, obs_int, theo, tolerance, tolerance_unit
        )
        results.append((scan, filename, seq, charge, n_peaks, n_matched, prop))

    return results, n_skipped, unknown_mods


# ---------------------------------------------------------------------------
# Fragment coverage — main function
# ---------------------------------------------------------------------------


def fragment_coverage(
    mgf_file,
    tolerance=10.0,
    tolerance_unit="ppm",
    output_tsv="fragment_coverage.tsv",
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
    output_plot : str
        Output histogram path (default: fragment_coverage.png).
    """
    if tolerance_unit not in ("ppm", "Da"):
        raise ValueError(
            f"tolerance_unit must be 'ppm' or 'Da', got '{tolerance_unit}'"
        )

    with mgf.MGF(mgf_file) as reader:
        results, n_skipped, unknown_mods = _compute_coverage_results(
            reader, tolerance, tolerance_unit
        )

    count = len(results) + n_skipped
    print(f"Processed {count} spectra total.", file=sys.stderr)
    print(f"  {len(results)} scored, {n_skipped} skipped.", file=sys.stderr)
    if unknown_mods:
        print(
            f"  Unknown modifications (spectra skipped): {unknown_mods}",
            file=sys.stderr,
        )

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
            unknown_mods: set = set()
            if n_with_seq > 0:
                print("Computing fragment ion coverage ...", file=sys.stderr)
                coverage_results_list, _, unknown_mods = _compute_coverage_results(
                    spectra, tolerance, tolerance_unit
                )
                print(
                    f"  {len(coverage_results_list):,} spectra scored.",
                    file=sys.stderr,
                )
                if unknown_mods:
                    print(
                        f"  Unknown modifications: {unknown_mods}",
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

            peaks_counts_map: dict[int, int] = {}
            for n in peak_counts_list:
                peaks_counts_map[n] = peaks_counts_map.get(n, 0) + 1
            peaks_tsv = os.path.join(output_root, "peak_counts.tsv")
            _write_tsv(
                peaks_tsv,
                ["n_peaks", "count"],
                [(n, peaks_counts_map[n]) for n in sorted(peaks_counts_map)],
            )

            lengths_png: str | None = None
            if lengths:
                lengths_counts_map: dict[int, int] = {}
                for ln in lengths:
                    lengths_counts_map[ln] = lengths_counts_map.get(ln, 0) + 1
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
