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

Requires: pyteomics, numpy, matplotlib
"""

import csv
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
    charges = sorted(counts)
    vals = [counts[c] for c in charges]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(c) for c in charges], vals, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Charge state")
    ax.set_ylabel("Number of spectra")
    ax.set_title(f"Charge state distribution (n={total:,})")
    fig.tight_layout()
    fig.savefig(output_plot, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_plot}", file=sys.stderr)

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

    results = []
    n_skipped = 0
    unknown_mods = set()

    count = 0
    for spectrum in mgf.MGF(mgf_file):
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
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(proportions, bins=50, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Proportion of intensity matched by b/y ions")
    ax.set_ylabel("Number of spectra")
    ax.set_title(
        f"Fragment ion coverage  (n={len(proportions):,}, "
        f"median={np.median(proportions):.3f})"
    )
    fig.tight_layout()
    fig.savefig(output_plot, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_plot}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def charge_distribution_main() -> None:
    fire.Fire(charge_distribution)


def fragment_coverage_main() -> None:
    fire.Fire(fragment_coverage)
