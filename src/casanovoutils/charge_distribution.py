#!/usr/bin/env python
"""Compute the global charge state distribution from an MGF file.

Outputs:
  1. A TSV with columns: charge, count
  2. A bar chart (PNG) of the distribution.

Requires: pyteomics, matplotlib
"""

import csv
import sys
from os import PathLike
from typing import Iterable

import fire
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyteomics import mgf


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
        Number of spectra skipped due to multiple charge states.
    """
    counts = {}
    n_skipped = 0
    for spectrum in spectra:
        charge_raw = spectrum["params"].get("charge", [2])
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
    mgf_file : str
        Input MGF file.
    output_tsv : str
        Output TSV path (default: charge_distribution.tsv).
    output_plot : str
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


def main() -> None:
    fire.Fire(charge_distribution)


if __name__ == "__main__":
    main()
