"""
Filter an annotated MGF file using the same criteria applied by Casanovo.

Removes spectra that would be silently skipped or crash Casanovo during
training:

* Missing or empty ``SEQ`` field.
* Sequence contains tokens not present in the Casanovo residue vocabulary.
* Precursor charge is missing, ambiguous, or exceeds ``max_charge``.
* Spectrum has fewer than ``min_peaks`` peaks (before any Casanovo
  peak-intensity filtering).

Outputs a filtered MGF and a log file with per-criterion counts.
"""

import logging
import pathlib
from os import PathLike
from typing import Optional

import pyteomics.mgf
import tqdm
import yaml
from pyteomics import proforma as pf

from . import configure_logging
from .types import Commands, PyteomicsSpectrum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: PathLike) -> dict:
    """Load a Casanovo YAML config and return it as a dict."""
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def _seq_to_tokens(seq: str) -> list[str]:
    """
    Split a ProForma sequence into tokens using the same format as
    depthcharge's ``PeptideTokenizer.split()``.

    Named modifications produce tokens like ``C[Carbamidomethyl]`` and
    ``[Acetyl]-``; mass modifications produce tokens like
    ``K[+229.163000]`` and ``[+271.174000]-``.

    Returns a list of token strings, or raises ``ValueError`` if the
    ProForma string cannot be parsed.
    """
    residues, meta = pf.parse(seq)

    def _mod_str(mods: list) -> str:
        """Format a list of pyteomics modification objects as ``[...]``."""
        if len(mods) == 1:
            try:
                return f"[{mods[0].name}]"
            except (AttributeError, ValueError):
                return f"[{mods[0].mass:+0.6f}]"
        # Multiple mods: sum masses.
        total = sum(m.mass for m in mods)
        return f"[{total:+0.6f}]"

    tokens: list[str] = []

    # N-terminal modification.
    n_term = meta.get("n_term")
    if n_term:
        tokens.append(f"{_mod_str(n_term)}-")

    # Residues.
    for aa, mods in residues:
        if mods:
            tokens.append(f"{aa}{_mod_str(mods)}")
        else:
            tokens.append(aa)

    # C-terminal modification.
    c_term = meta.get("c_term")
    if c_term:
        tokens.append(f"-{_mod_str(c_term)}")

    return tokens


def _parse_charge(charge_raw) -> Optional[int]:
    """
    Return the charge as a positive int, or ``None`` if it is missing,
    ambiguous, zero, or cannot be converted.
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
# Core command
# ---------------------------------------------------------------------------

def filter_spectra(
    mgf_file: PathLike,
    config: PathLike,
    output_root: str = "filtered",
    output_dir: PathLike = ".",
) -> None:
    """
    Filter an annotated MGF file using Casanovo's spectrum acceptance criteria.

    Reads *mgf_file*, removes spectra that Casanovo would reject, and writes
    the survivors to ``<output_dir>/<output_root>.mgf``.  A summary of how
    many spectra were removed by each criterion is written to
    ``<output_dir>/<output_root>.log``.

    Parameters
    ----------
    mgf_file : PathLike
        Input annotated MGF file.
    config : PathLike
        Casanovo YAML configuration file.  The fields ``residues``,
        ``replace_isoleucine_with_leucine``, ``min_peaks``, and
        ``max_charge`` are read from this file.
    output_root : str, optional
        Stem used for output file names (default ``"filtered"``).
    output_dir : PathLike, optional
        Directory in which to write output files (default: current directory).
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mgf_out = output_dir / f"{output_root}.mgf"
    log_out = output_dir / f"{output_root}.log"

    configure_logging(log_out)

    cfg = _load_config(config)
    min_peaks: int = cfg.get("min_peaks", 20)
    max_charge: int = cfg.get("max_charge", 10)
    replace_il: bool = cfg.get("replace_isoleucine_with_leucine", False)

    # Build the valid-token set from the config residues, mirroring
    # PeptideTokenizer: always include the 20 standard amino acids, then
    # add (or override with) whatever is in the config.
    _STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")
    residues: dict = cfg.get("residues", {})
    valid_tokens: set[str] = _STANDARD_AAS | set(residues.keys())
    if replace_il and "I" in valid_tokens:
        valid_tokens.discard("I")

    logger.info("Input MGF  : %s", mgf_file)
    logger.info("Config     : %s", config)
    logger.info("min_peaks  : %d", min_peaks)
    logger.info("max_charge : %d", max_charge)
    logger.info("Output MGF : %s", mgf_out)

    n_total = 0
    n_no_seq = 0
    n_bad_seq = 0
    n_bad_charge = 0
    n_few_peaks = 0

    passing: list[PyteomicsSpectrum] = []

    for spectrum in tqdm.tqdm(
        pyteomics.mgf.read(str(mgf_file), use_index=False),
        desc=f"Filtering {mgf_file}",
        unit="psm",
    ):
        n_total += 1

        # --- 1. Missing or empty SEQ ----------------------------------------
        seq = spectrum["params"].get("seq", "")
        if not seq:
            n_no_seq += 1
            continue

        # --- 2. Unknown tokens in sequence ----------------------------------
        try:
            tokens = _seq_to_tokens(seq)
        except Exception:
            n_bad_seq += 1
            continue
        if any(t not in valid_tokens for t in tokens):
            n_bad_seq += 1
            continue

        # --- 3. Invalid charge ----------------------------------------------
        charge = _parse_charge(spectrum["params"].get("charge"))
        if charge is None or charge <= 0 or charge > max_charge:
            n_bad_charge += 1
            continue

        # --- 4. Too few peaks -----------------------------------------------
        if len(spectrum.get("m/z array", [])) < min_peaks:
            n_few_peaks += 1
            continue

        passing.append(spectrum)

    n_pass = len(passing)
    n_filtered = n_total - n_pass

    logger.info("---")
    logger.info("Total spectra read        : %d", n_total)
    logger.info("Filtered: missing SEQ     : %d", n_no_seq)
    logger.info("Filtered: invalid tokens  : %d", n_bad_seq)
    logger.info("Filtered: invalid charge  : %d", n_bad_charge)
    logger.info("Filtered: too few peaks   : %d", n_few_peaks)
    logger.info("Filtered: total           : %d", n_filtered)
    logger.info("Passing spectra written   : %d", n_pass)

    pyteomics.mgf.write(
        tqdm.tqdm(passing, desc=f"Writing {mgf_out}", unit="psm"),
        output=str(mgf_out),
    )


COMMANDS = filter_spectra
