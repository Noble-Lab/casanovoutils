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
from typing import Generator, Optional

import pyteomics.mgf
import tqdm
import yaml
from depthcharge.tokenizers import PeptideTokenizer

from . import configure_logging
from .types import PyteomicsSpectrum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(config_path: PathLike) -> dict:
    """Load a Casanovo YAML config and return it as a dict."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(
            f"Expected a YAML mapping in {config_path!r}, got {type(cfg).__name__}."
        )
    return cfg


def _make_tokenizer(cfg: dict) -> PeptideTokenizer:
    """Build a PeptideTokenizer from the residues in a Casanovo config."""
    return PeptideTokenizer(
        residues=cfg.get("residues", {}),
        replace_isoleucine_with_leucine=cfg.get(
            "replace_isoleucine_with_leucine", False
        ),
    )


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
    overwrite: bool = False,
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
    overwrite : bool, optional
        If False (default), raise ``FileExistsError`` if any output file
        already exists.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mgf_out = output_dir / f"{output_root}.mgf"
    log_out = output_dir / f"{output_root}.log"

    if not overwrite:
        existing = [p for p in (mgf_out, log_out) if p.exists()]
        if existing:
            paths = ", ".join(str(p) for p in existing)
            raise FileExistsError(
                f"Output file(s) already exist: {paths}. "
                f"Use --overwrite to overwrite."
            )

    # Always attach a file handler for log_out, even if root handlers exist
    # from a prior configure_logging call.
    configure_logging(log_out)
    if not any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(log_out.resolve())
        for h in logging.root.handlers
    ):
        file_handler = logging.FileHandler(log_out)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logging.root.addHandler(file_handler)

    cfg = _load_config(config)
    min_peaks: int = cfg.get("min_peaks", 20)
    max_charge: int = cfg.get("max_charge", 10)
    tokenizer = _make_tokenizer(cfg)

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
    n_pass = 0

    def _filtered_spectra() -> Generator[PyteomicsSpectrum, None, None]:
        nonlocal n_total, n_no_seq, n_bad_seq, n_bad_charge, n_few_peaks, n_pass

        for spectrum in tqdm.tqdm(
            pyteomics.mgf.read(str(mgf_file), use_index=False),
            desc=f"Filtering {mgf_file}",
            unit="psm",
        ):
            n_total += 1

            # --- 1. Missing or empty SEQ ------------------------------------
            seq = spectrum["params"].get("seq", "")
            if not seq:
                n_no_seq += 1
                continue

            # --- 2. Unknown tokens in sequence ------------------------------
            try:
                tokenizer.tokenize(seq)
            except ValueError:
                n_bad_seq += 1
                continue

            # --- 3. Invalid charge ------------------------------------------
            charge = _parse_charge(spectrum["params"].get("charge"))
            if charge is None or charge <= 0 or charge > max_charge:
                n_bad_charge += 1
                continue

            # --- 4. Too few peaks -------------------------------------------
            if len(spectrum.get("m/z array", [])) < min_peaks:
                n_few_peaks += 1
                continue

            n_pass += 1
            yield spectrum

    pyteomics.mgf.write(
        tqdm.tqdm(_filtered_spectra(), desc=f"Writing {mgf_out}", unit="psm"),
        output=str(mgf_out),
    )

    n_filtered = n_total - n_pass
    logger.info("---")
    logger.info("Total spectra read        : %d", n_total)
    logger.info("Filtered: missing SEQ     : %d", n_no_seq)
    logger.info("Filtered: invalid tokens  : %d", n_bad_seq)
    logger.info("Filtered: invalid charge  : %d", n_bad_charge)
    logger.info("Filtered: too few peaks   : %d", n_few_peaks)
    logger.info("Filtered: total           : %d", n_filtered)
    logger.info("Passing spectra written   : %d", n_pass)


COMMANDS = filter_spectra
