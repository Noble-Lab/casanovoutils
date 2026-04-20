"""
Utilities for reading and sampling spectra from mzML files.

Provides a single ``sample_spectra`` command that makes one streaming pass
through an mzML file, sampling a proportion ``k`` of spectra from each
read buffer.  Output is written to MGF format.  A ``main`` entry point
exposes commands as CLI subcommands via ``fire``.
"""

import logging
import pathlib
import random
from os import PathLike
from typing import Optional

import fire
import numpy as np
import pyteomics.mgf
import pyteomics.mzml
import tqdm

from . import configure_logging
from .types import Commands, PyteomicsSpectrum


def _to_mgf_spectrum(s: PyteomicsSpectrum) -> dict:
    """Convert a pyteomics mzML spectrum dict to pyteomics MGF spectrum format."""
    params: dict = {"TITLE": s.get("id", "")}

    prec_list = s.get("precursorList", {})
    if prec_list.get("count", 0) > 0:
        precursor = prec_list["precursor"][0]
        sel_ions = precursor.get("selectedIonList", {})
        if sel_ions.get("count", 0) > 0:
            ion = sel_ions["selectedIon"][0]
            pepmass = ion.get("selected ion m/z")
            if pepmass is not None:
                params["PEPMASS"] = pepmass
            charge = ion.get("charge state")
            if charge is not None:
                params["CHARGE"] = f"{int(charge)}+"

    scan_list = s.get("scanList", {})
    if scan_list.get("count", 0) > 0:
        scan = scan_list["scan"][0]
        rt = scan.get("scan start time")
        if rt is not None:
            params["RTINSECONDS"] = rt

    return {
        "params": params,
        "m/z array": s["m/z array"],
        "intensity array": s["intensity array"],
    }


def _write_spectra(spectra: list[PyteomicsSpectrum], path: PathLike) -> None:
    """Write spectra to MGF."""
    suffix = pathlib.Path(path).suffix.lower()
    if suffix == ".mgf":
        mgf_spectra = [_to_mgf_spectrum(s) for s in spectra]
        out_iter = tqdm.tqdm(mgf_spectra, desc=f"Writing {path}", unit="spectrum")
        pyteomics.mgf.write(out_iter, output=str(path))
    else:
        raise ValueError(
            f"Unsupported output extension {suffix!r}; expected '.mgf'."
        )


def sample_mzml(
    input_file: PathLike,
    k: float,
    buffer_size: int = 1000,
    random_seed: int = 42,
) -> list[PyteomicsSpectrum]:
    """
    Sample a proportion of spectra from an mzML file in a single streaming pass.

    Reads the file in chunks of ``buffer_size`` spectra.  From each chunk,
    ``round(k * chunk_size)`` spectra are drawn without replacement using
    ``random.sample``.  No second pass or total-count is required.

    Note: the final sample count equals ``sum(round(k * b) for b in buffers)``
    which may differ slightly from ``round(k * total)`` due to per-buffer
    rounding.  Use a ``buffer_size`` that is large relative to ``1 / k`` to
    minimise this effect.

    Parameters
    ----------
    input_file : PathLike
        Path to the input mzML file.
    k : float
        Proportion of spectra to sample; must be in (0, 1).
    buffer_size : int, default=1000
        Number of spectra read per I/O chunk.
    random_seed : int, default=42
        Seed for reproducible sampling.

    Returns
    -------
    list[PyteomicsSpectrum]
        Sampled spectra in file order within each buffer.
    """
    if not isinstance(k, float) or not (0.0 < k < 1.0):
        raise ValueError(f"k must be a float proportion in (0, 1), got {k!r}.")

    rng = random.Random(random_seed)
    result: list[PyteomicsSpectrum] = []

    with pyteomics.mzml.MzML(str(input_file)) as reader:
        buf: list[PyteomicsSpectrum] = []
        for spectrum in tqdm.tqdm(reader, desc="Sampling spectra", unit="spectrum"):
            buf.append(spectrum)
            if len(buf) >= buffer_size:
                n_sample = round(k * len(buf))
                result.extend(rng.sample(buf, n_sample))
                buf.clear()
        if buf:
            n_sample = round(k * len(buf))
            result.extend(rng.sample(buf, n_sample))

    logging.info("Sampled %d spectra", len(result))
    return result


def sample_spectra(
    input_file: PathLike,
    k: float,
    outfile: PathLike,
    buffer_size: int = 1000,
    random_seed: int = 42,
) -> None:
    """
    Sample spectra from an mzML file and write them to an MGF file.
    
    Note: writing to mzML is a pain so I didn't implement it here. If you need
    an mzML output I would recommend running this and then using msconvert
    to convert to mzML.

    Parameters
    ----------
    input_file : PathLike
        Path to the input mzML file.
    k : float
        Proportion of spectra to sample; must be in (0, 1).
    outfile : PathLike
        Output path; must have a ``.mgf`` extension.
    buffer_size : int, default=1000
        Number of spectra read per I/O chunk.
    random_seed : int, default=42
        Seed for reproducible sampling.
    """
    configure_logging(pathlib.Path(outfile).with_suffix(".log"))
    result = sample_mzml(input_file, k, buffer_size=buffer_size, random_seed=random_seed)
    _write_spectra(result, outfile)


COMMANDS: Commands = sample_spectra


def main() -> None:
    fire.Fire(COMMANDS)


if __name__ == "__main__":
    main()
