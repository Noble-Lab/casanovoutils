"""
Utilities for reading, writing, and processing MGF spectrum files.

Provides functions to iterate over spectra from MGF files or in-memory
dicts, downsample by peptide, shuffle, and purge near-duplicate peaks.
A ``pipeline`` function chains these stages, and a ``main`` entry point
exposes them all as CLI subcommands via ``fire``.
"""

import itertools
import logging
import os
import pathlib
import random
from os import PathLike
from typing import Iterable, Optional

import fire
import numpy as np
import pyteomics.mgf
import tqdm

from . import configure_logging
from .types import Commands, PyteomicsSpectrum

SpectraInput = PathLike | Iterable[PathLike] | Iterable[PyteomicsSpectrum]


def iter_spectra(
    spectra: SpectraInput,
    desc: Optional[str] = None,
) -> Iterable[PyteomicsSpectrum]:
    """
    Normalize various spectrum input types to an iterable of PyteomicsSpectrum.

    Parameters
    ----------
    spectra : PathLike, Iterable[PathLike], or Iterable[PyteomicsSpectrum]
        One of:
        - A single path to an MGF file.
        - An iterable of paths to MGF files.
        - An iterable of Pyteomics spectrum dictionaries.
    desc : str, optional
        Description for the tqdm progress bar. If ``None``, no progress bar
        is shown.

    Yields
    ------
    PyteomicsSpectrum
        Spectrum dictionaries, one at a time.
    """
    if isinstance(spectra, (str, os.PathLike)):
        raw = pyteomics.mgf.read(spectra, use_index=False)
    else:
        it = iter(spectra)
        try:
            first = next(it)
        except StopIteration:
            return
        if isinstance(first, (str, os.PathLike)):
            raw = (
                s
                for path in itertools.chain([first], it)
                for s in pyteomics.mgf.read(path, use_index=False)
            )
        else:
            raw = itertools.chain([first], it)

    if desc is not None:
        raw = tqdm.tqdm(raw, desc=desc, unit="psm")

    yield from raw


def get_pep_dict_mgf(spectra: SpectraInput) -> dict[str, list[PyteomicsSpectrum]]:
    """
    Read spectra and group them by peptide sequence.

    Parameters
    ----------
    spectra : PathLike, Iterable[PathLike], or Iterable[PyteomicsSpectrum]
        Spectrum source — see :func:`iter_spectra` for accepted types.

    Returns
    -------
    dict[str, list[PyteomicsSpectrum]]
        A dictionary mapping peptide sequence strings (taken from
        ``spectrum["params"]["seq"]``) to a list of Pyteomics spectrum
        dictionaries corresponding to that sequence.
    """
    out = {}
    for curr in iter_spectra(spectra, desc="Reading mgf file"):
        seq = curr["params"]["seq"]
        if seq not in out:
            out[seq] = []
        out[seq].append(curr)
    return out


def write_spectra(
    spectra: Iterable[PyteomicsSpectrum], outfile: Optional[PathLike]
) -> None:
    """
    Write spectra to an MGF file, if an output path is provided.

    Parameters
    ----------
    spectra : Iterable[PyteomicsSpectrum]
        Spectra to write.
    outfile : PathLike, optional
        Destination MGF file path. If ``None``, this function is a no-op.
    """
    if outfile is None:
        return

    out_iter = tqdm.tqdm(spectra, desc=f"Writing {outfile}", unit="psm")
    pyteomics.mgf.write(out_iter, output=str(outfile))


def downsample(
    spectra: SpectraInput,
    k: int = 1,
    outfile: Optional[PathLike] = None,
    random_seed: int = 42,
) -> list[PyteomicsSpectrum]:
    """
    Downsample spectra by limiting the number of PSMs per peptide sequence.

    Spectra are grouped by peptide sequence, then up to ``k`` spectra are
    randomly sampled for each unique peptide. If ``outfile`` is provided, the
    result is also written to disk in MGF format.

    Parameters
    ----------
    spectra : PathLike, Iterable[PathLike], or Iterable[PyteomicsSpectrum]
        Spectrum source — see :func:`iter_spectra` for accepted types.
    k : int, default=1
        Maximum number of spectra (PSMs) to retain per unique peptide sequence.
    outfile : PathLike, optional
        If provided, write the downsampled spectra to this MGF file path.
    random_seed : int, default=42
        Random seed for reproducible sampling.

    Returns
    -------
    list[PyteomicsSpectrum]
        Downsampled spectra; each peptide sequence appears at most ``k`` times.
    """
    configure_logging(pathlib.Path(outfile).with_suffix(".log") if outfile else None)

    logging.info("Downsampling to k=%d per peptide (random_seed=%d)", k, random_seed)
    random.seed(random_seed)

    pep_dict = get_pep_dict_mgf(spectra)
    n_before = sum(len(v) for v in pep_dict.values())
    for pep, psms in tqdm.tqdm(
        pep_dict.items(), desc="Sampling peptides", unit="peptide"
    ):
        pep_dict[pep] = random.sample(psms, min(len(psms), k))

    result = list(itertools.chain.from_iterable(pep_dict.values()))
    logging.info("Downsampled %d -> %d spectra", n_before, len(result))
    write_spectra(result, outfile)
    return result


def remove_redundant_peaks(
    spectrum: PyteomicsSpectrum, eps: float
) -> PyteomicsSpectrum:
    """
    Remove redundant peaks that are too close together along the m/z axis.

    Peaks are sorted by m/z. Any peak within ``eps`` of the preceding peak
    is discarded, keeping the first peak in each run of close peaks.

    Parameters
    ----------
    spectrum : PyteomicsSpectrum
        A spectrum dict as returned by ``pyteomics.mgf.read``, containing
        ``"m/z array"`` and ``"intensity array"`` keys.
    eps : float, optional
        Maximum m/z distance between two peaks to be considered redundant.
        Defaults to the 32-bit float machine epsilon
        (``numpy.finfo(numpy.float32).eps`` ≈ 1.19e-7).

    Returns
    -------
    PyteomicsSpectrum
        A new spectrum dict with ``"m/z array"`` and ``"intensity array"``
        replaced by the deduplicated arrays. All other keys are unchanged.
    """
    mz = np.asarray(spectrum["m/z array"])
    intensity = np.asarray(spectrum["intensity array"])

    order = np.argsort(mz)
    mz = mz[order]
    intensity = intensity[order]
    keep = np.concatenate([[True], np.diff(mz) >= eps])

    logging.debug(
        "Removed %d redundant peaks from spectrum %s",
        len(mz) - np.sum(keep),
        spectrum.get("title", "UNKNOWN SPECTRUM"),
    )

    return {**spectrum, "m/z array": mz[keep], "intensity array": intensity[keep]}


def purge_redundant(
    spectra: SpectraInput,
    epsilon: float = np.finfo(np.float32).eps,
    outfile: Optional[PathLike] = None,
) -> list[PyteomicsSpectrum]:
    """
    Remove peaks with near-duplicate m/z values from each spectrum.

    For each spectrum, peaks are sorted by m/z and any peak whose m/z differs
    from the previous peak by less than ``epsilon`` is discarded. If
    ``outfile`` is provided, the result is also written to disk in MGF format.

    Parameters
    ----------
    spectra : PathLike, Iterable[PathLike], or Iterable[PyteomicsSpectrum]
        Spectrum source — see :func:`iter_spectra` for accepted types.
    epsilon : float
        Minimum m/z separation (in daltons) required to keep a peak.
    outfile : PathLike, optional
        If provided, write the purged spectra to this MGF file path.

    Returns
    -------
    list[PyteomicsSpectrum]
        Spectra with redundant peaks removed and peaks sorted by m/z.
    """
    configure_logging(pathlib.Path(outfile).with_suffix(".log") if outfile else None)
    logging.info("Purging redundant peaks with epsilon=%g Da", epsilon)
    spectra = iter_spectra(spectra, desc="Purging redundant peaks")
    spectra = map(lambda s: remove_redundant_peaks(s, epsilon), spectra)
    spectra = list(spectra)
    write_spectra(spectra, outfile)

    return spectra


def shuffle(
    spectra: SpectraInput,
    outfile: Optional[PathLike] = None,
    random_seed: int = 42,
) -> list[PyteomicsSpectrum]:
    """
    Read all spectra and return them in a shuffled order.

    Parameters
    ----------
    spectra : PathLike, Iterable[PathLike], or Iterable[PyteomicsSpectrum]
        Spectrum source — see :func:`iter_spectra` for accepted types.
    outfile : PathLike, optional
        If provided, write the shuffled spectra to this MGF file path.
    random_seed : int, default=42
        Random seed for reproducible shuffling.

    Returns
    -------
    list[PyteomicsSpectrum]
        All spectra in shuffled order.
    """
    configure_logging(pathlib.Path(outfile).with_suffix(".log") if outfile else None)

    logging.info("Shuffling spectra (random_seed=%d)", random_seed)
    random.seed(random_seed)

    result = list(iter_spectra(spectra, desc="Reading spectra"))
    random.shuffle(result)
    logging.info("Shuffled %d spectra", len(result))
    write_spectra(result, outfile)
    return result


def pipeline(
    spectra: SpectraInput,
    outfile: Optional[PathLike] = None,
    do_shuffle: bool = True,
    downsample_k: Optional[int] = None,
    purge_epsilon: Optional[float] = None,
    random_seed: int = 42,
) -> list[PyteomicsSpectrum]:
    """
    Run spectra through an optional chain of processing stages.

    Stages are applied in order: shuffle → downsample → purge redundant peaks.
    Each stage is skipped when its enabling parameter is ``None`` (or
    ``False`` for ``do_shuffle``).

    Parameters
    ----------
    spectra : PathLike, Iterable[PathLike], or Iterable[PyteomicsSpectrum]
        Spectrum source — see :func:`iter_spectra` for accepted types.
    outfile : PathLike, optional
        If provided, write the final spectra to this MGF file path.
    do_shuffle : bool, default=True
        Whether to shuffle the spectra.
    downsample_k : int, optional
        If provided, downsample to at most this many PSMs per peptide sequence.
    purge_epsilon : float, optional
        If provided, remove peaks whose m/z differs from the previous peak by
        less than this value (in daltons).
    random_seed : int, default=42
        Random seed passed to shuffle and downsample.

    Returns
    -------
    list[PyteomicsSpectrum]
        Processed spectra.
    """
    configure_logging(pathlib.Path(outfile).with_suffix(".log") if outfile else None)

    stages = []
    if do_shuffle:
        stages.append("shuffle")
    if downsample_k is not None:
        stages.append(f"downsample(k={downsample_k})")
    if purge_epsilon is not None:
        stages.append(f"purge-redundant(epsilon={purge_epsilon})")
    logging.info(
        "Running pipeline stages: %s", " -> ".join(stages) if stages else "none"
    )

    result: SpectraInput = spectra

    if do_shuffle:
        result = shuffle(result, random_seed=random_seed)

    if downsample_k is not None:
        result = downsample(result, k=downsample_k, random_seed=random_seed)

    if purge_epsilon is not None:
        result = purge_redundant(result, epsilon=purge_epsilon)
    else:
        result = list(iter_spectra(result))

    write_spectra(result, outfile)
    return result


COMMANDS: Commands = {
    "pipeline": pipeline,
    "shuffle": shuffle,
    "downsample": downsample,
    "purge-redundant": purge_redundant,
}


def main() -> None:
    fire.Fire(COMMANDS)


if __name__ == "__main__":
    main()
