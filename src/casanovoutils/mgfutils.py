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
import re
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
    miniters: int = 1,
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
    miniters : int, default=1
        Minimum number of iterations between progress bar updates.

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
        raw = tqdm.tqdm(raw, desc=desc, unit="psm", miniters=miniters)

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


def _iter_raw_blocks(paths: Iterable[PathLike]) -> Iterable[str]:
    """Yield raw MGF text blocks from one or more file paths without parsing."""
    for path in paths:
        with open(path) as f:
            block: list[str] = []
            for line in f:
                if line.strip() == "BEGIN IONS":
                    block = [line]
                elif line.strip() == "END IONS":
                    block.append(line)
                    yield "".join(block)
                    block = []
                elif block:
                    block.append(line)


def shuffle(
    spectra: SpectraInput,
    outfile: Optional[PathLike] = None,
    random_seed: int = 42,
) -> list[PyteomicsSpectrum]:
    """
    Read all spectra and return them in a shuffled order.

    When *spectra* is a file path (or iterable of file paths) and *outfile* is
    provided, a fast raw-text path is used: entries are shuffled as opaque
    strings without parsing peaks, which is significantly faster than the
    pyteomics parse/serialize round-trip.

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
        All spectra in shuffled order, or an empty list when the fast raw-text
        path is used (output was written directly to *outfile*).
    """
    configure_logging(pathlib.Path(outfile).with_suffix(".log") if outfile else None)

    logging.info("Shuffling spectra (random_seed=%d)", random_seed)
    random.seed(random_seed)

    # Fast path: avoid pyteomics parse/serialize when input is file path(s).
    if outfile is not None:
        if isinstance(spectra, (str, os.PathLike)):
            paths: list[PathLike] = [spectra]
            use_raw = True
        else:
            it = iter(spectra)
            try:
                first = next(it)
            except StopIteration:
                return []
            if isinstance(first, (str, os.PathLike)):
                paths = list(itertools.chain([first], it))
                use_raw = True
            else:
                spectra = itertools.chain([first], it)
                use_raw = False

        if use_raw:
            blocks = list(
                tqdm.tqdm(_iter_raw_blocks(paths), desc="Reading spectra", unit="psm")
            )
            random.shuffle(blocks)
            logging.info("Shuffled %d spectra", len(blocks))
            with open(outfile, "w") as f:
                f.writelines(tqdm.tqdm(blocks, desc=f"Writing {outfile}", unit="psm"))
            return []

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


_VALID_DOWNSAMPLE_TYPES = frozenset({"number", "proportion"})


def _group_key(spectrum, precursor=False, ignore_mods=False):
    """Return the grouping key for reservoir sampling.

    Parameters
    ----------
    spectrum : PyteomicsSpectrum
        A single spectrum dict.
    precursor : bool
        If True, include the charge state in the key so that the same peptide
        in different charge states is treated as distinct groups.
    ignore_mods : bool
        If True, strip ProForma bracketed modification annotations from the
        sequence before forming the key.
    """
    seq = spectrum["params"]["seq"]
    if ignore_mods:
        seq = re.sub(r"\[.*?\]", "", seq).strip("-")
    if precursor:
        charge = spectrum["params"].get("charge", "")
        # pyteomics returns ChargeList (a list subclass) for the charge field,
        # which is unhashable. Convert to a canonical string for use as a key.
        charge = str(charge)
        return (seq, charge)
    return seq


def spectra_per_peptide(
    spectra: SpectraInput,
    outfile: Optional[PathLike] = None,
    k: int = 1,
    precursor: bool = False,
    ignore_mods: bool = False,
    random_seed: int = 42,
) -> list[PyteomicsSpectrum]:
    """
    Sample up to k spectra per peptide using reservoir sampling.

    Makes a single streaming pass through *spectra*, maintaining a reservoir
    of size k per unique group.  For the j-th occurrence of a group: if
    j <= k, add unconditionally; if j > k, replace a uniformly random
    reservoir slot with probability k/j.  Memory usage is
    O(unique groups x k) rather than O(total spectra).

    Parameters
    ----------
    spectra : PathLike, Iterable[PathLike], or Iterable[PyteomicsSpectrum]
        Spectrum source — see :func:`iter_spectra` for accepted types.
    outfile : PathLike, optional
        If provided, write the sampled spectra to this MGF file path.
    k : int, default=1
        Maximum number of spectra to retain per group.
    precursor : bool, default=False
        If True, group by peptide sequence *and* charge state, so that the
        same peptide observed in different charge states is treated as
        separate groups.
    ignore_mods : bool, default=False
        If True, strip ProForma bracketed modification annotations (e.g.
        ``[Acetyl]``, ``[Carbamidomethyl]``) from the sequence before
        grouping, so modified and unmodified forms of the same peptide are
        counted together.
    random_seed : int, default=42
        Seed for the local random number generator.

    Returns
    -------
    list[PyteomicsSpectrum]
        Sampled spectra, grouped by key in first-seen order.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"--k must be a positive integer, got {k!r}.")
    configure_logging(pathlib.Path(outfile).with_suffix(".log") if outfile else None)
    logging.info(
        "Sampling up to k=%d spectra per %s (precursor=%s, ignore_mods=%s, "
        "random_seed=%d)",
        k,
        "precursor" if precursor else "peptide",
        precursor,
        ignore_mods,
        random_seed,
    )

    rng = random.Random(random_seed)
    reservoir: dict = {}
    counts: dict = {}

    for spectrum in iter_spectra(spectra, desc="Streaming spectra", miniters=100_000):
        key = _group_key(spectrum, precursor=precursor, ignore_mods=ignore_mods)
        count = counts.get(key, 0) + 1
        counts[key] = count
        if count <= k:
            reservoir.setdefault(key, []).append(spectrum)
        else:
            j = rng.randint(0, count - 1)
            if j < k:
                reservoir[key][j] = spectrum

    result = list(itertools.chain.from_iterable(reservoir.values()))
    logging.info(
        "Retained %d spectra from %d unique groups", len(result), len(reservoir)
    )
    write_spectra(result, outfile)
    return result


def downsample_spectra(
    input_file: PathLike,
    output_file: PathLike,
    downsample_type: str = "number",
    downsample_rate: float = 100,
    random_seed: int = 42,
) -> None:
    """
    Downsample an MGF file to a target number or proportion of spectra.

    Makes two streaming passes: the first counts total spectra, the second
    streams with an adaptive acceptance probability (needed/remaining) that
    guarantees exactly k spectra are written.

    Parameters
    ----------
    input_file : PathLike
        Path to the input MGF file.
    output_file : PathLike
        Path for the downsampled output MGF file.  Must differ from
        *input_file*.
    downsample_type : str, default ``"number"``
        One of ``"number"`` (retain exactly *downsample_rate* spectra) or
        ``"proportion"`` (retain exactly ``round(total × downsample_rate)``).
    downsample_rate : float, default 100
        Target rate.  Positive integer for ``"number"``; in ``(0, 1]`` for
        ``"proportion"``.
    random_seed : int, default 42
        Seed for the random number generator.
    """
    configure_logging(pathlib.Path(output_file).with_suffix(".log"))

    if pathlib.Path(input_file).resolve() == pathlib.Path(output_file).resolve():
        raise ValueError(
            "input_file and output_file must be different paths; "
            "overwriting the input in-place is not supported."
        )

    if downsample_type not in _VALID_DOWNSAMPLE_TYPES:
        raise ValueError(
            f"--downsample_type must be one of {sorted(_VALID_DOWNSAMPLE_TYPES)}, "
            f"got {downsample_type!r}."
        )

    if downsample_type == "number":
        if (
            not np.isfinite(downsample_rate)
            or downsample_rate != int(downsample_rate)
            or int(downsample_rate) < 1
        ):
            raise ValueError(
                "--downsample_rate must be a positive integer when "
                f"--downsample_type is 'number', got {downsample_rate!r}."
            )
    else:
        if not (0 < downsample_rate <= 1):
            raise ValueError(
                "--downsample_rate must be in (0, 1] when "
                f"--downsample_type is '{downsample_type}', "
                f"got {downsample_rate!r}."
            )

    rng = random.Random(random_seed)

    # First pass: count total spectra.
    with pyteomics.mgf.read(str(input_file), use_index=False) as reader:
        n = sum(1 for _ in tqdm.tqdm(reader, desc="Counting spectra", unit="spectrum"))

    if downsample_type == "number":
        k = min(int(downsample_rate), n)
    else:
        k = min(round(n * downsample_rate), n)

    pct = k / n if n > 0 else 0.0
    logging.info("Targeting %d of %d spectra (%.1f%%)", k, n, 100 * pct)

    # Second pass: stream with adaptive acceptance probability.
    needed = k
    remaining = n

    def _filtered():
        nonlocal needed, remaining
        with pyteomics.mgf.read(str(input_file), use_index=False) as reader:
            for spectrum in tqdm.tqdm(
                reader, desc="Streaming spectra", unit="spectrum"
            ):
                if needed > 0 and rng.random() < needed / remaining:
                    needed -= 1
                    yield spectrum
                remaining -= 1

    pyteomics.mgf.write(_filtered(), output=str(output_file))
    logging.info("Done writing %s", output_file)


COMMANDS: Commands = {
    "pipeline": pipeline,
    "shuffle": shuffle,
    "downsample": downsample,
    "spectra-per-peptide": spectra_per_peptide,
    "downsample-spectra": downsample_spectra,
    "purge-redundant": purge_redundant,
}


def main() -> None:
    fire.Fire(COMMANDS, serialize=lambda _: "")


if __name__ == "__main__":
    main()
