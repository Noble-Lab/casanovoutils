import itertools
import random
from os import PathLike
from typing import Iterable, Optional

import fire
import pyteomics.mgf
import tqdm

from . import PyteomicsSpectrum, get_pep_dict_mgf


def downsample_spectra(
    spectra: PathLike | Iterable[PyteomicsSpectrum],
    k: int,
    shuffle: bool,
    random_seed: Optional[int],
) -> Iterable[PyteomicsSpectrum]:
    """
    Downsample spectra by limiting the number of PSMs per peptide sequence.

    Spectra are first grouped by peptide sequence (via ``get_pep_dict_mgf``),
    then up to ``k`` spectra are randomly sampled for each unique peptide.
    The resulting spectra can optionally be shuffled.

    Parameters
    ----------
    spectra : PathLike or Iterable[PyteomicsSpectrum]
        Either:
        - A path to an MGF file, or
        - An iterable of Pyteomics spectrum dictionaries.
        In both cases, spectra must contain the peptide sequence in
        ``spectrum["params"]["seq"]``.

    k : int
        Maximum number of spectra (PSMs) to retain per unique peptide sequence.

    shuffle : bool
        If True, shuffle the final pooled list of sampled spectra.

    random_seed : int
        Random seed used for reproducible sampling and shuffling.

    Returns
    -------
    Iterable[PyteomicsSpectrum]
        A list-like iterable of downsampled spectrum dictionaries.
        Each peptide sequence will appear at most ``k`` times.
    """
    if random_seed is not None:
        random.seed(random_seed)

    pep_dict = get_pep_dict_mgf(spectra)
    for pep, psms in tqdm.tqdm(
        pep_dict.items(), desc="Sampling peptides", unit="peptide"
    ):
        pep_dict[pep] = random.sample(psms, min(len(psms), k))

    spectra = list(itertools.chain.from_iterable(pep_dict.values()))
    if shuffle:
        random.shuffle(spectra)

    return spectra


def downsample_mgf_pep(
    spectra: PathLike | Iterable[PyteomicsSpectrum],
    outfile: PathLike = "out.mgf",
    k: int = 1,
    shuffle: bool = False,
    random_seed: int = 42,
) -> None:
    """
    Downsample spectra by peptide and write the result to an MGF file.

    This is a convenience wrapper around ``downsample_spectra`` that also
    writes the resulting spectra to disk in MGF format.

    Parameters
    ----------
    spectra : PathLike or Iterable[PyteomicsSpectrum]
        Either a path to an input MGF file or an iterable of Pyteomics spectrum
        dictionaries to be downsampled.

    outfile : PathLike, default="out.mgf"
        Output path for the downsampled MGF file.

    k : int, default=1
        Maximum number of spectra (PSMs) to retain per peptide sequence.

    shuffle : bool, default=False
        Whether to shuffle the sampled spectra before writing.

    random_seed : int, default=42
        Random seed for reproducible sampling and shuffling.

    Returns
    -------
    None
        Writes the downsampled spectra to ``outfile``.
    """
    spectra = downsample_spectra(spectra, k, shuffle, random_seed)
    spectra = tqdm.tqdm(spectra, desc=f"Writing output file {outfile}", unit="psm")
    pyteomics.mgf.write(spectra, output=outfile)


def main() -> None:
    """CLI entry"""
    fire.Fire(downsample_mgf_pep)


if __name__ == "__main__":
    main()
