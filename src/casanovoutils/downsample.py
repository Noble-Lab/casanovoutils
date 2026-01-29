import random
import itertools
from os import PathLike

import fire
import tqdm
import pyteomics.mgf

from . import get_pep_dict_mgf


def downsample_mgf_pep(
    infile: PathLike,
    outfile: PathLike = "out.mgf",
    k: int = 1,
    shuffle: bool = False,
    random_seed: int = 42,
) -> None:
    """
    Downsample an MGF file by sampling up to ``k`` PSMs per unique peptide
    sequence, optionally shuffling, and writing the resulting spectra to an
    output MGF file.

    Parameters
    ----------
    infile : PathLike
        One or more input MGF file paths. All spectra from all files are
        aggregated before downsampling.
    outfile : PathLike, default="out.mgf"
        Output MGF file path. Defaults to "out.mgf".
    k : int, default=1
        Maximum number of PSMs to sample per peptide sequence.
    shuffle : bool, default=False
        Whether to shuffle the pooled sample of spectra prior to writing.

    Returns
    -------
    None
    """
    random.seed(random_seed)
    pep_dict = get_pep_dict_mgf(infile)

    for pep, psms in tqdm.tqdm(
        pep_dict.items(), desc="Sampling peptides", unit="peptide"
    ):
        pep_dict[pep] = random.sample(psms, min(len(psms), k))

    spectra = list(itertools.chain.from_iterable(pep_dict.values()))
    if shuffle:
        random.shuffle(spectra)

    pyteomics.mgf.write(
        tqdm.tqdm(spectra, desc=f"Writing output file {outfile}", unit="psm")
    )


def main() -> None:
    """CLI entry"""
    fire.Fire(downsample_mgf_pep)


if __name__ == "__main__":
    main()
