import random
from os import PathLike
from typing import Iterable

import fire
import pyteomics.mgf
import tqdm

from . import PyteomicsSpectrum
from .downsample import downsample_spectra


class MgfUtils:
    """
    Utilities for working with one or more MGF files.

    Parameters
    ----------
    *mgf_files : Iterable[PathLike]
        One or more paths to MGF files. These are streamed using
        ``pyteomics.mgf.chain``.
    random_seed : int, default=42
        Seed used for reproducible shuffling/sampling.
    """

    def __init__(self, *mgf_files: Iterable[PathLike], random_seed: int = 42) -> None:
        random.seed(random_seed)
        self.spectra: Iterable[PyteomicsSpectrum] = pyteomics.mgf.chain(
            *mgf_files, use_index=False
        )

    def shuffle(self) -> None:
        """
        Shuffle spectra in memory.

        This method consumes the current ``self.spectra`` iterable, loads all
        spectra into a list, shuffles them in-place, and replaces
        ``self.spectra`` with the shuffled list.
        """
        spectra_iter = tqdm.tqdm(self.spectra, unit="PSM", desc="Reading Spectra")
        spectra_iter = list(spectra_iter)
        spectra_iter = random.shuffle(spectra_iter)
        self.spectra = spectra_iter

    def downsample(self, k: int = 1) -> None:
        """
        Downsample spectra by peptide sequence.

        Delegates to :func:`downsample_spectra`, which groups spectra by peptide
        sequence (typically ``spectrum["params"]["seq"]``) and retains up to
        ``k`` spectra per unique peptide.

        Parameters
        ----------
        k : int, default=1
            Maximum number of spectra to retain per peptide sequence.

        Notes
        -----
        - Depending on the implementation of :func:`downsample_spectra`, this
          may require grouping all spectra and can be memory-intensive.
        """
        self.spectra = downsample_spectra(self.spectra, k, False, None)

    def write(self, outfile: PathLike = "out.mgf") -> None:
        """
        Write current spectra to an MGF file.

        Parameters
        ----------
        outfile : PathLike, default="out.mgf"
            Output file path for the written MGF.
        """
        spectra_iter = tqdm.tqdm(self.spectra, unit="PSM", desc="Writing Spectra")
        pyteomics.mgf.write(spectra=spectra_iter, output=outfile)


def main() -> None:
    """CLI Entry"""
    fire.Fire(MgfUtils)


if __name__ == "__main__":
    main()
