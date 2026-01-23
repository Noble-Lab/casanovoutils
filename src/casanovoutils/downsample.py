import dataclasses
import random
import itertools
from os import PathLike
from typing import Iterable

import fire
import tqdm
import pyteomics.mgf

from ..casanovoutils import get_pep_dict_mgf

@dataclasses.dataclass
class DownsampleMS:
    """
    Utilities for downsampling data from one or more PSM files.

    Parameters
    ----------
    random_seed : int | float, default=42
        Random seed used for reproducible downsampling.
    """

    random_seed: int | float = 42

    def downsample_mgf_pep(
        self,
        *infiles: Iterable[PathLike],
        outfile: PathLike = "out.mgf",
        k: int = 1,
        shuffle: bool = True,
    ) -> None:
        """
        Downsample MGF files by sampling up to ``k`` PSMs per unique peptide
        sequence, optionally shuffling, and writing the resulting spectra to an
        output MGF file.

        Parameters
        ----------
        *infiles : Iterable[PathLike]
            One or more input MGF file paths. All spectra from all files are
            aggregated before downsampling.
        outfile : PathLike, default="out.mgf"
            Output MGF file path. Defaults to "out.mgf".
        k : int, default=1
            Maximum number of PSMs to sample per peptide sequence.
        shuffle : bool, default=True
            Whether to shuffle the pooled sample of spectra prior to writing.

        Returns
        -------
        None
        """
        random.seed(self.random_seed)
        pep_dict = get_pep_dict_mgf(infiles)

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
    fire.Fire(DownsampleMS)

if __name__ == "__main__":
    main()
