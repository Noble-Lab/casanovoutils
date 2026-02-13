"""Create train/validation/test dataset splits from annotated MGF files."""

import itertools
import logging
import pathlib
import random
from os import PathLike
from typing import Iterable, Optional

import fire
import pyteomics.mgf
import tqdm


def create_datasets(
    *mgf_files: Iterable[PathLike],
    output_root: str,
    spectra_per_peptide: Optional[int] = None,
    random_seed: int = 42,
    overwrite: bool = False,
) -> None:
    """Create peptide-level train/validation/test splits from annotated MGF files.

    All spectra from the input MGF files are combined and grouped by peptide
    sequence. The unique peptides are randomly split into training (80%),
    validation (10%), and test (10%) sets. Spectra are then assigned to splits
    based on their associated peptide, ensuring no peptide-level leakage
    between splits.

    Parameters
    ----------
    *mgf_files : PathLike
        One or more paths to annotated MGF files. Each spectrum must contain
        the peptide sequence in ``spectrum["params"]["seq"]``.
    output_root : str
        Root path for the output files. Three MGF files will be created:
        ``<output_root>.train.mgf``, ``<output_root>.validation.mgf``, and
        ``<output_root>.test.mgf``. A log file ``<output_root>.log`` will
        also be created.
    spectra_per_peptide : int, optional
        If provided, randomly select at most this many spectra for each
        peptide. By default all spectra are retained.
    random_seed : int, default=42
        Random seed for reproducible splitting and sampling.
    overwrite : bool, default=False
        If False, raise an error when any output file already exists.
        If True, overwrite existing output files.
    """
    if not mgf_files:
        raise ValueError("At least one MGF file must be provided.")

    if not overwrite:
        expected_files = [
            pathlib.Path(f"{output_root}.{split}.mgf")
            for split in ("train", "validation", "test")
        ]
        expected_files.append(pathlib.Path(f"{output_root}.log"))
        existing = [f for f in expected_files if f.exists()]
        if existing:
            file_list = ", ".join(str(f) for f in existing)
            raise FileExistsError(
                f"Output files already exist: {file_list}. "
                f"Use --overwrite to overwrite."
            )

    logger = logging.getLogger("create_datasets")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(f"{output_root}.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    random.seed(random_seed)

    pep_dict: dict[str, list] = {}
    total_spectra = 0
    for mgf_file in mgf_files:
        file_count = 0
        for spectrum in tqdm.tqdm(
            pyteomics.mgf.read(str(mgf_file), use_index=False),
            desc=f"Reading {mgf_file}",
            unit="PSM",
        ):
            seq = spectrum["params"]["seq"]
            pep_dict.setdefault(seq, []).append(spectrum)
            file_count += 1
        logger.info(f"Read {file_count} spectra from {mgf_file}.")
        total_spectra += file_count

    logger.info(f"Total spectra read: {total_spectra}")
    logger.info(f"Unique peptides: {len(pep_dict)}")

    spectra_before = sum(len(psms) for psms in pep_dict.values())
    if spectra_per_peptide is not None:
        for pep, psms in pep_dict.items():
            if len(psms) > spectra_per_peptide:
                pep_dict[pep] = random.sample(psms, spectra_per_peptide)
        spectra_after = sum(len(psms) for psms in pep_dict.values())
        eliminated = spectra_before - spectra_after
        logger.info(
            f"Spectra eliminated by spectra_per_peptide={spectra_per_peptide}: "
            f"{eliminated}"
        )

    peptides = sorted(pep_dict.keys())
    random.shuffle(peptides)

    n = len(peptides)
    n_val = max(1, round(n * 0.1))
    n_test = max(1, round(n * 0.1))
    n_train = n - n_val - n_test

    train_peps = peptides[:n_train]
    val_peps = peptides[n_train : n_train + n_val]
    test_peps = peptides[n_train + n_val :]

    splits = {
        "train": train_peps,
        "validation": val_peps,
        "test": test_peps,
    }

    for split_name, peps in splits.items():
        split_spectra = list(
            itertools.chain.from_iterable(pep_dict[p] for p in peps)
        )
        outfile = f"{output_root}.{split_name}.mgf"
        split_spectra_iter = tqdm.tqdm(
            split_spectra,
            desc=f"Writing {outfile}",
            unit="PSM",
        )
        pyteomics.mgf.write(spectra=split_spectra_iter, output=outfile)
        logger.info(
            f"{split_name}: {len(split_spectra)} spectra, "
            f"{len(peps)} peptides"
        )

    file_handler.close()
    logger.removeHandler(file_handler)


def main() -> None:
    """CLI entry point for create-datasets."""
    fire.Fire(create_datasets)


if __name__ == "__main__":
    main()
