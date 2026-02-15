"""Create train/validation/test dataset splits from annotated MGF files."""

import itertools
import logging
import pathlib
import random
import sys
from os import PathLike
from typing import Optional

import fire
import pyteomics.mgf
import tqdm


def create_datasets(
    *mgf_files: PathLike,
    output_root: str,
    spectra_per_peptide: Optional[int] = None,
    random_seed: int = 42,
    overwrite: bool = False,
    existing_splits: Optional[tuple[PathLike, PathLike, PathLike]] = None,
    combine_with_existing: bool = False,
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
        ``<output_root>.train.mgf``, ``<output_root>.val.mgf``, and
        ``<output_root>.test.mgf``. A log file ``<output_root>.log`` will
        also be created.
    spectra_per_peptide : int, optional
        If provided, randomly select at most this many spectra for each
        peptide from the new input files. When ``combine_with_existing``
        is True, existing spectra are not subject to this cap. By default
        all spectra are retained.
    random_seed : int, default=42
        Random seed for reproducible splitting and sampling.
    overwrite : bool, default=False
        If False, raise an error when any output file already exists.
        If True, overwrite existing output files.
    existing_splits : tuple of PathLike, optional
        A tuple of three MGF file paths (train, validation, test) containing
        pre-existing splits. Peptides from new input files that already appear
        in an existing split are routed to that same split.
    combine_with_existing : bool, default=False
        If True, output MGF files include both existing and new spectra.
        If False, only new spectra are written.
    """
    if not mgf_files:
        raise ValueError("At least one MGF file must be provided.")

    if spectra_per_peptide is not None and spectra_per_peptide <= 0:
        raise ValueError(
            f"spectra_per_peptide must be a positive integer, "
            f"got {spectra_per_peptide}."
        )

    if combine_with_existing and existing_splits is None:
        raise ValueError(
            "combine_with_existing=True requires existing_splits to be provided."
        )

    if not overwrite:
        expected_files = [
            pathlib.Path(f"{output_root}.{split}.mgf")
            for split in ("train", "val", "test")
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
    logger.propagate = False
    stream_handler = None
    file_handler = None
    try:
        logger.setLevel(logging.INFO)
        # Close and remove any pre-existing handlers to avoid leaking resources.
        for handler in list(logger.handlers):
            try:
                handler.close()
            finally:
                logger.removeHandler(handler)

        formatter = logging.Formatter("%(message)s")

        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(f"{output_root}.log", mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        # Clean up any handlers that were added before the failure.
        for handler in list(logger.handlers):
            try:
                handler.close()
            finally:
                logger.removeHandler(handler)
        raise

    try:
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
                f"Spectra eliminated by spectra_per_peptide="
                f"{spectra_per_peptide}: {eliminated}"
            )

        # Handle existing splits if provided.
        existing_peps = {"train": set(), "val": set(), "test": set()}
        existing_spectra = {"train": [], "val": [], "test": []}
        if existing_splits is not None:
            split_names = ("train", "val", "test")
            if len(existing_splits) != 3:
                raise ValueError(
                    f"existing_splits must contain exactly three paths "
                    f"(train, validation, test), but {len(existing_splits)} "
                    f"were provided."
                )
            for split_name, split_path in zip(split_names, existing_splits):
                for spectrum in tqdm.tqdm(
                    pyteomics.mgf.read(str(split_path), use_index=False),
                    desc=f"Reading existing {split_name}",
                    unit="PSM",
                ):
                    seq = spectrum["params"]["seq"]
                    existing_peps[split_name].add(seq)
                    if combine_with_existing:
                        existing_spectra[split_name].append(spectrum)
                logger.info(
                    f"Existing {split_name}: "
                    f"{len(existing_peps[split_name])} peptides"
                )

            # Validate mutual exclusivity of existing splits.
            for name_a, name_b in (
                ("train", "val"),
                ("train", "test"),
                ("val", "test"),
            ):
                shared = existing_peps[name_a] & existing_peps[name_b]
                if shared:
                    raise ValueError(
                        f"Peptide(s) found in multiple existing splits "
                        f"({name_a}, {name_b}): "
                        f"{', '.join(sorted(shared))}"
                    )

            all_existing = (
                existing_peps["train"] | existing_peps["val"] | existing_peps["test"]
            )
            overlapping = all_existing & set(pep_dict.keys())
            logger.info(
                f"Peptides overlapping with existing splits: {len(overlapping)}"
            )

        # Partition peptides into pre-assigned and new.
        pre_assigned = {"train": [], "val": [], "test": []}
        new_peptides = []
        for pep in sorted(pep_dict.keys()):
            assigned = False
            if existing_splits is not None:
                if pep in existing_peps["train"]:
                    pre_assigned["train"].append(pep)
                    assigned = True
                elif pep in existing_peps["val"]:
                    pre_assigned["val"].append(pep)
                    assigned = True
                elif pep in existing_peps["test"]:
                    pre_assigned["test"].append(pep)
                    assigned = True
            if not assigned:
                new_peptides.append(pep)

        random.shuffle(new_peptides)

        if existing_splits is not None:
            # Distribute new peptides to reach 80/10/10 overall,
            # counting ALL existing peptides (not just overlapping ones).
            total_peptides = (
                len(existing_peps["train"])
                + len(existing_peps["val"])
                + len(existing_peps["test"])
                + len(new_peptides)
            )
            if total_peptides >= 3:
                target_val = max(1, round(total_peptides * 0.1))
                target_test = max(1, round(total_peptides * 0.1))
                target_train = total_peptides - target_val - target_test
            else:
                logger.warning(
                    "Fewer than 3 peptides available across existing and "
                    "new data (%d peptides). One or more of the "
                    "train/validation/test splits may be empty.",
                    total_peptides,
                )
                target_train = round(total_peptides * 0.8)
                target_val = round(total_peptides * 0.1)
                target_test = total_peptides - target_train - target_val

            need_train = max(0, target_train - len(existing_peps["train"]))
            need_val = max(0, target_val - len(existing_peps["val"]))
            need_test = max(0, target_test - len(existing_peps["test"]))

            total_needed = need_train + need_val + need_test
            available = len(new_peptides)

            if available >= total_needed:
                # Enough new peptides to fill all targets.
                train_new = new_peptides[:need_train]
                val_new = new_peptides[need_train : need_train + need_val]
                test_new = new_peptides[
                    need_train + need_val : need_train + need_val + need_test
                ]
            else:
                # Not enough; distribute proportionally.
                if total_needed > 0:
                    train_new = new_peptides[
                        : round(available * need_train / total_needed)
                    ]
                    remaining = new_peptides[len(train_new) :]
                    adjusted_need_val = need_val
                    adjusted_need_test = need_test
                    adjusted_total = adjusted_need_val + adjusted_need_test
                    if adjusted_total > 0:
                        val_count = round(
                            len(remaining) * adjusted_need_val / adjusted_total
                        )
                    else:
                        val_count = 0
                    val_new = remaining[:val_count]
                    test_new = remaining[val_count:]
                else:
                    train_new = new_peptides
                    val_new = []
                    test_new = []

            train_peps = pre_assigned["train"] + train_new
            val_peps = pre_assigned["val"] + val_new
            test_peps = pre_assigned["test"] + test_new
        else:
            # No existing splits: original behavior.
            n = len(new_peptides)
            if n < 3:
                logger.warning(
                    "Only %d unique peptides available; assigning all to "
                    "training set and leaving validation/test splits empty.",
                    n,
                )
                train_peps = new_peptides
                val_peps = []
                test_peps = []
            else:
                n_val = max(1, round(n * 0.1))
                n_test = max(1, round(n * 0.1))
                n_train = n - n_val - n_test

                train_peps = new_peptides[:n_train]
                val_peps = new_peptides[n_train : n_train + n_val]
                test_peps = new_peptides[n_train + n_val :]

        splits = {
            "train": train_peps,
            "val": val_peps,
            "test": test_peps,
        }

        for split_name, peps in splits.items():
            split_spectra = list(
                itertools.chain.from_iterable(pep_dict[p] for p in peps)
            )
            if combine_with_existing:
                split_spectra = existing_spectra[split_name] + split_spectra
            outfile = f"{output_root}.{split_name}.mgf"
            split_spectra_iter = tqdm.tqdm(
                split_spectra,
                desc=f"Writing {outfile}",
                unit="PSM",
            )
            pyteomics.mgf.write(spectra=split_spectra_iter, output=outfile)
            if combine_with_existing:
                new_pep_count = len(set(peps) - existing_peps[split_name])
                total_peps = len(set(peps) | existing_peps[split_name])
                logger.info(
                    f"{split_name}: {len(split_spectra)} spectra, "
                    f"{new_pep_count} new peptides, "
                    f"{total_peps} total peptides"
                )
            else:
                logger.info(
                    f"{split_name}: {len(split_spectra)} spectra, "
                    f"{len(peps)} peptides"
                )
    finally:
        if file_handler is not None:
            file_handler.close()
            logger.removeHandler(file_handler)
        if stream_handler is not None:
            stream_handler.close()
            logger.removeHandler(stream_handler)


def main() -> None:
    """CLI entry point for create-datasets."""
    fire.Fire(create_datasets)


if __name__ == "__main__":
    main()
