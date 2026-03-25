"""Create train/validation/test dataset splits from annotated MGF files."""

import logging
import pathlib
import random
import sys
from os import PathLike
from typing import Optional

import fire
import pyteomics.mgf
import tqdm


# Number of spectra to buffer per output file before flushing to disk.
_WRITE_BUFFER_SIZE = 1000


def _collect_peptide_counts(
    mgf_files: tuple[PathLike, ...],
    logger: logging.Logger,
) -> tuple[dict[str, int], int]:
    """Pass 1: stream all input MGF files and count spectra per peptide.

    Parameters
    ----------
    mgf_files : tuple of PathLike
        Paths to the input annotated MGF files.
    logger : logging.Logger
        Logger for progress messages.

    Returns
    -------
    pep_counts : dict[str, int]
        Mapping of peptide sequence to spectrum count.
    total_spectra : int
        Total number of spectra read across all files.
    """
    pep_counts: dict[str, int] = {}
    total_spectra = 0
    for mgf_file in mgf_files:
        file_count = 0
        for spectrum_index, spectrum in enumerate(
            tqdm.tqdm(
                pyteomics.mgf.read(str(mgf_file), use_index=False),
                desc=f"Reading {mgf_file} (pass 1)",
                unit="PSM",
            ),
            start=1,
        ):
            try:
                seq = spectrum["params"]["seq"]
            except KeyError as exc:
                raise KeyError(
                    f"Missing 'seq' in spectrum params for spectrum "
                    f"{spectrum_index} in file {mgf_file}"
                ) from exc
            pep_counts[seq] = pep_counts.get(seq, 0) + 1
            file_count += 1
        logger.info(f"Read {file_count} spectra from {mgf_file}.")
        total_spectra += file_count

    logger.info(f"Total spectra read: {total_spectra}")
    logger.info(f"Unique peptides: {len(pep_counts)}")
    return pep_counts, total_spectra


def _assign_splits(
    pep_counts: dict[str, int],
    total_spectra: int,
    existing_splits: Optional[tuple[PathLike, PathLike, PathLike]],
    spectra_per_peptide: Optional[int],
    logger: logging.Logger,
) -> tuple[dict[str, str], dict[str, set[int]], dict[str, set[str]]]:
    """Compute per-peptide split assignments and sampling indices.

    Uses the global random state (caller must seed before calling).

    Parameters
    ----------
    pep_counts : dict[str, int]
        Mapping of peptide sequence to spectrum count from pass 1.
    total_spectra : int
        Total spectra across all input files.
    existing_splits : tuple of PathLike or None
        Paths to existing train/val/test MGF files, or None.
    spectra_per_peptide : int or None
        Maximum spectra to retain per peptide, or None.
    logger : logging.Logger
        Logger for progress messages.

    Returns
    -------
    pep_to_split : dict[str, str]
        Mapping of peptide sequence to split name ("train", "val", "test").
    sampled_indices : dict[str, set[int]]
        For peptides that exceed spectra_per_peptide, the 0-based indices
        of spectra to retain. Empty dict if spectra_per_peptide is None.
    existing_peps : dict[str, set[str]]
        Peptides already present in each existing split. Empty sets if
        existing_splits is None.
    """
    # Pre-compute which spectrum indices to keep for each peptide when
    # spectra_per_peptide is set.
    sampled_indices: dict[str, set[int]] = {}
    if spectra_per_peptide is not None:
        spectra_after = 0
        for pep, count in pep_counts.items():
            if count > spectra_per_peptide:
                sampled_indices[pep] = set(
                    random.sample(range(count), spectra_per_peptide)
                )
                spectra_after += spectra_per_peptide
            else:
                spectra_after += count
        eliminated = total_spectra - spectra_after
        logger.info(
            f"Spectra eliminated by spectra_per_peptide="
            f"{spectra_per_peptide}: {eliminated}"
        )

    # Handle existing splits if provided.
    existing_peps: dict[str, set[str]] = {
        "train": set(),
        "val": set(),
        "test": set(),
    }
    if existing_splits is not None:
        split_names = ("train", "val", "test")
        if len(existing_splits) != 3:
            raise ValueError(
                f"existing_splits must contain exactly three paths "
                f"(train, validation, test), but {len(existing_splits)} "
                f"were provided."
            )
        for split_name, split_path in zip(
            split_names, existing_splits, strict=True
        ):
            for spectrum in tqdm.tqdm(
                pyteomics.mgf.read(str(split_path), use_index=False),
                desc=f"Reading existing {split_name}",
                unit="PSM",
            ):
                try:
                    seq = spectrum["params"]["seq"]
                except KeyError as exc:
                    raise KeyError(
                        f"Missing 'seq' in spectrum params while reading "
                        f"existing split '{split_name}' from file "
                        f"'{split_path}'"
                    ) from exc
                existing_peps[split_name].add(seq)
            logger.info(
                f"Existing {split_name}: "
                f"{len(existing_peps[split_name])} peptide"
                f"{'s' if len(existing_peps[split_name]) != 1 else ''}"
            )

        # Validate mutual exclusivity of existing splits.
        for name_a, name_b in (
            ("train", "val"),
            ("train", "test"),
            ("val", "test"),
        ):
            shared = existing_peps[name_a] & existing_peps[name_b]
            if shared:
                examples = sorted(shared)[:5]
                suffix = (
                    f" (and {len(shared) - 5} more)"
                    if len(shared) > 5
                    else ""
                )
                raise ValueError(
                    f"Peptide(s) found in multiple existing splits "
                    f"({name_a}, {name_b}): "
                    f"{', '.join(examples)}{suffix}"
                )

        all_existing = (
            existing_peps["train"]
            | existing_peps["val"]
            | existing_peps["test"]
        )
        overlapping = all_existing & set(pep_counts.keys())
        logger.info(
            f"Peptides overlapping with existing splits: {len(overlapping)}"
        )

    # Partition peptides into pre-assigned and new.
    pre_assigned: dict[str, list[str]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    new_peptides: list[str] = []
    for pep in sorted(pep_counts.keys()):
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
                f"Fewer than 3 peptides available across existing and "
                f"new data ({total_peptides} peptides). One or more of "
                f"the train/validation/test splits may be empty."
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
                        len(remaining)
                        * adjusted_need_val
                        / adjusted_total
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
                f"Only {n} unique peptides available; assigning all to "
                f"training set and leaving validation/test splits empty."
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

    # Build peptide -> split lookup for pass 2.
    pep_to_split: dict[str, str] = {}
    for pep in train_peps:
        pep_to_split[pep] = "train"
    for pep in val_peps:
        pep_to_split[pep] = "val"
    for pep in test_peps:
        pep_to_split[pep] = "test"

    return pep_to_split, sampled_indices, existing_peps


def _write_splits(
    mgf_files: tuple[PathLike, ...],
    output_root: str,
    pep_to_split: dict[str, str],
    sampled_indices: dict[str, set[int]],
    spectra_per_peptide: Optional[int],
    existing_splits: Optional[tuple[PathLike, PathLike, PathLike]],
    combine_with_existing: bool,
    logger: logging.Logger,
) -> tuple[dict[str, int], dict[str, set[str]]]:
    """Pass 2: stream spectra to output MGF files.

    Parameters
    ----------
    mgf_files : tuple of PathLike
        Paths to the input annotated MGF files.
    output_root : str
        Root path for output files.
    pep_to_split : dict[str, str]
        Mapping from peptide sequence to split name.
    sampled_indices : dict[str, set[int]]
        Per-peptide sets of 0-based spectrum indices to retain.
    spectra_per_peptide : int or None
        Maximum spectra per peptide (used to check sampled_indices).
    existing_splits : tuple of PathLike or None
        Paths to existing split files, required when combine_with_existing.
    combine_with_existing : bool
        If True, prepend existing split spectra to each output file.
    logger : logging.Logger
        Logger for progress messages.

    Returns
    -------
    split_spectra_counts : dict[str, int]
        Number of spectra written to each split.
    split_pep_sets : dict[str, set[str]]
        Set of new peptides assigned to each split.
    """
    split_pep_sets = {
        split: {pep for pep, s in pep_to_split.items() if s == split}
        for split in ("train", "val", "test")
    }

    outfiles = {
        split: f"{output_root}.{split}.mgf"
        for split in ("train", "val", "test")
    }

    split_spectra_counts: dict[str, int] = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    with (
        open(outfiles["train"], "w") as f_train,
        open(outfiles["val"], "w") as f_val,
        open(outfiles["test"], "w") as f_test,
    ):
        file_handles = {
            "train": f_train,
            "val": f_val,
            "test": f_test,
        }
        buffers: dict[str, list] = {
            "train": [],
            "val": [],
            "test": [],
        }

        def flush_buffer(split_name: str) -> None:
            """Write buffered spectra to the output file."""
            if buffers[split_name]:
                pyteomics.mgf.write(
                    spectra=buffers[split_name],
                    output=file_handles[split_name],
                )
                buffers[split_name].clear()

        def write_spectrum(split_name: str, spectrum: dict) -> None:
            """Buffer a spectrum and flush when buffer is full."""
            buffers[split_name].append(spectrum)
            split_spectra_counts[split_name] += 1
            if len(buffers[split_name]) >= _WRITE_BUFFER_SIZE:
                flush_buffer(split_name)

        # If combine_with_existing, stream existing split files first.
        if combine_with_existing:
            split_names = ("train", "val", "test")
            for split_name, split_path in zip(
                split_names, existing_splits, strict=True
            ):
                for spectrum in tqdm.tqdm(
                    pyteomics.mgf.read(str(split_path), use_index=False),
                    desc=f"Writing existing {split_name} (pass 2)",
                    unit="PSM",
                ):
                    write_spectrum(split_name, spectrum)

        # Stream new input MGFs.
        pep_counters: dict[str, int] = {}
        for mgf_file in mgf_files:
            for spectrum in tqdm.tqdm(
                pyteomics.mgf.read(str(mgf_file), use_index=False),
                desc=f"Writing {mgf_file} (pass 2)",
                unit="PSM",
            ):
                seq = spectrum["params"]["seq"]

                # Apply spectra_per_peptide filtering.
                if spectra_per_peptide is not None:
                    idx = pep_counters.get(seq, 0)
                    pep_counters[seq] = idx + 1
                    if seq in sampled_indices:
                        if idx not in sampled_indices[seq]:
                            continue
                    # If seq not in sampled_indices, count <= limit,
                    # so keep all.

                split_name = pep_to_split.get(seq)
                if split_name is not None:
                    write_spectrum(split_name, spectrum)

        # Flush remaining buffers.
        for split_name in ("train", "val", "test"):
            flush_buffer(split_name)

    return split_spectra_counts, split_pep_sets


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

        pep_counts, total_spectra = _collect_peptide_counts(mgf_files, logger)

        pep_to_split, sampled_indices, existing_peps = _assign_splits(
            pep_counts,
            total_spectra,
            existing_splits,
            spectra_per_peptide,
            logger,
        )

        split_spectra_counts, split_pep_sets = _write_splits(
            mgf_files,
            output_root,
            pep_to_split,
            sampled_indices,
            spectra_per_peptide,
            existing_splits,
            combine_with_existing,
            logger,
        )

        # Log split summaries.
        for split_name in ("train", "val", "test"):
            peps = split_pep_sets[split_name]
            count = split_spectra_counts[split_name]
            if combine_with_existing:
                new_pep_count = len(peps - existing_peps[split_name])
                total_peps = len(peps | existing_peps[split_name])
                logger.info(
                    f"{split_name}: {count} spectra, "
                    f"{new_pep_count} new peptides, "
                    f"{total_peps} total peptides"
                )
            else:
                logger.info(
                    f"{split_name}: {count} spectra, "
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
