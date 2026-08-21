"""Create train/validation/test dataset splits from annotated MGF files."""

import logging
import pathlib
import random
import re
import tempfile
from os import PathLike
from typing import Optional

import fire
import pyteomics.mgf
import tqdm

from . import configure_logging
from .filter_spectra import _load_config, _make_tokenizer, _parse_charge
from .mskb2proforma import convert as _mskb2proforma_convert
from .types import Commands

# Number of spectra to buffer per output file before flushing to disk.
_WRITE_BUFFER_SIZE = 1000

# Matches a bracket-enclosed modification token with an optional leading or
# trailing dash (N-terminal: "[Acetyl]-"; C-terminal: "-[Amidated]").
_MOD_RE = re.compile(r"-?\[[^\]]*\]-?")

# Matches MassIVE-KB PTM notation: a sign+mass shift (integer or decimal) at
# the start of a sequence (N-terminal) or immediately after a residue letter.
_MSKB_SEQ_RE = re.compile(r"(?:^|[A-Z])[+-]\d+(?:\.\d+)?")


def _canonical(seq: str) -> str:
    """Return the canonical form of a peptide sequence.

    Residues that are indistinguishable by mass spectrometry are mapped to a
    single representative token so that mass-equivalent variants are always
    grouped together during splitting:

    * Isoleucine (I) → Leucine (L)  [identical mass]
    * N[Deamidated] → D  [deamidation of Asn yields Asp; identical mass]
    * Q[Deamidated] → E  [deamidation of Gln yields Glu; identical mass]

    The canonical form is used only as a grouping key; original sequences are
    preserved unchanged in the output MGF files.

    Parameters
    ----------
    seq : str
        Peptide sequence, optionally containing bracket-enclosed modifications
        in the form ``AA[ModName]``.

    Returns
    -------
    str
        Canonical peptide sequence.
    """
    # Replace I with L only at residue positions (bracket depth 0), so that
    # uppercase I characters inside modification names are left untouched.
    chars = []
    depth = 0
    for ch in seq:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "I" and depth == 0:
            ch = "L"
        chars.append(ch)
    seq = "".join(chars)
    seq = seq.replace("N[Deamidated]", "D")
    seq = seq.replace("Q[Deamidated]", "E")
    return seq


def _strip_mods(seq: str) -> str:
    """Remove all modification tokens from a ProForma peptide sequence.

    Strips bracket-enclosed modification names and masses, including
    N-terminal modification tokens (e.g. ``[Acetyl]-``).

    Parameters
    ----------
    seq : str
        ProForma peptide sequence, e.g. ``"[Acetyl]-AC[Carbamidomethyl]GK"``.

    Returns
    -------
    str
        Bare amino-acid sequence, e.g. ``"ACGK"``.
    """
    return _MOD_RE.sub("", seq)


def _write_peptides_txt(
    output_root: str,
    mod_to_bare_by_split: dict[str, dict[str, str]],
) -> None:
    """Write ``<split>.peptides.txt`` files for each split.

    Each output line contains two tab-separated columns: the modified sequence
    (ProForma, as it appears in the MGF) and the canonical bare sequence.  The
    canonical bare sequence is produced by applying the same isobaric
    substitutions used for splitting (I→L, N[Deamidated]→D, Q[Deamidated]→E)
    and then stripping all remaining modification tokens, so mass-equivalent
    variants share a single bare sequence.  Lines are sorted alphabetically by
    the canonical bare sequence, then by the modified sequence.

    Also logs, per split, the number of unique modified sequences and the
    number of unique bare sequences.

    Parameters
    ----------
    output_root : str
        Root path used to write peptides files.
    mod_to_bare_by_split : dict[str, dict[str, str]]
        Mapping of split name to ``{modified_seq: bare_seq}`` dict, collected
        during pass 2 by ``_write_splits``.
    """
    for split in ("train", "val", "test"):
        pep_path = pathlib.Path(f"{output_root}.{split}.peptides.txt")
        mod_to_bare = mod_to_bare_by_split[split]
        pairs = sorted(mod_to_bare.items(), key=lambda kv: (kv[1], kv[0]))
        with open(pep_path, "w") as fh:
            for modified, bare in pairs:
                fh.write(f"{modified}\t{bare}\n")
        n_modified = len(mod_to_bare)
        n_bare = len(set(mod_to_bare.values()))
        logging.info(
            f"{split} peptides.txt: {n_modified} unique modified sequences, "
            f"{n_bare} unique bare sequences"
        )


def _sampling_key(spectrum: dict) -> tuple[str, tuple]:
    """Return the key used for spectra_per_precursor counting.

    The key is ``(canonical_seq, charge_tuple)`` so that spectra with the
    same peptide sequence but different precursor charge states are counted
    independently.  This allows each charge state to contribute up to
    ``spectra_per_precursor`` spectra.

    Parameters
    ----------
    spectrum : dict
        A spectrum dict as returned by pyteomics.

    Returns
    -------
    tuple[str, tuple]
        ``(_canonical(seq), tuple(charge))`` where *charge* is the list of
        charge values from the spectrum params (may be empty).
    """
    seq = spectrum["params"]["seq"]
    charge_raw = spectrum["params"].get("charge", [])
    if isinstance(charge_raw, (list, tuple)):
        charge = tuple(charge_raw)
    elif charge_raw is None:
        charge = ()
    else:
        charge = (charge_raw,)
    return (_canonical(seq), charge)


def _collect_peptide_counts(
    mgf_files: tuple[PathLike, ...],
) -> tuple[dict[str, int], dict[tuple, int], int]:
    """Pass 1: stream all input MGF files and count spectra per peptide.

    Parameters
    ----------
    mgf_files : tuple of PathLike
        Paths to the input annotated MGF files.

    Returns
    -------
    pep_counts : dict[str, int]
        Mapping of canonical peptide sequence to spectrum count.  Used for
        split assignment (peptide-level).
    sampling_counts : dict[tuple, int]
        Mapping of ``(canonical_seq, charge_tuple)`` to spectrum count.
        Used for per-charge-state ``spectra_per_precursor`` capping.
    total_spectra : int
        Total number of spectra read across all files.
    """
    pep_counts: dict[str, int] = {}
    sampling_counts: dict[tuple, int] = {}
    raw_seqs: set[str] = set()
    total_spectra = 0
    for mgf_file in mgf_files:
        file_count = 0
        with pyteomics.mgf.read(str(mgf_file), use_index=False) as reader:
            for spectrum_index, spectrum in enumerate(
                tqdm.tqdm(
                    reader,
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
                raw_seqs.add(seq)
                pep_key = _canonical(seq)
                pep_counts[pep_key] = pep_counts.get(pep_key, 0) + 1
                samp_key = _sampling_key(spectrum)
                sampling_counts[samp_key] = sampling_counts.get(samp_key, 0) + 1
                file_count += 1
        logging.info(f"Read {file_count} spectra from {mgf_file}.")
        total_spectra += file_count

    logging.info(f"Total spectra read: {total_spectra}")
    logging.info(f"Unique raw sequences: {len(raw_seqs)}")
    n_collapsed = len(raw_seqs) - len(pep_counts)
    logging.info(
        f"Unique canonical peptides (after I/L and deamidation collapsing): "
        f"{len(pep_counts)}"
        + (f" ({n_collapsed} sequences merged)" if n_collapsed > 0 else "")
    )
    logging.info(f"Unique precursors (sequence + charge state): {len(sampling_counts)}")
    return pep_counts, sampling_counts, total_spectra


def _assign_splits(
    pep_counts: dict[str, int],
    sampling_counts: dict[tuple, int],
    total_spectra: int,
    existing_splits: Optional[tuple[PathLike, PathLike, PathLike]],
    spectra_per_precursor: Optional[int],
) -> tuple[dict[str, str], dict[tuple, set[int]], dict[str, set[str]]]:
    """Compute per-peptide split assignments and sampling indices.

    Uses the global random state (caller must seed before calling).

    Parameters
    ----------
    pep_counts : dict[str, int]
        Mapping of canonical peptide sequence to spectrum count from pass 1.
        Used for split-proportion calculations (peptide-level).
    sampling_counts : dict[tuple, int]
        Mapping of ``(canonical_seq, charge_tuple)`` to spectrum count from
        pass 1.  Used to cap spectra per (peptide, charge) combination.
    total_spectra : int
        Total spectra across all input files.
    existing_splits : tuple of PathLike or None
        Paths to existing train/val/test MGF files, or None.
    spectra_per_precursor : int or None
        Maximum spectra to retain per (peptide, charge state), or None.

    Returns
    -------
    pep_to_split : dict[str, str]
        Mapping of canonical peptide sequence to split name
        ("train", "val", "test").
    sampled_indices : dict[tuple, set[int]]
        For (peptide, charge) combinations that exceed spectra_per_precursor,
        the 0-based indices of spectra to retain.  Empty dict if
        spectra_per_precursor is None.
    existing_peps : dict[str, set[str]]
        Canonical peptides already present in each existing split. Empty sets
        if existing_splits is None.
    """
    # Pre-compute which spectrum indices to keep for each (peptide, charge)
    # when spectra_per_precursor is set.
    sampled_indices: dict[tuple, set[int]] = {}
    if spectra_per_precursor is not None:
        spectra_after = 0
        for samp_key, count in sampling_counts.items():
            if count > spectra_per_precursor:
                sampled_indices[samp_key] = set(
                    random.sample(range(count), spectra_per_precursor)
                )
                spectra_after += spectra_per_precursor
            else:
                spectra_after += count
        eliminated = total_spectra - spectra_after
        n_capped = len(sampled_indices)
        n_precursors = len(sampling_counts)
        logging.info(
            f"Unique precursors exceeding spectra_per_precursor="
            f"{spectra_per_precursor}: {n_capped} of {n_precursors}"
        )
        logging.info(
            f"Spectra eliminated by spectra_per_precursor="
            f"{spectra_per_precursor}: {eliminated} "
            f"({spectra_after} retained)"
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
        for split_name, split_path in zip(split_names, existing_splits, strict=True):
            with pyteomics.mgf.read(str(split_path), use_index=False) as reader:
                for spectrum in tqdm.tqdm(
                    reader,
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
                    if _MSKB_SEQ_RE.search(seq):
                        raise ValueError(
                            f"Existing split '{split_name}' ({split_path}) "
                            f"appears to use MassIVE-KB PTM notation "
                            f"(e.g. sequence {seq!r}). "
                            f"existing_splits must be in ProForma format."
                        )
                    existing_peps[split_name].add(_canonical(seq))
            logging.info(
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
                suffix = f" (and {len(shared) - 5} more)" if len(shared) > 5 else ""
                raise ValueError(
                    f"Peptide(s) found in multiple existing splits "
                    f"({name_a}, {name_b}): "
                    f"{', '.join(examples)}{suffix}"
                )

        all_existing = (
            existing_peps["train"] | existing_peps["val"] | existing_peps["test"]
        )
        overlapping = all_existing & set(pep_counts.keys())
        logging.info(f"Peptides overlapping with existing splits: {len(overlapping)}")

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
            logging.warning(
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
                train_new = new_peptides[: round(available * need_train / total_needed)]
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
            logging.warning(
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

    logging.info(
        f"Canonical peptides assigned: "
        f"train={len(train_peps)}, "
        f"val={len(val_peps)}, "
        f"test={len(test_peps)}"
    )

    return pep_to_split, sampled_indices, existing_peps


def _write_splits(
    mgf_files: tuple[PathLike, ...],
    output_root: str,
    pep_to_split: dict[str, str],
    sampled_indices: dict[tuple, set[int]],
    spectra_per_precursor: Optional[int],
    existing_splits: Optional[tuple[PathLike, PathLike, PathLike]],
    combine_with_existing: bool,
) -> tuple[dict[str, int], dict[str, set[str]], dict[str, dict[str, str]]]:
    """Pass 2: stream spectra to output MGF files.

    Parameters
    ----------
    mgf_files : tuple of PathLike
        Paths to the input annotated MGF files.
    output_root : str
        Root path for output files.
    pep_to_split : dict[str, str]
        Mapping from canonical peptide sequence to split name.
    sampled_indices : dict[tuple, set[int]]
        Per-(peptide, charge) sets of 0-based spectrum indices to retain.
    spectra_per_precursor : int or None
        Maximum spectra per (peptide, charge) combination.  Used to decide
        whether to consult *sampled_indices* for a given precursor.  If
        ``None``, all spectra are retained (no cap applied).
    existing_splits : tuple of PathLike or None
        Paths to existing split files, required when combine_with_existing.
    combine_with_existing : bool
        If True, prepend existing split spectra to each output file.

    Returns
    -------
    split_spectra_counts : dict[str, int]
        Number of spectra written to each split.
    split_pep_sets : dict[str, set[str]]
        Set of new canonical peptides assigned to each split.
    mod_to_bare_by_split : dict[str, dict[str, str]]
        Per-split mapping of modified sequence to canonical bare sequence,
        collected during writing for use by ``_write_peptides_txt``.
    """
    split_pep_sets = {
        split: {pep for pep, s in pep_to_split.items() if s == split}
        for split in ("train", "val", "test")
    }

    outfiles = {
        split: f"{output_root}.{split}.mgf" for split in ("train", "val", "test")
    }

    split_spectra_counts: dict[str, int] = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    mod_to_bare_by_split: dict[str, dict[str, str]] = {
        "train": {},
        "val": {},
        "test": {},
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
            """Buffer a spectrum, track its sequence, and flush when full."""
            seq = spectrum["params"].get("seq")
            if seq and seq not in mod_to_bare_by_split[split_name]:
                mod_to_bare_by_split[split_name][seq] = _strip_mods(_canonical(seq))
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
                with pyteomics.mgf.read(str(split_path), use_index=False) as reader:
                    for spectrum in tqdm.tqdm(
                        reader,
                        desc=f"Writing existing {split_name} (pass 2)",
                        unit="PSM",
                    ):
                        write_spectrum(split_name, spectrum)

        # Stream new input MGFs.
        precursor_counters: dict[tuple, int] = {}
        for mgf_file in mgf_files:
            with pyteomics.mgf.read(str(mgf_file), use_index=False) as reader:
                for spectrum in tqdm.tqdm(
                    reader,
                    desc=f"Writing {mgf_file} (pass 2)",
                    unit="PSM",
                ):
                    seq = spectrum["params"]["seq"]
                    pep_key = _canonical(seq)
                    samp_key = _sampling_key(spectrum)

                    # Apply spectra_per_precursor filtering per (peptide, charge).
                    if spectra_per_precursor is not None:
                        idx = precursor_counters.get(samp_key, 0)
                        precursor_counters[samp_key] = idx + 1
                        if samp_key in sampled_indices:
                            if idx not in sampled_indices[samp_key]:
                                continue
                        # If samp_key not in sampled_indices, count <= limit,
                        # so keep all.

                    split_name = pep_to_split.get(pep_key)
                    if split_name is not None:
                        write_spectrum(split_name, spectrum)

        # Flush remaining buffers.
        for split_name in ("train", "val", "test"):
            flush_buffer(split_name)

    return split_spectra_counts, split_pep_sets, mod_to_bare_by_split


def _filter_mgf_files(
    mgf_files: tuple[PathLike, ...],
    casanovo_config: PathLike,
    tmpdir: pathlib.Path,
) -> tuple[PathLike, ...]:
    """Filter MGF files using a Casanovo config, writing results to *tmpdir*.

    For each input file, spectra that fail any of the following checks are
    removed:

    * Missing or empty ``SEQ`` field.
    * Sequence contains tokens absent from the Casanovo residue vocabulary.
    * Precursor charge is missing, ambiguous, zero, or exceeds ``max_charge``.
    * Fewer than ``min_peaks`` peaks.

    Per-criterion counts are written to the log. Returns a tuple of paths to
    the filtered MGF files in the same order as *mgf_files*.
    """
    cfg = _load_config(casanovo_config)
    min_peaks: int = cfg.get("min_peaks", 20)
    max_charge: int = cfg.get("max_charge", 10)
    tokenizer = _make_tokenizer(cfg)

    logging.info(
        f"Filtering with Casanovo config: {casanovo_config} "
        f"(min_peaks={min_peaks}, max_charge={max_charge})"
    )

    n_total = n_no_seq = n_bad_seq = n_bad_charge = n_few_peaks = 0

    filtered: list[PathLike] = []
    for i, src in enumerate(mgf_files):
        src = pathlib.Path(src)
        dst = tmpdir / f"filtered_{i}_{src.name}"

        def _passing(path):
            nonlocal n_total, n_no_seq, n_bad_seq, n_bad_charge, n_few_peaks
            for spectrum in tqdm.tqdm(
                pyteomics.mgf.read(str(path), use_index=False),
                desc=f"Filtering {path.name}",
                unit="psm",
            ):
                n_total += 1

                seq = spectrum["params"].get("seq", "")
                if not seq:
                    n_no_seq += 1
                    continue

                try:
                    tokenizer.tokenize(seq)
                except ValueError:
                    n_bad_seq += 1
                    continue

                charge = _parse_charge(spectrum["params"].get("charge"))
                if charge is None or charge <= 0 or charge > max_charge:
                    n_bad_charge += 1
                    continue

                if len(spectrum.get("m/z array", [])) < min_peaks:
                    n_few_peaks += 1
                    continue

                yield spectrum

        pyteomics.mgf.write(_passing(src), output=str(dst))
        filtered.append(dst)

    n_pass = n_total - n_no_seq - n_bad_seq - n_bad_charge - n_few_peaks
    logging.info(f"Filtering complete — total spectra read : {n_total}")
    logging.info(f"  Filtered: missing SEQ     : {n_no_seq}")
    logging.info(f"  Filtered: invalid tokens  : {n_bad_seq}")
    logging.info(f"  Filtered: invalid charge  : {n_bad_charge}")
    logging.info(f"  Filtered: too few peaks   : {n_few_peaks}")
    logging.info(f"  Filtered: total           : {n_total - n_pass}")
    logging.info(f"  Passing spectra           : {n_pass}")

    return tuple(filtered)


def create_datasets(
    *mgf_files: PathLike,
    output_root: str,
    spectra_per_precursor: Optional[int] = None,
    random_seed: int = 42,
    overwrite: bool = False,
    existing_splits: Optional[tuple[PathLike, PathLike, PathLike]] = None,
    combine_with_existing: bool = False,
    mskb_format: bool = False,
    casanovo_config: Optional[PathLike] = None,
    tmp_dir: Optional[PathLike] = None,
) -> None:
    """Create peptide-level train/validation/test splits from annotated MGF files.

    All spectra from the input MGF files are combined and grouped by peptide
    sequence. The unique peptides are randomly split into training (80%),
    validation (10%), and test (10%) sets. Spectra are then assigned to splits
    based on their associated peptide, ensuring no peptide-level leakage
    between splits.

    Several pairs of residues are indistinguishable by mass spectrometry and
    are always grouped together during splitting to prevent leakage:

    * Isoleucine (I) and leucine (L) have identical masses.
    * Deamidated asparagine (N[Deamidated]) and aspartic acid (D) have
      identical masses.
    * Deamidated glutamine (Q[Deamidated]) and glutamic acid (E) have
      identical masses.

    Each spectrum's peptide sequence is mapped to a canonical form (see
    ``_canonical``) before the split assignment is looked up, so all
    mass-equivalent variants are always grouped together. The original
    sequences are preserved unchanged in the output MGF files.

    Parameters
    ----------
    *mgf_files : PathLike
        One or more paths to annotated MGF files. Each spectrum must contain
        the peptide sequence in ``spectrum["params"]["seq"]``.
    output_root : str
        Root path for the output files. Three MGF files will be created:
        ``<output_root>.train.mgf``, ``<output_root>.val.mgf``, and
        ``<output_root>.test.mgf``. A two-column tab-separated peptides
        file is written for each split:
        ``<output_root>.[train,val,test].peptides.txt``. Column 1 is the
        modified sequence (ProForma) and column 2 is the canonical bare
        sequence (isobaric substitutions applied, then modification tokens
        stripped); rows are sorted by canonical bare sequence then by
        modified sequence. A log file ``<output_root>.log.txt``
        is also created.
    spectra_per_precursor : int, optional
        If provided, randomly select at most this many spectra for each
        (peptide, precursor charge state) combination from the new input
        files.  Spectra with the same sequence but different charge states
        are treated independently, so each charge state may contribute up
        to ``spectra_per_precursor`` spectra.  When ``combine_with_existing``
        is True, existing spectra are not subject to this cap.  By default
        all spectra are retained.
    random_seed : int, default=42
        Random seed for reproducible splitting and sampling.
    overwrite : bool, default=False
        If False, raise an error when any output file already exists.
        If True, overwrite existing output files.
    existing_splits : tuple of PathLike, optional
        A tuple of three MGF file paths (train, validation, test) containing
        pre-existing splits. Peptides from new input files that already appear
        in an existing split are routed to that same split. The files must use
        ProForma sequence notation; MassIVE-KB notation is not supported and
        will raise a ``ValueError``.
    combine_with_existing : bool, default=False
        If True, output MGF files include both existing and new spectra.
        If False, only new spectra are written.
    mskb_format : bool, default=False
        If True, input MGF files are assumed to use MassIVE-KB PTM notation
        and are converted to ProForma format before splitting. The output MGF
        files will contain the converted ProForma sequences, not the original
        MassIVE-KB strings. Conversion raises a ``ValueError`` if any
        sequence cannot be converted.
    casanovo_config : PathLike, optional
        Path to a Casanovo YAML configuration file. If provided, spectra are
        filtered before splitting using the vocabulary and thresholds in the
        config (``residues``, ``min_peaks``, ``max_charge``, and
        ``replace_isoleucine_with_leucine``). Spectra with a missing or empty
        SEQ field, tokens outside the vocabulary, an invalid or out-of-range
        charge, or fewer than ``min_peaks`` peaks are removed. Filtering is
        applied after any MassIVE-KB conversion. Per-criterion counts are
        written to the log file.
    tmp_dir : PathLike, optional
        Directory to use for temporary files (converted and filtered MGFs).
        Defaults to the system temporary directory (usually ``/tmp``). Set
        this to a directory on a volume with sufficient free space when
        processing large MGF files.
    """
    if not mgf_files:
        raise ValueError("At least one MGF file must be provided.")

    if spectra_per_precursor is not None and spectra_per_precursor <= 0:
        raise ValueError(
            f"spectra_per_precursor must be a positive integer, "
            f"got {spectra_per_precursor}."
        )

    if combine_with_existing and existing_splits is None:
        raise ValueError(
            "combine_with_existing=True requires existing_splits to be provided."
        )

    if not overwrite:
        expected_files = [
            pathlib.Path(f"{output_root}.{split}.{ext}")
            for split in ("train", "val", "test")
            for ext in ("mgf", "peptides.txt")
        ]
        expected_files.append(pathlib.Path(f"{output_root}.log.txt"))
        existing = [f for f in expected_files if f.exists()]
        if existing:
            file_list = ", ".join(str(f) for f in existing)
            raise FileExistsError(
                f"Output files already exist: {file_list}. "
                f"Use --overwrite to overwrite."
            )

    # Use file_mode="w" so log.txt is always a single-run artifact, consistent
    # with overwrite semantics for the MGF and peptides.txt outputs.
    file_handler = configure_logging(
        pathlib.Path(f"{output_root}.log.txt"),
        file_mode="w",
    )

    try:
        random.seed(random_seed)

        with tempfile.TemporaryDirectory(dir=tmp_dir) as tmpdir:
            tmp = pathlib.Path(tmpdir)
            if mskb_format:
                converted = []
                for i, src in enumerate(mgf_files):
                    src = pathlib.Path(src)
                    dst = tmp / f"converted_{i}_{src.name}"
                    logging.info(
                        f"Converting {src} from MassIVE-KB format to ProForma..."
                    )
                    _mskb2proforma_convert(src, dst)
                    converted.append(dst)
                mgf_files = tuple(converted)

            if casanovo_config is not None:
                mgf_files = _filter_mgf_files(mgf_files, casanovo_config, tmp)

            pep_counts, sampling_counts, total_spectra = _collect_peptide_counts(
                mgf_files
            )

            pep_to_split, sampled_indices, existing_peps = _assign_splits(
                pep_counts,
                sampling_counts,
                total_spectra,
                existing_splits,
                spectra_per_precursor,
            )

            split_spectra_counts, split_pep_sets, mod_to_bare_by_split = _write_splits(
                mgf_files,
                output_root,
                pep_to_split,
                sampled_indices,
                spectra_per_precursor,
                existing_splits,
                combine_with_existing,
            )

        # Log split summaries.
        for split_name in ("train", "val", "test"):
            peps = split_pep_sets[split_name]
            count = split_spectra_counts[split_name]
            if combine_with_existing:
                new_pep_count = len(peps - existing_peps[split_name])
                total_peps = len(peps | existing_peps[split_name])
                logging.info(
                    f"{split_name}: {count} spectra, "
                    f"{new_pep_count} new peptides, "
                    f"{total_peps} total peptides"
                )
            else:
                logging.info(f"{split_name}: {count} spectra, {len(peps)} peptides")

        _write_peptides_txt(output_root, mod_to_bare_by_split)
    finally:
        logging.root.removeHandler(file_handler)
        file_handler.close()


COMMANDS: Commands = create_datasets


def main() -> None:
    """CLI entry point for create-datasets."""
    fire.Fire(COMMANDS)


if __name__ == "__main__":
    main()
