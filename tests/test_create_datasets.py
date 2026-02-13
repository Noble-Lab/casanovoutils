"""Tests for the create_datasets module."""

import pathlib
import random

import numpy as np
import pyteomics.mgf
import pytest

from casanovoutils.create_datasets import create_datasets


def _write_mgf(path, spectra):
    """Write a list of (seq, mz, intensity) tuples as a valid MGF file.

    Parameters
    ----------
    path : pathlib.Path
        Output file path.
    spectra : list[tuple[str, list[float], list[float]]]
        Each element is (peptide_sequence, mz_values, intensity_values).

    Returns
    -------
    str
        The string path to the written file.
    """
    records = []
    for seq, mz, intensity in spectra:
        records.append(
            {
                "params": {"seq": seq, "pepmass": (100.0,)},
                "m/z array": np.array(mz),
                "intensity array": np.array(intensity),
            }
        )
    pyteomics.mgf.write(records, output=str(path))
    return str(path)


def _read_mgf(path):
    """Read all spectra from an MGF file into a list."""
    return list(pyteomics.mgf.read(str(path), use_index=False))


def _get_peptides(spectra):
    """Extract peptide sequences from a list of spectra."""
    return [s["params"]["seq"] for s in spectra]


class TestCreateDatasetsBasic:
    """Basic functionality tests for create_datasets."""

    def test_creates_three_output_files(self, tmp_path):
        """Output files for train, validation, and test are created."""
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [(f"PEP{i}", [100.0], [1.0]) for i in range(20)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        assert (tmp_path / "out.train.mgf").exists()
        assert (tmp_path / "out.validation.mgf").exists()
        assert (tmp_path / "out.test.mgf").exists()

    def test_all_spectra_preserved(self, tmp_path):
        """Total spectra across splits equals the input count."""
        n_peptides = 20
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [(f"PEP{i}", [100.0], [1.0]) for i in range(n_peptides)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.validation.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        assert len(train) + len(val) + len(test) == n_peptides

    def test_no_peptide_leakage_between_splits(self, tmp_path):
        """No peptide should appear in more than one split."""
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [(f"PEP{i}", [100.0], [1.0]) for i in range(30)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        train_peps = set(_get_peptides(_read_mgf(tmp_path / "out.train.mgf")))
        val_peps = set(_get_peptides(_read_mgf(tmp_path / "out.validation.mgf")))
        test_peps = set(_get_peptides(_read_mgf(tmp_path / "out.test.mgf")))

        assert train_peps.isdisjoint(val_peps)
        assert train_peps.isdisjoint(test_peps)
        assert val_peps.isdisjoint(test_peps)

    def test_approximate_split_ratio(self, tmp_path):
        """Peptide counts roughly follow the 80/10/10 split."""
        n_peptides = 100
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [(f"PEP{i}", [100.0], [1.0]) for i in range(n_peptides)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        train_peps = set(_get_peptides(_read_mgf(tmp_path / "out.train.mgf")))
        val_peps = set(_get_peptides(_read_mgf(tmp_path / "out.validation.mgf")))
        test_peps = set(_get_peptides(_read_mgf(tmp_path / "out.test.mgf")))

        assert len(train_peps) == 80
        assert len(val_peps) == 10
        assert len(test_peps) == 10


class TestCreateDatasetsMultipleFiles:
    """Tests for combining multiple input MGF files."""

    def test_multiple_mgf_files_combined(self, tmp_path):
        """Spectra from multiple input files are merged before splitting."""
        mgf1 = _write_mgf(
            tmp_path / "a.mgf",
            [(f"PEPA{i}", [100.0], [1.0]) for i in range(15)],
        )
        mgf2 = _write_mgf(
            tmp_path / "b.mgf",
            [(f"PEPB{i}", [200.0], [2.0]) for i in range(15)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(mgf1, mgf2, output_root=output_root)

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.validation.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        total = len(train) + len(val) + len(test)
        assert total == 30

        all_peps = set(
            _get_peptides(train) + _get_peptides(val) + _get_peptides(test)
        )
        assert any(p.startswith("PEPA") for p in all_peps)
        assert any(p.startswith("PEPB") for p in all_peps)


class TestCreateDatasetsSpectraPerPeptide:
    """Tests for the spectra_per_peptide option."""

    def test_limits_spectra_per_peptide(self, tmp_path):
        """Each peptide should have at most k spectra in the output."""
        spectra = []
        for i in range(10):
            for _ in range(5):
                spectra.append((f"PEP{i}", [100.0], [1.0]))
        mgf = _write_mgf(tmp_path / "input.mgf", spectra)
        output_root = str(tmp_path / "out")

        create_datasets(
            mgf, output_root=output_root, spectra_per_peptide=2,
        )

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.validation.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        all_spectra = train + val + test
        pep_counts = {}
        for s in all_spectra:
            pep = s["params"]["seq"]
            pep_counts[pep] = pep_counts.get(pep, 0) + 1

        for pep, count in pep_counts.items():
            assert count <= 2, f"{pep} has {count} spectra, expected <= 2"

    def test_no_limit_keeps_all_spectra(self, tmp_path):
        """Without spectra_per_peptide, all spectra are retained."""
        spectra = []
        for i in range(10):
            for _ in range(5):
                spectra.append((f"PEP{i}", [100.0], [1.0]))
        mgf = _write_mgf(tmp_path / "input.mgf", spectra)
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.validation.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        assert len(train) + len(val) + len(test) == 50

    def test_spectra_per_peptide_with_fewer_available(self, tmp_path):
        """Peptides with fewer than k spectra keep all of them."""
        spectra = [
            ("PEPA", [100.0], [1.0]),
            ("PEPB", [100.0], [1.0]),
            ("PEPB", [200.0], [2.0]),
            ("PEPB", [300.0], [3.0]),
        ]
        # Add more peptides to have enough for splitting.
        for i in range(18):
            spectra.append((f"PEPC{i}", [100.0], [1.0]))
        mgf = _write_mgf(tmp_path / "input.mgf", spectra)
        output_root = str(tmp_path / "out")

        create_datasets(
            mgf, output_root=output_root, spectra_per_peptide=5,
        )

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.validation.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        all_spectra = train + val + test
        pep_counts = {}
        for s in all_spectra:
            pep = s["params"]["seq"]
            pep_counts[pep] = pep_counts.get(pep, 0) + 1

        assert pep_counts.get("PEPA", 0) <= 1
        assert pep_counts.get("PEPB", 0) <= 3


class TestCreateDatasetsReproducibility:
    """Tests for deterministic behavior with random seeds."""

    def test_same_seed_produces_same_splits(self, tmp_path):
        """Running with the same seed should produce identical outputs."""
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [(f"PEP{i}", [100.0], [1.0]) for i in range(30)],
        )

        for run in ("run1", "run2"):
            output_root = str(tmp_path / run)
            create_datasets(mgf, output_root=output_root, random_seed=123)

        for split in ("train", "validation", "test"):
            spectra_1 = _read_mgf(tmp_path / f"run1.{split}.mgf")
            spectra_2 = _read_mgf(tmp_path / f"run2.{split}.mgf")
            peps_1 = _get_peptides(spectra_1)
            peps_2 = _get_peptides(spectra_2)
            assert peps_1 == peps_2

    def test_different_seed_produces_different_splits(self, tmp_path):
        """Different seeds should (almost certainly) produce different splits."""
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [(f"PEP{i}", [100.0], [1.0]) for i in range(30)],
        )

        create_datasets(
            mgf, output_root=str(tmp_path / "seed1"), random_seed=1,
        )
        create_datasets(
            mgf, output_root=str(tmp_path / "seed2"), random_seed=999,
        )

        peps_1 = set(
            _get_peptides(_read_mgf(tmp_path / "seed1.train.mgf"))
        )
        peps_2 = set(
            _get_peptides(_read_mgf(tmp_path / "seed2.train.mgf"))
        )
        assert peps_1 != peps_2


class TestCreateDatasetsEdgeCases:
    """Edge case and error handling tests."""

    def test_no_mgf_files_raises_error(self, tmp_path):
        """Passing zero MGF files should raise a ValueError."""
        with pytest.raises(ValueError, match="At least one MGF file"):
            create_datasets(output_root=str(tmp_path / "out"))

    def test_small_dataset_minimum_one_per_split(self, tmp_path):
        """With very few peptides, each split still gets at least one."""
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [(f"PEP{i}", [100.0], [1.0]) for i in range(3)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.validation.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        assert len(train) >= 1
        assert len(val) >= 1
        assert len(test) >= 1

    def test_duplicate_peptides_across_files_merged(self, tmp_path):
        """Same peptide in multiple files should be grouped together."""
        mgf1 = _write_mgf(
            tmp_path / "a.mgf",
            [("SHARED", [100.0], [1.0]), ("ONLYA", [100.0], [1.0])],
        )
        mgf2 = _write_mgf(
            tmp_path / "b.mgf",
            [("SHARED", [200.0], [2.0]), ("ONLYB", [100.0], [1.0])],
        )
        # Add more unique peptides so there are enough to split.
        mgf3 = _write_mgf(
            tmp_path / "c.mgf",
            [(f"EXTRA{i}", [100.0], [1.0]) for i in range(17)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(mgf1, mgf2, mgf3, output_root=output_root)

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.validation.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        all_spectra = train + val + test
        shared_spectra = [
            s for s in all_spectra if s["params"]["seq"] == "SHARED"
        ]
        assert len(shared_spectra) == 2

        # Both SHARED spectra must be in the same split.
        shared_in_train = [
            s for s in train if s["params"]["seq"] == "SHARED"
        ]
        shared_in_val = [
            s for s in val if s["params"]["seq"] == "SHARED"
        ]
        shared_in_test = [
            s for s in test if s["params"]["seq"] == "SHARED"
        ]
        counts = [len(shared_in_train), len(shared_in_val), len(shared_in_test)]
        assert sorted(counts) == [0, 0, 2]


class TestCreateDatasetsLogging:
    """Tests for log file creation and content."""

    def test_log_file_created(self, tmp_path):
        """A log file should be created at <output_root>.log."""
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [(f"PEP{i}", [100.0], [1.0]) for i in range(20)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        assert (tmp_path / "out.log").exists()

    def test_log_contains_per_file_counts(self, tmp_path):
        """Log should report how many spectra were read from each file."""
        mgf1 = _write_mgf(
            tmp_path / "a.mgf",
            [(f"PEPA{i}", [100.0], [1.0]) for i in range(5)],
        )
        mgf2 = _write_mgf(
            tmp_path / "b.mgf",
            [(f"PEPB{i}", [100.0], [1.0]) for i in range(15)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(mgf1, mgf2, output_root=output_root)

        log = (tmp_path / "out.log").read_text()
        assert "Read 5 spectra from" in log
        assert "Read 15 spectra from" in log

    def test_log_contains_total_and_unique(self, tmp_path):
        """Log should report total spectra and unique peptide counts."""
        spectra = []
        for i in range(10):
            for _ in range(3):
                spectra.append((f"PEP{i}", [100.0], [1.0]))
        mgf = _write_mgf(tmp_path / "input.mgf", spectra)
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        log = (tmp_path / "out.log").read_text()
        assert "Total spectra read: 30" in log
        assert "Unique peptides: 10" in log

    def test_log_contains_eliminated_count(self, tmp_path):
        """Log should report spectra eliminated by spectra_per_peptide."""
        spectra = []
        for i in range(10):
            for _ in range(5):
                spectra.append((f"PEP{i}", [100.0], [1.0]))
        mgf = _write_mgf(tmp_path / "input.mgf", spectra)
        output_root = str(tmp_path / "out")

        create_datasets(
            mgf, output_root=output_root, spectra_per_peptide=2,
        )

        log = (tmp_path / "out.log").read_text()
        assert "Spectra eliminated by spectra_per_peptide=2: 30" in log

    def test_log_no_eliminated_line_without_flag(self, tmp_path):
        """Without spectra_per_peptide, no elimination line in the log."""
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [(f"PEP{i}", [100.0], [1.0]) for i in range(10)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        log = (tmp_path / "out.log").read_text()
        assert "Spectra eliminated by" not in log

    def test_log_contains_split_summaries(self, tmp_path):
        """Log should report spectra and peptide counts per split."""
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [(f"PEP{i}", [100.0], [1.0]) for i in range(100)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        log = (tmp_path / "out.log").read_text()
        assert "train: 80 spectra, 80 peptides" in log
        assert "validation: 10 spectra, 10 peptides" in log
        assert "test: 10 spectra, 10 peptides" in log


class TestCreateDatasetsOverwrite:
    """Tests for the overwrite option."""

    def test_error_when_output_exists(self, tmp_path):
        """Raise FileExistsError if output files already exist."""
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [(f"PEP{i}", [100.0], [1.0]) for i in range(20)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        with pytest.raises(FileExistsError, match="already exist"):
            create_datasets(mgf, output_root=output_root)

    def test_overwrite_allows_rerun(self, tmp_path):
        """With overwrite=True, existing output files are replaced."""
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [(f"PEP{i}", [100.0], [1.0]) for i in range(20)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)
        create_datasets(mgf, output_root=output_root, overwrite=True)

        assert (tmp_path / "out.train.mgf").exists()
        assert (tmp_path / "out.validation.mgf").exists()
        assert (tmp_path / "out.test.mgf").exists()

    def test_error_lists_existing_files(self, tmp_path):
        """The error message should list which files already exist."""
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [(f"PEP{i}", [100.0], [1.0]) for i in range(20)],
        )
        output_root = str(tmp_path / "out")

        # Create only the train file.
        (tmp_path / "out.train.mgf").write_text("")

        with pytest.raises(FileExistsError, match="out.train.mgf"):
            create_datasets(mgf, output_root=output_root)

    def test_no_error_without_existing_files(self, tmp_path):
        """No error when output files do not exist."""
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [(f"PEP{i}", [100.0], [1.0]) for i in range(20)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        assert (tmp_path / "out.train.mgf").exists()
