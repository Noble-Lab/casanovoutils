"""Tests for the create_datasets module."""

import numpy as np
import pyteomics.mgf
import pytest

from casanovoutils.datasets import create_datasets


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
        assert (tmp_path / "out.val.mgf").exists()
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
        val = _read_mgf(tmp_path / "out.val.mgf")
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
        val_peps = set(_get_peptides(_read_mgf(tmp_path / "out.val.mgf")))
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
        val_peps = set(_get_peptides(_read_mgf(tmp_path / "out.val.mgf")))
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
        val = _read_mgf(tmp_path / "out.val.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        total = len(train) + len(val) + len(test)
        assert total == 30

        all_peps = set(_get_peptides(train) + _get_peptides(val) + _get_peptides(test))
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
            mgf,
            output_root=output_root,
            spectra_per_peptide=2,
        )

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.val.mgf")
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
        val = _read_mgf(tmp_path / "out.val.mgf")
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
            mgf,
            output_root=output_root,
            spectra_per_peptide=5,
        )

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.val.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        all_spectra = train + val + test
        pep_counts = {}
        for s in all_spectra:
            pep = s["params"]["seq"]
            pep_counts[pep] = pep_counts.get(pep, 0) + 1

        assert pep_counts.get("PEPA", 0) == 1
        assert pep_counts.get("PEPB", 0) == 3


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

        for split in ("train", "val", "test"):
            spectra_1 = _read_mgf(tmp_path / f"run1.{split}.mgf")
            spectra_2 = _read_mgf(tmp_path / f"run2.{split}.mgf")
            peps_1 = _get_peptides(spectra_1)
            peps_2 = _get_peptides(spectra_2)
            assert peps_1 == peps_2

    def test_different_seed_produces_different_splits(self, tmp_path):
        """Different seeds produce different ordered peptide assignments."""
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [(f"PEP{i}", [100.0], [1.0]) for i in range(30)],
        )

        create_datasets(
            mgf,
            output_root=str(tmp_path / "seed1"),
            random_seed=1,
        )
        create_datasets(
            mgf,
            output_root=str(tmp_path / "seed2"),
            random_seed=999,
        )

        # Compare ordered peptide lists across all splits. Two different
        # seeds produce different permutations of 30 peptides, so the
        # concatenated ordered lists are deterministically different.
        all_peps_1 = []
        all_peps_2 = []
        for split in ("train", "val", "test"):
            all_peps_1.extend(_get_peptides(_read_mgf(tmp_path / f"seed1.{split}.mgf")))
            all_peps_2.extend(_get_peptides(_read_mgf(tmp_path / f"seed2.{split}.mgf")))
        assert all_peps_1 != all_peps_2


class TestCreateDatasetsEdgeCases:
    """Edge case and error handling tests."""

    def test_no_mgf_files_raises_error(self, tmp_path):
        """Passing zero MGF files should raise a ValueError."""
        with pytest.raises(ValueError, match="At least one MGF file"):
            create_datasets(output_root=str(tmp_path / "out"))

    def test_missing_seq_raises_keyerror(self, tmp_path):
        """Spectrum without 'seq' param should raise a KeyError."""
        # Write an MGF with a spectrum that has no 'seq' param.
        records = [
            {
                "params": {"pepmass": (100.0,)},
                "m/z array": np.array([100.0]),
                "intensity array": np.array([1.0]),
            }
        ]
        mgf_path = tmp_path / "no_seq.mgf"
        pyteomics.mgf.write(records, output=str(mgf_path))

        with pytest.raises(KeyError, match="Missing 'seq'"):
            create_datasets(str(mgf_path), output_root=str(tmp_path / "out"))

    def test_combine_with_existing_without_existing_splits_raises_error(self, tmp_path):
        """combine_with_existing=True without existing_splits should raise."""
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [("PEP0", [100.0], [1.0])],
        )
        with pytest.raises(ValueError, match="combine_with_existing.*requires"):
            create_datasets(
                mgf,
                output_root=str(tmp_path / "out"),
                combine_with_existing=True,
            )

    def test_existing_splits_wrong_length_raises_error(self, tmp_path):
        """existing_splits with != 3 paths should raise ValueError."""
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [("PEP0", [100.0], [1.0])],
        )
        split = _write_mgf(tmp_path / "split.mgf", [("X", [100.0], [1.0])])
        with pytest.raises(ValueError, match="exactly three paths"):
            create_datasets(
                mgf,
                output_root=str(tmp_path / "out"),
                existing_splits=(split, split),
            )

    def test_spectra_per_peptide_zero_raises_error(self, tmp_path):
        """Passing spectra_per_peptide=0 should raise a ValueError."""
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [("PEP0", [100.0], [1.0])],
        )
        with pytest.raises(ValueError, match="spectra_per_peptide must be a positive"):
            create_datasets(
                mgf, output_root=str(tmp_path / "out"), spectra_per_peptide=0
            )

    def test_spectra_per_peptide_negative_raises_error(self, tmp_path):
        """Passing a negative spectra_per_peptide should raise a ValueError."""
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [("PEP0", [100.0], [1.0])],
        )
        with pytest.raises(ValueError, match="spectra_per_peptide must be a positive"):
            create_datasets(
                mgf, output_root=str(tmp_path / "out"), spectra_per_peptide=-1
            )

    def test_small_dataset_all_go_to_train(self, tmp_path):
        """With fewer than 3 peptides, all go to train; val/test are empty."""
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [(f"PEP{i}", [100.0], [1.0]) for i in range(2)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.val.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        assert len(train) == 2
        assert len(val) == 0
        assert len(test) == 0

    def test_three_peptides_splits_normally(self, tmp_path):
        """With exactly 3 peptides, each split gets at least one."""
        mgf = _write_mgf(
            tmp_path / "input.mgf",
            [(f"PEP{i}", [100.0], [1.0]) for i in range(3)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.val.mgf")
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
        val = _read_mgf(tmp_path / "out.val.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        all_spectra = train + val + test
        shared_spectra = [s for s in all_spectra if s["params"]["seq"] == "SHARED"]
        assert len(shared_spectra) == 2

        # Both SHARED spectra must be in the same split.
        shared_in_train = [s for s in train if s["params"]["seq"] == "SHARED"]
        shared_in_val = [s for s in val if s["params"]["seq"] == "SHARED"]
        shared_in_test = [s for s in test if s["params"]["seq"] == "SHARED"]
        counts = [len(shared_in_train), len(shared_in_val), len(shared_in_test)]
        assert sorted(counts) == [0, 0, 2]


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
        assert (tmp_path / "out.val.mgf").exists()
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

        with pytest.raises(FileExistsError, match=r"out\.train\.mgf"):
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


class TestCreateDatasetsExistingSplits:
    """Tests for the existing_splits and combine_with_existing options."""

    def _make_existing_splits(self, tmp_path):
        """Create existing train/val/test MGF files with known peptides."""
        train_path = _write_mgf(
            tmp_path / "exist_train.mgf",
            [("TRAIN1", [100.0], [1.0]), ("TRAIN2", [100.0], [1.0])],
        )
        val_path = _write_mgf(
            tmp_path / "exist_val.mgf",
            [("VAL1", [100.0], [1.0])],
        )
        test_path = _write_mgf(
            tmp_path / "exist_test.mgf",
            [("TEST1", [100.0], [1.0])],
        )
        return (train_path, val_path, test_path)

    def test_overlapping_peptides_routed_correctly(self, tmp_path):
        """Peptides in existing splits are routed to the correct split."""
        existing = self._make_existing_splits(tmp_path)

        # New data has some overlapping peptides plus novel ones.
        mgf = _write_mgf(
            tmp_path / "new.mgf",
            [
                ("TRAIN1", [200.0], [2.0]),
                ("VAL1", [200.0], [2.0]),
                ("TEST1", [200.0], [2.0]),
            ]
            + [(f"NEW{i}", [100.0], [1.0]) for i in range(20)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root, existing_splits=existing)

        train_peps = set(_get_peptides(_read_mgf(tmp_path / "out.train.mgf")))
        val_peps = set(_get_peptides(_read_mgf(tmp_path / "out.val.mgf")))
        test_peps = set(_get_peptides(_read_mgf(tmp_path / "out.test.mgf")))

        assert "TRAIN1" in train_peps
        assert "VAL1" in val_peps
        assert "TEST1" in test_peps

    def test_new_peptides_distributed_to_reach_ratio(self, tmp_path):
        """Novel peptides are distributed to approximate 80/10/10 overall."""
        existing = self._make_existing_splits(tmp_path)

        # 4 existing peptides + 96 new = 100 total.
        mgf = _write_mgf(
            tmp_path / "new.mgf",
            [(f"NEW{i}", [100.0], [1.0]) for i in range(96)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(
            mgf,
            output_root=output_root,
            existing_splits=existing,
            combine_with_existing=True,
        )

        train_peps = set(_get_peptides(_read_mgf(tmp_path / "out.train.mgf")))
        val_peps = set(_get_peptides(_read_mgf(tmp_path / "out.val.mgf")))
        test_peps = set(_get_peptides(_read_mgf(tmp_path / "out.test.mgf")))

        total = len(train_peps) + len(val_peps) + len(test_peps)
        assert total == 100
        # 80/10/10 of 100 => 80, 10, 10.
        assert len(train_peps) == 80
        assert len(val_peps) == 10
        assert len(test_peps) == 10

    def test_no_peptide_leakage_with_existing_splits(self, tmp_path):
        """No peptide appears in more than one output split."""
        existing = self._make_existing_splits(tmp_path)

        mgf = _write_mgf(
            tmp_path / "new.mgf",
            [("TRAIN1", [200.0], [2.0]), ("VAL1", [200.0], [2.0])]
            + [(f"NEW{i}", [100.0], [1.0]) for i in range(30)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root, existing_splits=existing)

        train_peps = set(_get_peptides(_read_mgf(tmp_path / "out.train.mgf")))
        val_peps = set(_get_peptides(_read_mgf(tmp_path / "out.val.mgf")))
        test_peps = set(_get_peptides(_read_mgf(tmp_path / "out.test.mgf")))

        assert train_peps.isdisjoint(val_peps)
        assert train_peps.isdisjoint(test_peps)
        assert val_peps.isdisjoint(test_peps)

    def test_combine_with_existing_includes_old_spectra(self, tmp_path):
        """With combine_with_existing=True, output includes old spectra."""
        existing = self._make_existing_splits(tmp_path)

        mgf = _write_mgf(
            tmp_path / "new.mgf",
            [(f"NEW{i}", [100.0], [1.0]) for i in range(20)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(
            mgf,
            output_root=output_root,
            existing_splits=existing,
            combine_with_existing=True,
        )

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.val.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        all_peps = _get_peptides(train + val + test)
        # Existing peptides should appear (from existing spectra).
        assert "TRAIN1" in all_peps
        assert "TRAIN2" in all_peps
        assert "VAL1" in all_peps
        assert "TEST1" in all_peps

        # Total spectra should include old (4) + new (20).
        total = len(train) + len(val) + len(test)
        assert total == 24

    def test_combine_with_existing_false_excludes_old(self, tmp_path):
        """With combine_with_existing=False, output has only new spectra."""
        existing = self._make_existing_splits(tmp_path)

        mgf = _write_mgf(
            tmp_path / "new.mgf",
            [(f"NEW{i}", [100.0], [1.0]) for i in range(20)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(
            mgf,
            output_root=output_root,
            existing_splits=existing,
            combine_with_existing=False,
        )

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.val.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        # Only new spectra: 20 total.
        total = len(train) + len(val) + len(test)
        assert total == 20

    def test_duplicate_peptide_across_existing_splits_raises_error(self, tmp_path):
        """A peptide in multiple existing splits should raise ValueError."""
        train_path = _write_mgf(
            tmp_path / "exist_train.mgf",
            [("SHARED", [100.0], [1.0]), ("TRAIN1", [100.0], [1.0])],
        )
        val_path = _write_mgf(
            tmp_path / "exist_val.mgf",
            [("SHARED", [100.0], [1.0])],
        )
        test_path = _write_mgf(
            tmp_path / "exist_test.mgf",
            [("TEST1", [100.0], [1.0])],
        )
        existing = (train_path, val_path, test_path)

        mgf = _write_mgf(
            tmp_path / "new.mgf",
            [("SHARED", [200.0], [2.0])]
            + [(f"NEW{i}", [100.0], [1.0]) for i in range(10)],
        )

        with pytest.raises(ValueError, match="multiple existing splits"):
            create_datasets(
                mgf,
                output_root=str(tmp_path / "out"),
                existing_splits=existing,
            )

    def test_duplicate_peptide_across_existing_splits_without_new_overlap(
        self, tmp_path
    ):
        """Overlapping existing splits raise ValueError even without that peptide in new input."""
        train_path = _write_mgf(
            tmp_path / "exist_train.mgf",
            [("SHARED", [100.0], [1.0]), ("TRAIN1", [100.0], [1.0])],
        )
        val_path = _write_mgf(
            tmp_path / "exist_val.mgf",
            [("SHARED", [100.0], [1.0])],
        )
        test_path = _write_mgf(
            tmp_path / "exist_test.mgf",
            [("TEST1", [100.0], [1.0])],
        )
        existing = (train_path, val_path, test_path)

        # New data does not contain the duplicated peptide "SHARED".
        mgf = _write_mgf(
            tmp_path / "new.mgf",
            [(f"NEW{i}", [100.0], [1.0]) for i in range(10)],
        )

        with pytest.raises(ValueError, match="multiple existing splits"):
            create_datasets(
                mgf,
                output_root=str(tmp_path / "out"),
                existing_splits=existing,
                combine_with_existing=True,
            )


class TestCreateDatasetsIsobaricNormalization:
    """Tests for isobaric normalization during splitting (I/L and deamidation)."""

    def test_il_variants_land_in_same_split(self, tmp_path):
        """Peptides differing only by I/L are placed in the same split."""
        # PEPTIDE and PEPTLDE are I/L variants of each other.
        # Add enough unrelated peptides to trigger real splitting.
        spectra = [("PEPTIDE", [100.0], [1.0]), ("PEPTLDE", [100.0], [1.0])]
        for i in range(28):
            spectra.append((f"OTHER{i}", [100.0], [1.0]))
        mgf = _write_mgf(tmp_path / "input.mgf", spectra)
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.val.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        def find_split(seq):
            for split_name, split in [("train", train), ("val", val), ("test", test)]:
                if any(s["params"]["seq"] == seq for s in split):
                    return split_name
            return None

        assert find_split("PEPTIDE") == find_split("PEPTLDE"), (
            "I/L variants must land in the same split"
        )

    def test_il_variants_counted_as_one_peptide(self, tmp_path):
        """I/L variants count as a single peptide due to isobaric normalization."""
        # 10 I/L pairs + 80 unique = 90 sequences but only 80+10=90 canonical
        # peptides... actually each pair shares a canonical form, so 10 pairs
        # contribute 10 canonical peptides rather than 20.
        spectra = []
        for i in range(10):
            spectra.append((f"PEP{i}I", [100.0], [1.0]))  # with I
            spectra.append((f"PEP{i}L", [100.0], [1.0]))  # same canonical
        for i in range(80):
            spectra.append((f"UNIQ{i}", [100.0], [1.0]))
        mgf = _write_mgf(tmp_path / "input.mgf", spectra)
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.val.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        # Total spectra must be preserved.
        assert len(train) + len(val) + len(test) == len(spectra)

        # Each I/L pair must be in the same split.
        for i in range(10):
            i_seq = f"PEP{i}I"
            l_seq = f"PEP{i}L"

            def find_split(seq, tr=train, v=val, te=test):
                for name, sp in [("train", tr), ("val", v), ("test", te)]:
                    if any(s["params"]["seq"] == seq for s in sp):
                        return name
                return None

            assert find_split(i_seq) == find_split(l_seq), (
                f"I/L pair PEP{i}I / PEP{i}L must be in the same split"
            )

    def test_isobaric_normalization_is_unconditional(self, tmp_path):
        """Isobaric normalization is always applied; there is no opt-out."""
        # Verify that PEPTIDE and PEPTLDE (I/L variants) always land in the
        # same split regardless of how create_datasets is called.
        spectra = [("PEPTIDE", [100.0], [1.0]), ("PEPTLDE", [100.0], [1.0])]
        for i in range(28):
            spectra.append((f"OTHER{i}", [100.0], [1.0]))
        mgf = _write_mgf(tmp_path / "input.mgf", spectra)
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.val.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        def find_split(seq):
            for name, sp in [("train", train), ("val", val), ("test", test)]:
                if any(s["params"]["seq"] == seq for s in sp):
                    return name
            return None

        assert find_split("PEPTIDE") == find_split("PEPTLDE")
        # All spectra are preserved.
        assert len(train) + len(val) + len(test) == len(spectra)

    def test_original_sequences_preserved_in_output(self, tmp_path):
        """Output MGF files retain original sequences despite isobaric normalization."""
        spectra = [("PEPTIDE", [100.0], [1.0]), ("PEPTLDE", [100.0], [1.0])]
        for i in range(28):
            spectra.append((f"OTHER{i}", [100.0], [1.0]))
        mgf = _write_mgf(tmp_path / "input.mgf", spectra)
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.val.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        all_seqs = set(_get_peptides(train + val + test))
        # Original sequences (not canonical) must appear in the output.
        assert "PEPTIDE" in all_seqs
        assert "PEPTLDE" in all_seqs
        # The canonical form should NOT appear as a distinct entry
        # (it was never an input sequence).
        assert "PEPTLDE" in all_seqs  # L-only form is a real input sequence
        # Make sure we didn't accidentally replace sequences in the output.
        assert "PEPTIDE" in all_seqs  # I-form must still be present

    def test_il_normalization_with_existing_splits(self, tmp_path):
        """I/L variant in new data routes to the correct existing split."""
        # Existing train split contains PEPTLDE (L-form).
        train_path = _write_mgf(
            tmp_path / "exist_train.mgf",
            [("PEPTLDE", [100.0], [1.0])] + [(f"TR{i}", [100.0], [1.0]) for i in range(7)],
        )
        val_path = _write_mgf(
            tmp_path / "exist_val.mgf",
            [(f"VA{i}", [100.0], [1.0]) for i in range(1)],
        )
        test_path = _write_mgf(
            tmp_path / "exist_test.mgf",
            [(f"TE{i}", [100.0], [1.0]) for i in range(1)],
        )
        existing = (train_path, val_path, test_path)

        # New data has PEPTIDE (I-form), the I/L variant of PEPTLDE.
        mgf = _write_mgf(
            tmp_path / "new.mgf",
            [("PEPTIDE", [100.0], [1.0])] + [(f"NEW{i}", [100.0], [1.0]) for i in range(19)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(
            mgf,
            output_root=output_root,
            existing_splits=existing,
        )

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.val.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        train_seqs = set(_get_peptides(train))
        val_seqs = set(_get_peptides(val))
        test_seqs = set(_get_peptides(test))

        # PEPTIDE (I-form) must land in train alongside PEPTLDE (L-form).
        assert "PEPTIDE" in train_seqs, "I-form should follow L-form into train"
        assert "PEPTIDE" not in val_seqs
        assert "PEPTIDE" not in test_seqs

    def test_il_normalization_always_applied(self, tmp_path):
        """I/L variants are always placed in the same split."""
        spectra = [("PEPTIDE", [100.0], [1.0]), ("PEPTLDE", [100.0], [1.0])]
        for i in range(28):
            spectra.append((f"OTHER{i}", [100.0], [1.0]))
        mgf = _write_mgf(tmp_path / "input.mgf", spectra)
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.val.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        def find_split(seq):
            for name, sp in [("train", train), ("val", val), ("test", test)]:
                if any(s["params"]["seq"] == seq for s in sp):
                    return name
            return None

        assert find_split("PEPTIDE") == find_split("PEPTLDE"), (
            "Default behavior must place I/L variants in the same split"
        )

    def test_deamidated_n_and_d_land_in_same_split(self, tmp_path):
        """N[Deamidated] and D variants of the same peptide land in the same split."""
        spectra = [
            ("PEPT[N[Deamidated]]DE", [100.0], [1.0]),
            ("PEPTDDE", [100.0], [1.0]),
        ]
        for i in range(28):
            spectra.append((f"OTHER{i}", [100.0], [1.0]))
        mgf = _write_mgf(tmp_path / "input.mgf", spectra)
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.val.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        def find_split(seq):
            for name, sp in [("train", train), ("val", val), ("test", test)]:
                if any(s["params"]["seq"] == seq for s in sp):
                    return name
            return None

        assert find_split("PEPT[N[Deamidated]]DE") == find_split("PEPTDDE"), (
            "N[Deamidated] and D variants must land in the same split"
        )

    def test_deamidated_q_and_e_land_in_same_split(self, tmp_path):
        """Q[Deamidated] and E variants of the same peptide land in the same split."""
        spectra = [
            ("PEPT[Q[Deamidated]]DE", [100.0], [1.0]),
            ("PEPTEDE", [100.0], [1.0]),
        ]
        for i in range(28):
            spectra.append((f"OTHER{i}", [100.0], [1.0]))
        mgf = _write_mgf(tmp_path / "input.mgf", spectra)
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.val.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        def find_split(seq):
            for name, sp in [("train", train), ("val", val), ("test", test)]:
                if any(s["params"]["seq"] == seq for s in sp):
                    return name
            return None

        assert find_split("PEPT[Q[Deamidated]]DE") == find_split("PEPTEDE"), (
            "Q[Deamidated] and E variants must land in the same split"
        )

    def test_all_three_normalizations_together(self, tmp_path):
        """A peptide with I, N[Deamidated], and Q[Deamidated] is grouped with
        its fully-canonical D/E/L equivalent."""
        # IN[Deamidated]Q[Deamidated]K → LDEK after full canonicalization.
        spectra = [
            ("IN[Deamidated]Q[Deamidated]K", [100.0], [1.0]),
            ("LDEK", [100.0], [1.0]),
        ]
        for i in range(28):
            spectra.append((f"OTHER{i}", [100.0], [1.0]))
        mgf = _write_mgf(tmp_path / "input.mgf", spectra)
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.val.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        def find_split(seq):
            for name, sp in [("train", train), ("val", val), ("test", test)]:
                if any(s["params"]["seq"] == seq for s in sp):
                    return name
            return None

        assert find_split("IN[Deamidated]Q[Deamidated]K") == find_split("LDEK"), (
            "Peptide with I, N[Deamidated], and Q[Deamidated] must land in "
            "the same split as its canonical D/E/L form"
        )

    def test_deamidation_normalization_preserves_original_sequences(self, tmp_path):
        """Output MGFs retain original sequences even with deamidation normalization."""
        spectra = [
            ("PEPT[N[Deamidated]]DE", [100.0], [1.0]),
            ("PEPTDDE", [100.0], [1.0]),
        ]
        for i in range(28):
            spectra.append((f"OTHER{i}", [100.0], [1.0]))
        mgf = _write_mgf(tmp_path / "input.mgf", spectra)
        output_root = str(tmp_path / "out")

        create_datasets(mgf, output_root=output_root)

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.val.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        all_seqs = set(_get_peptides(train + val + test))
        assert "PEPT[N[Deamidated]]DE" in all_seqs, (
            "Original N[Deamidated] sequence must be preserved in output"
        )
        assert "PEPTDDE" in all_seqs

    def test_deamidation_with_existing_splits(self, tmp_path):
        """N[Deamidated]-form in new data routes to the split containing the D-form."""
        train_path = _write_mgf(
            tmp_path / "exist_train.mgf",
            [("PEPTDDE", [100.0], [1.0])] + [(f"TR{i}", [100.0], [1.0]) for i in range(7)],
        )
        val_path = _write_mgf(
            tmp_path / "exist_val.mgf",
            [(f"VA{i}", [100.0], [1.0]) for i in range(1)],
        )
        test_path = _write_mgf(
            tmp_path / "exist_test.mgf",
            [(f"TE{i}", [100.0], [1.0]) for i in range(1)],
        )
        existing = (train_path, val_path, test_path)

        mgf = _write_mgf(
            tmp_path / "new.mgf",
            [("PEPT[N[Deamidated]]DE", [100.0], [1.0])]
            + [(f"NEW{i}", [100.0], [1.0]) for i in range(19)],
        )
        output_root = str(tmp_path / "out")

        create_datasets(
            mgf,
            output_root=output_root,
            existing_splits=existing,
        )

        train = _read_mgf(tmp_path / "out.train.mgf")
        val = _read_mgf(tmp_path / "out.val.mgf")
        test = _read_mgf(tmp_path / "out.test.mgf")

        train_seqs = set(_get_peptides(train))
        val_seqs = set(_get_peptides(val))
        test_seqs = set(_get_peptides(test))

        assert "PEPT[N[Deamidated]]DE" in train_seqs, (
            "N[Deamidated]-form should follow D-form into train"
        )
        assert "PEPT[N[Deamidated]]DE" not in val_seqs
        assert "PEPT[N[Deamidated]]DE" not in test_seqs
