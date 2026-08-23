"""Tests for the filter_spectra command."""

import numpy as np
import pyteomics.mgf
import pytest
import yaml

from casanovoutils.filter_spectra import filter_spectra


def _write_config(path, residues=None, min_peaks=2, max_charge=5):
    """Write a minimal Casanovo YAML config."""
    cfg = {
        "min_peaks": min_peaks,
        "max_charge": max_charge,
    }
    if residues is not None:
        cfg["residues"] = residues
    with open(path, "w") as fh:
        yaml.dump(cfg, fh)
    return str(path)


def _write_mgf(path, spectra):
    """Write spectra as an MGF file.

    Parameters
    ----------
    spectra : list[dict]
        Each dict may contain keys: seq, charge, mz, intensity.
    """
    records = []
    for s in spectra:
        params = {"pepmass": (500.0,)}
        if "seq" in s:
            params["seq"] = s["seq"]
        if "charge" in s:
            params["charge"] = [s["charge"]]
        records.append(
            {
                "params": params,
                "m/z array": np.array(s.get("mz", [100.0, 200.0, 300.0])),
                "intensity array": np.array(s.get("intensity", [1.0, 1.0, 1.0])),
            }
        )
    pyteomics.mgf.write(records, output=str(path))
    return str(path)


def _read_seqs(path):
    spectra = list(pyteomics.mgf.read(str(path), use_index=False))
    return [s["params"].get("seq") for s in spectra]


# Three peaks satisfies min_peaks=2; we use this as the "good" default.
_GOOD = {
    "seq": "PEPTIDE",
    "charge": 2,
    "mz": [100.0, 200.0, 300.0],
    "intensity": [1.0, 1.0, 1.0],
}


class TestFilterSpectraMissingSeq:
    def test_missing_seq_filtered(self, tmp_path):
        mgf_in = tmp_path / "in.mgf"
        cfg = tmp_path / "config.yaml"
        _write_mgf(
            mgf_in,
            [{"charge": 2, "mz": [100.0, 200.0, 300.0], "intensity": [1.0, 1.0, 1.0]}],
        )
        _write_config(cfg)
        filter_spectra(
            mgf_in, cfg, output_root="out", output_dir=tmp_path, overwrite=True
        )
        assert _read_seqs(tmp_path / "out.mgf") == []

    def test_empty_seq_filtered(self, tmp_path):
        mgf_in = tmp_path / "in.mgf"
        cfg = tmp_path / "config.yaml"
        _write_mgf(
            mgf_in,
            [
                {
                    "seq": "",
                    "charge": 2,
                    "mz": [100.0, 200.0, 300.0],
                    "intensity": [1.0, 1.0, 1.0],
                }
            ],
        )
        _write_config(cfg)
        filter_spectra(
            mgf_in, cfg, output_root="out", output_dir=tmp_path, overwrite=True
        )
        assert _read_seqs(tmp_path / "out.mgf") == []


class TestFilterSpectraInvalidTokens:
    def test_unknown_token_filtered(self, tmp_path):
        mgf_in = tmp_path / "in.mgf"
        cfg = tmp_path / "config.yaml"
        # B is not a standard amino acid and not in the config residues
        _write_mgf(
            mgf_in,
            [
                {
                    "seq": "PEPBIDE",
                    "charge": 2,
                    "mz": [100.0, 200.0, 300.0],
                    "intensity": [1.0, 1.0, 1.0],
                }
            ],
        )
        _write_config(cfg)
        filter_spectra(
            mgf_in, cfg, output_root="out", output_dir=tmp_path, overwrite=True
        )
        assert _read_seqs(tmp_path / "out.mgf") == []

    def test_known_modification_passes(self, tmp_path):
        mgf_in = tmp_path / "in.mgf"
        cfg = tmp_path / "config.yaml"
        _write_mgf(
            mgf_in,
            [
                {
                    "seq": "PEPTM[Oxidation]IDE",
                    "charge": 2,
                    "mz": [100.0, 200.0, 300.0],
                    "intensity": [1.0, 1.0, 1.0],
                }
            ],
        )
        _write_config(cfg, residues={"M[Oxidation]": 147.035})
        filter_spectra(
            mgf_in, cfg, output_root="out", output_dir=tmp_path, overwrite=True
        )
        assert _read_seqs(tmp_path / "out.mgf") == ["PEPTM[Oxidation]IDE"]


class TestFilterSpectraInvalidCharge:
    def test_missing_charge_filtered(self, tmp_path):
        mgf_in = tmp_path / "in.mgf"
        cfg = tmp_path / "config.yaml"
        _write_mgf(
            mgf_in,
            [
                {
                    "seq": "PEPTIDE",
                    "mz": [100.0, 200.0, 300.0],
                    "intensity": [1.0, 1.0, 1.0],
                }
            ],
        )
        _write_config(cfg)
        filter_spectra(
            mgf_in, cfg, output_root="out", output_dir=tmp_path, overwrite=True
        )
        assert _read_seqs(tmp_path / "out.mgf") == []

    def test_charge_exceeds_max_filtered(self, tmp_path):
        mgf_in = tmp_path / "in.mgf"
        cfg = tmp_path / "config.yaml"
        _write_mgf(
            mgf_in,
            [
                {
                    "seq": "PEPTIDE",
                    "charge": 6,
                    "mz": [100.0, 200.0, 300.0],
                    "intensity": [1.0, 1.0, 1.0],
                }
            ],
        )
        _write_config(cfg, max_charge=5)
        filter_spectra(
            mgf_in, cfg, output_root="out", output_dir=tmp_path, overwrite=True
        )
        assert _read_seqs(tmp_path / "out.mgf") == []

    def test_charge_at_max_passes(self, tmp_path):
        mgf_in = tmp_path / "in.mgf"
        cfg = tmp_path / "config.yaml"
        _write_mgf(
            mgf_in,
            [
                {
                    "seq": "PEPTIDE",
                    "charge": 5,
                    "mz": [100.0, 200.0, 300.0],
                    "intensity": [1.0, 1.0, 1.0],
                }
            ],
        )
        _write_config(cfg, max_charge=5)
        filter_spectra(
            mgf_in, cfg, output_root="out", output_dir=tmp_path, overwrite=True
        )
        assert _read_seqs(tmp_path / "out.mgf") == ["PEPTIDE"]


class TestFilterSpectraTooFewPeaks:
    def test_too_few_peaks_filtered(self, tmp_path):
        mgf_in = tmp_path / "in.mgf"
        cfg = tmp_path / "config.yaml"
        _write_mgf(
            mgf_in, [{"seq": "PEPTIDE", "charge": 2, "mz": [100.0], "intensity": [1.0]}]
        )
        _write_config(cfg, min_peaks=2)
        filter_spectra(
            mgf_in, cfg, output_root="out", output_dir=tmp_path, overwrite=True
        )
        assert _read_seqs(tmp_path / "out.mgf") == []

    def test_exactly_min_peaks_passes(self, tmp_path):
        mgf_in = tmp_path / "in.mgf"
        cfg = tmp_path / "config.yaml"
        _write_mgf(
            mgf_in,
            [
                {
                    "seq": "PEPTIDE",
                    "charge": 2,
                    "mz": [100.0, 200.0],
                    "intensity": [1.0, 1.0],
                }
            ],
        )
        _write_config(cfg, min_peaks=2)
        filter_spectra(
            mgf_in, cfg, output_root="out", output_dir=tmp_path, overwrite=True
        )
        assert _read_seqs(tmp_path / "out.mgf") == ["PEPTIDE"]


class TestFilterSpectraPassThrough:
    def test_valid_spectrum_passes(self, tmp_path):
        mgf_in = tmp_path / "in.mgf"
        cfg = tmp_path / "config.yaml"
        _write_mgf(mgf_in, [_GOOD])
        _write_config(cfg)
        filter_spectra(
            mgf_in, cfg, output_root="out", output_dir=tmp_path, overwrite=True
        )
        assert _read_seqs(tmp_path / "out.mgf") == ["PEPTIDE"]

    def test_mixed_spectra_only_valid_pass(self, tmp_path):
        mgf_in = tmp_path / "in.mgf"
        cfg = tmp_path / "config.yaml"
        _write_mgf(
            mgf_in,
            [
                _GOOD,
                {
                    "seq": "PEPBIDE",
                    "charge": 2,
                    "mz": [100.0, 200.0, 300.0],
                    "intensity": [1.0, 1.0, 1.0],
                },
                {
                    "seq": "VALIDPEP",
                    "charge": 3,
                    "mz": [100.0, 200.0, 300.0],
                    "intensity": [1.0, 1.0, 1.0],
                },
            ],
        )
        _write_config(cfg)
        filter_spectra(
            mgf_in, cfg, output_root="out", output_dir=tmp_path, overwrite=True
        )
        assert _read_seqs(tmp_path / "out.mgf") == ["PEPTIDE", "VALIDPEP"]

    def test_overwrite_false_raises_on_existing(self, tmp_path):
        mgf_in = tmp_path / "in.mgf"
        cfg = tmp_path / "config.yaml"
        _write_mgf(mgf_in, [_GOOD])
        _write_config(cfg)
        filter_spectra(
            mgf_in, cfg, output_root="out", output_dir=tmp_path, overwrite=True
        )
        with pytest.raises(FileExistsError):
            filter_spectra(
                mgf_in, cfg, output_root="out", output_dir=tmp_path, overwrite=False
            )
