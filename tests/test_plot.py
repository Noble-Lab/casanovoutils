"""Tests for the spectrum plotting commands."""

import matplotlib

matplotlib.use("Agg")

import polars as pl
import pytest

from casanovoutils import plot


def _write_mgf(path):
    path.write_text(
        "BEGIN IONS\n"
        "TITLE=s0\n"
        "PEPMASS=500.0\n"
        "CHARGE=2+\n"
        "100.0 1.0\n"
        "200.0 2.0\n"
        "300.0 3.0\n"
        "END IONS\n"
    )


def _mztab(peptide):
    return pl.DataFrame(
        {
            "mztab_spectra_ref": ["ms_run[1]:index=0"],
            plot._PROFORMA_COLUMN: [peptide],
        }
    )


def test_spectrum_plot(tmp_path):
    mgf = tmp_path / "test.mgf"
    _write_mgf(mgf)
    out = tmp_path / "spectrum.png"
    plot.spectrum(_mztab("PEPTIDEK"), str(mgf), 0, out=str(out))
    assert out.is_file()


def test_mirror_plot(tmp_path):
    mgf = tmp_path / "test.mgf"
    _write_mgf(mgf)
    out = tmp_path / "mirror.png"
    plot.mirror(_mztab("PEPTIDEK"), str(mgf), 0, _mztab("PEPTIDER"), out=str(out))
    assert out.is_file()


def test_peptide_at_index_missing():
    with pytest.raises(ValueError):
        plot._peptide_at_index(_mztab("PEPTIDEK"), 5)
