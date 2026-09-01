"""Tests for the spectrum plotting commands."""

import matplotlib

matplotlib.use("Agg")

import polars as pl
import pytest

from casanovoutils import plot
from casanovoutils.constants import Constants


def _write_mgf(path, charge="CHARGE=2+\n"):
    path.write_text(
        "BEGIN IONS\n"
        "TITLE=s0\n"
        "PEPMASS=500.0\n"
        f"{charge}"
        "100.0 1.0\n"
        "200.0 2.0\n"
        "300.0 3.0\n"
        "END IONS\n"
    )


def _mztab(peptide, index=0):
    return pl.DataFrame(
        {
            "mztab_spectra_ref": [f"ms_run[1]:index={index}"],
            Constants.proforma_column: [peptide],
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


def test_read_spectrum_no_charge_warns(tmp_path):
    mgf = tmp_path / "nocharge.mgf"
    _write_mgf(mgf, charge="")
    with pytest.warns(UserWarning, match="No precursor charge"):
        spec = plot._read_spectrum(str(mgf), 0)
    assert spec.precursor_charge == 0


def test_peptide_at_index_multiple_warns():
    df = pl.DataFrame(
        {
            "mztab_spectra_ref": ["ms_run[1]:index=0", "ms_run[1]:index=0"],
            Constants.proforma_column: ["PEPTIDEK", "PEPTIDER"],
        }
    )
    with pytest.warns(UserWarning, match="Multiple PSMs"):
        peptide = plot._peptide_at_index(df, 0)
    assert peptide == "PEPTIDEK"


def test_spectrum_title_and_annotation_params(tmp_path):
    mgf = tmp_path / "test.mgf"
    _write_mgf(mgf)
    out = tmp_path / "spectrum.png"
    plot.spectrum(
        _mztab("PEPTIDEK"),
        str(mgf),
        0,
        out=str(out),
        max_charge=1,
        max_isotope=1,
        title="My spectrum",
    )
    assert out.is_file()


def test_mirror_title(tmp_path):
    mgf = tmp_path / "test.mgf"
    _write_mgf(mgf)
    out = tmp_path / "mirror.png"
    plot.mirror(
        _mztab("PEPTIDEK"),
        str(mgf),
        0,
        _mztab("PEPTIDER"),
        out=str(out),
        title="My mirror",
    )
    assert out.is_file()
