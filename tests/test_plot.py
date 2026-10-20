"""Tests for the spectrum plotting commands."""

import logging

import matplotlib

matplotlib.use("Agg")

import polars as pl
import pytest

from casanovoutils import plot
from casanovoutils.constants import Constants


def _write_mgf(path, charge="CHARGE=2+\n", title="TITLE=s0\n", seq=""):
    path.write_text(
        "BEGIN IONS\n"
        f"{title}"
        f"{seq}"
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


def test_spectrum_titleless_mgf(tmp_path):
    """Spectra are read by position, so TITLE-less MGFs work (MassIVE-KB)."""
    mgf = tmp_path / "notitle.mgf"
    _write_mgf(mgf, title="")
    out = tmp_path / "spectrum.png"
    plot.spectrum(_mztab("PEPTIDEK"), str(mgf), 0, out=str(out))
    assert out.is_file()


def test_mirror_plot(tmp_path):
    mgf = tmp_path / "test.mgf"
    _write_mgf(mgf)
    out = tmp_path / "mirror.png"
    plot.mirror(_mztab("PEPTIDEK"), str(mgf), 0, ground_truth="PEPTIDER", out=str(out))
    assert out.is_file()


def test_mirror_ground_truth_from_seq(tmp_path):
    """When ground_truth is omitted, it is read from the MGF SEQ= field."""
    mgf = tmp_path / "seq.mgf"
    _write_mgf(mgf, seq="SEQ=PEPTIDER\n")
    out = tmp_path / "mirror.png"
    plot.mirror(_mztab("PEPTIDEK"), str(mgf), 0, out=str(out))
    assert out.is_file()


def test_mirror_no_ground_truth_raises(tmp_path):
    mgf = tmp_path / "test.mgf"
    _write_mgf(mgf)  # no SEQ=
    with pytest.raises(ValueError, match="SEQ="):
        plot.mirror(_mztab("PEPTIDEK"), str(mgf), 0, out=str(tmp_path / "m.png"))


def test_peptide_at_index_missing():
    with pytest.raises(ValueError):
        plot._peptide_at_index(_mztab("PEPTIDEK"), 5)


def test_peptide_at_index_multiple_warns(caplog):
    df = pl.DataFrame(
        {
            "mztab_spectra_ref": ["ms_run[1]:index=0", "ms_run[1]:index=0"],
            Constants.proforma_column: ["PEPTIDEK", "PEPTIDER"],
        }
    )
    with caplog.at_level(logging.WARNING):
        peptide = plot._peptide_at_index(df, 0)
    assert peptide == "PEPTIDEK"
    assert "Multiple PSMs" in caplog.text


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


def test_remove_precursor_peak(tmp_path):
    mgf = tmp_path / "prec.mgf"
    # Charge 1 precursor at m/z 500.0, with a peak sitting on it.
    mgf.write_text(
        "BEGIN IONS\n"
        "TITLE=s0\n"
        "PEPMASS=500.0\n"
        "CHARGE=1+\n"
        "100.0 1.0\n"
        "200.0 2.0\n"
        "500.0 9.0\n"
        "END IONS\n"
    )
    spectrum_dict = plot._spectrum_at_index(str(mgf), 0)

    kept = plot._annotated_spectrum(
        spectrum_dict, "PEPTIDEK", "0", 0.5, "Da", "by", None, 0, None, "Da"
    )
    assert any(abs(mz - 500.0) < 0.01 for mz in kept.mz)

    removed = plot._annotated_spectrum(
        spectrum_dict, "PEPTIDEK", "0", 0.5, "Da", "by", None, 0, 1.5, "Da"
    )
    assert not any(abs(mz - 500.0) < 0.01 for mz in removed.mz)
