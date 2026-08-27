"""Tests for the visualize_errors command."""

import pathlib

import numpy as np
import pytest

from casanovoutils.visualize_errors import (
    _delta_mz,
    _load_mgf_peaks,
    _make_mirror_plot,
    _make_spectrum,
    _parse_mgf_idx,
    _write_html,
    visualize_errors,
)

# ---------------------------------------------------------------------------
# Paths to small test fixtures extracted from the real Casanovo run
# ---------------------------------------------------------------------------

DATA_DIR = pathlib.Path(__file__).parent / "data"
TEST_MGF = DATA_DIR / "visualize_errors_test.mgf"
TEST_MZTAB = DATA_DIR / "visualize_errors_test.mztab"


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------


def test_delta_mz_zero():
    da, ppm = _delta_mz(500.0, 500.0)
    assert da == pytest.approx(0.0)
    assert ppm == pytest.approx(0.0)


def test_delta_mz_positive():
    da, ppm = _delta_mz(500.005, 500.0)
    assert da == pytest.approx(0.005)
    assert ppm == pytest.approx(10.0)


def test_delta_mz_zero_calc():
    da, ppm = _delta_mz(500.0, 0.0)
    assert da == pytest.approx(500.0)
    assert np.isnan(ppm)


def test_parse_mgf_idx_valid():
    assert _parse_mgf_idx("ms_run[1]:index=7") == 7


def test_parse_mgf_idx_zero():
    assert _parse_mgf_idx("ms_run[1]:index=0") == 0


def test_parse_mgf_idx_no_index():
    assert _parse_mgf_idx("ms_run[1]:scan=42") is None


def test_parse_mgf_idx_malformed():
    assert _parse_mgf_idx("ms_run[1]:index=abc") is None


def test_load_mgf_peaks_count():
    peaks = _load_mgf_peaks(TEST_MGF)
    assert len(peaks) == 20


def test_load_mgf_peaks_have_arrays():
    peaks = _load_mgf_peaks(TEST_MGF)
    spec = peaks[0]
    assert "m/z array" in spec
    assert "intensity array" in spec
    assert len(spec["m/z array"]) > 0


def test_make_spectrum_returns_msms():
    from spectrum_utils.spectrum import MsmsSpectrum

    peaks = _load_mgf_peaks(TEST_MGF)
    spec = _make_spectrum(peaks[0], "test_id")
    assert isinstance(spec, MsmsSpectrum)
    assert spec.precursor_charge > 0
    assert spec.precursor_mz > 0


def test_make_mirror_plot_returns_figure():
    import matplotlib.pyplot as plt

    peaks = _load_mgf_peaks(TEST_MGF)
    fig = _make_mirror_plot(
        spectrum_dict=peaks[0],
        predicted_seq="PEPTIDE",
        ground_truth_seq="PEPTIDER",
        score=0.95,
        exp_mz=450.2,
        calc_mz=450.0,
        charge=2,
        scan="42",
        rank=1,
        fragment_tol=0.05,
        fragment_tol_mode="Da",
        ion_types="by",
        neutral_losses=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_make_mirror_plot_bad_seq_does_not_raise():
    """Unannotatable sequences should log a warning but not raise."""
    import matplotlib.pyplot as plt

    peaks = _load_mgf_peaks(TEST_MGF)
    fig = _make_mirror_plot(
        spectrum_dict=peaks[0],
        predicted_seq="[BADMOD???]-PEPTIDE",
        ground_truth_seq="PEPTIDER",
        score=0.5,
        exp_mz=450.0,
        calc_mz=450.0,
        charge=2,
        scan="1",
        rank=1,
        fragment_tol=0.05,
        fragment_tol_mode="Da",
        ion_types="by",
        neutral_losses=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Integration test for the full command
# ---------------------------------------------------------------------------


def test_visualize_errors_creates_pngs(tmp_path):
    visualize_errors(
        mgf_file=TEST_MGF,
        mztab_file=TEST_MZTAB,
        output_dir=tmp_path / "out",
        k=3,
    )
    pngs = sorted((tmp_path / "out").glob("rank_*.png"))
    # At least 1 incorrect prediction expected in the 16-PSM test set
    assert len(pngs) >= 1
    assert len(pngs) <= 3


def test_visualize_errors_creates_log(tmp_path):
    visualize_errors(
        mgf_file=TEST_MGF,
        mztab_file=TEST_MZTAB,
        output_dir=tmp_path / "out",
        k=3,
    )
    log = tmp_path / "out" / "visualize_errors.log"
    assert log.exists()


def test_visualize_errors_overwrite_false_raises(tmp_path):
    out = tmp_path / "out"
    visualize_errors(mgf_file=TEST_MGF, mztab_file=TEST_MZTAB, output_dir=out, k=1)
    with pytest.raises(FileExistsError, match="rank_.*png"):
        visualize_errors(mgf_file=TEST_MGF, mztab_file=TEST_MZTAB, output_dir=out, k=1)


def test_visualize_errors_overwrite_true_succeeds(tmp_path):
    out = tmp_path / "out"
    visualize_errors(mgf_file=TEST_MGF, mztab_file=TEST_MZTAB, output_dir=out, k=1)
    # Should not raise
    visualize_errors(
        mgf_file=TEST_MGF,
        mztab_file=TEST_MZTAB,
        output_dir=out,
        k=1,
        overwrite=True,
    )


def test_visualize_errors_distinct_il_produces_output(tmp_path):
    """distinct_il=True should run without error and still produce plots."""
    visualize_errors(
        mgf_file=TEST_MGF,
        mztab_file=TEST_MZTAB,
        output_dir=tmp_path / "out",
        k=5,
        distinct_il=True,
    )
    pngs = list((tmp_path / "out").glob("rank_*.png"))
    # With distinct_il=True, I/L swaps count as errors, so we expect at least
    # as many (possibly more) incorrect predictions as without.
    assert len(pngs) >= 1


def test_distinct_il_produces_at_least_as_many_errors(tmp_path):
    """distinct_il=True should find >= as many errors as the default (I/L equiv)."""
    out_equiv = tmp_path / "il_equiv"
    out_distinct = tmp_path / "il_distinct"
    visualize_errors(
        mgf_file=TEST_MGF,
        mztab_file=TEST_MZTAB,
        output_dir=out_equiv,
        k=20,
        distinct_il=False,
    )
    visualize_errors(
        mgf_file=TEST_MGF,
        mztab_file=TEST_MZTAB,
        output_dir=out_distinct,
        k=20,
        distinct_il=True,
    )
    n_equiv = len(list(out_equiv.glob("rank_*.png")))
    n_distinct = len(list(out_distinct.glob("rank_*.png")))
    assert n_distinct >= n_equiv


# ---------------------------------------------------------------------------
# HTML output tests
# ---------------------------------------------------------------------------


def test_visualize_errors_creates_main_html(tmp_path):
    out = tmp_path / "out"
    visualize_errors(mgf_file=TEST_MGF, mztab_file=TEST_MZTAB, output_dir=out, k=3)
    assert (out / "out.html").exists()


def test_main_html_links_to_pngs(tmp_path):
    out = tmp_path / "out"
    visualize_errors(mgf_file=TEST_MGF, mztab_file=TEST_MZTAB, output_dir=out, k=3)
    html_text = (out / "out.html").read_text(encoding="utf-8")
    pngs = sorted(out.glob("rank_*.png"))
    for png in pngs:
        assert png.name in html_text


_DUMMY_STATS = dict(
    mgf_file="input.mgf",
    mztab_file="input.mztab",
    n_total=100,
    n_with_predictions=80,
    n_correct=60,
)


def test_write_html_standalone(tmp_path):
    """_write_html creates <stem>.html from a list of paths."""
    d = tmp_path / "myrun"
    d.mkdir()
    pngs = [d / "rank_0001_scan_6.png", d / "rank_0002_scan_91.png"]
    for p in pngs:
        p.touch()
    _write_html(d, pngs, **_DUMMY_STATS)
    assert (d / "myrun.html").exists()
    html = (d / "myrun.html").read_text(encoding="utf-8")
    assert "rank_0001_scan_6.png" in html
    assert "rank_0002_scan_91.png" in html


def test_main_html_contains_stats(tmp_path):
    """<stem>.html also includes the summary table."""
    out = tmp_path / "out"
    visualize_errors(mgf_file=TEST_MGF, mztab_file=TEST_MZTAB, output_dir=out, k=3)
    text = (out / "out.html").read_text(encoding="utf-8")
    assert "Total spectra" in text
    assert "With Casanovo predictions" in text
    assert "Correct predictions" in text
