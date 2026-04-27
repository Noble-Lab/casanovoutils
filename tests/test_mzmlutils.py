from unittest.mock import MagicMock, patch

import numpy as np
import pyteomics.mgf
import pytest

from casanovoutils.mzmlutils import _to_mgf_spectrum, _write_spectra, sample_mzml, sample_spectra


def make_spectrum(idx=0):
    rng = np.random.default_rng(idx)
    n = 10
    mz = np.sort(rng.random(n).astype(np.float32) * 1000)
    intensity = rng.random(n).astype(np.float32) * 1e4
    return {
        "ms level": 2,
        "m/z array": mz,
        "intensity array": intensity,
        "id": f"scan={idx + 1}",
    }


def make_spectra(n):
    return [make_spectrum(i) for i in range(n)]


def _mzml_cm(spectra):
    """Return a mock context manager that yields spectra on iteration."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=iter(spectra))
    cm.__exit__ = MagicMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# sample_mzml — basic sampling
# ---------------------------------------------------------------------------


def test_returns_proportion_of_spectra():
    # 100 spectra, one buffer → round(0.2 * 100) = 20
    spectra = make_spectra(100)
    with patch("casanovoutils.mzmlutils.pyteomics.mzml.MzML") as mock_cls:
        mock_cls.return_value = _mzml_cm(spectra)
        result = sample_mzml("dummy.mzML", k=0.2, buffer_size=100, random_seed=42)
    assert len(result) == 20


def test_all_results_from_input():
    spectra = make_spectra(50)
    ids = {s["id"] for s in spectra}
    with patch("casanovoutils.mzmlutils.pyteomics.mzml.MzML") as mock_cls:
        mock_cls.return_value = _mzml_cm(spectra)
        result = sample_mzml("dummy.mzML", k=0.4, buffer_size=50, random_seed=42)
    assert all(s["id"] in ids for s in result)


def test_large_proportion():
    # round(0.9 * 100) = 90
    spectra = make_spectra(100)
    with patch("casanovoutils.mzmlutils.pyteomics.mzml.MzML") as mock_cls:
        mock_cls.return_value = _mzml_cm(spectra)
        result = sample_mzml("dummy.mzML", k=0.9, buffer_size=100, random_seed=42)
    assert len(result) == 90


def test_small_proportion():
    # round(0.1 * 100) = 10
    spectra = make_spectra(100)
    with patch("casanovoutils.mzmlutils.pyteomics.mzml.MzML") as mock_cls:
        mock_cls.return_value = _mzml_cm(spectra)
        result = sample_mzml("dummy.mzML", k=0.1, buffer_size=100, random_seed=42)
    assert len(result) == 10


def test_multiple_buffers_sum_correctly():
    # 4 buffers of 25, k=0.5 → 4 * round(0.5 * 25) = 4 * 12 = 48
    # (round(12.5) = 12 in Python due to banker's rounding)
    spectra = make_spectra(100)
    with patch("casanovoutils.mzmlutils.pyteomics.mzml.MzML") as mock_cls:
        mock_cls.return_value = _mzml_cm(spectra)
        result = sample_mzml("dummy.mzML", k=0.5, buffer_size=25, random_seed=42)
    assert len(result) == 4 * round(0.5 * 25)


def test_empty_input_returns_empty():
    with patch("casanovoutils.mzmlutils.pyteomics.mzml.MzML") as mock_cls:
        mock_cls.return_value = _mzml_cm([])
        result = sample_mzml("dummy.mzML", k=0.5, random_seed=42)
    assert result == []


def test_reproducible_same_seed():
    spectra = make_spectra(50)
    with patch("casanovoutils.mzmlutils.pyteomics.mzml.MzML") as mock_cls:
        mock_cls.return_value = _mzml_cm(spectra)
        r1 = sample_mzml("dummy.mzML", k=0.5, buffer_size=50, random_seed=7)
    with patch("casanovoutils.mzmlutils.pyteomics.mzml.MzML") as mock_cls:
        mock_cls.return_value = _mzml_cm(spectra)
        r2 = sample_mzml("dummy.mzML", k=0.5, buffer_size=50, random_seed=7)
    assert [s["id"] for s in r1] == [s["id"] for s in r2]


def test_different_seeds_differ():
    spectra = make_spectra(100)
    with patch("casanovoutils.mzmlutils.pyteomics.mzml.MzML") as mock_cls:
        mock_cls.return_value = _mzml_cm(spectra)
        r1 = sample_mzml("dummy.mzML", k=0.5, buffer_size=100, random_seed=1)
    with patch("casanovoutils.mzmlutils.pyteomics.mzml.MzML") as mock_cls:
        mock_cls.return_value = _mzml_cm(spectra)
        r2 = sample_mzml("dummy.mzML", k=0.5, buffer_size=100, random_seed=2)
    assert [s["id"] for s in r1] != [s["id"] for s in r2]


# ---------------------------------------------------------------------------
# sample_mzml — validation
# ---------------------------------------------------------------------------


def test_k_int_raises():
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        sample_mzml("dummy.mzML", k=1)


def test_k_float_one_raises():
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        sample_mzml("dummy.mzML", k=1.0)


def test_k_float_greater_than_one_raises():
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        sample_mzml("dummy.mzML", k=1.5)


def test_k_float_zero_raises():
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        sample_mzml("dummy.mzML", k=0.0)


def test_k_float_negative_raises():
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        sample_mzml("dummy.mzML", k=-0.5)


def test_k_string_raises():
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        sample_mzml("dummy.mzML", k="half")


# ---------------------------------------------------------------------------
# sample_spectra — CLI dispatch
# ---------------------------------------------------------------------------


def test_sample_spectra_calls_mgf_write(tmp_path):
    spectra = make_spectra(10)
    outfile = tmp_path / "out.mgf"
    with (
        patch("casanovoutils.mzmlutils.pyteomics.mzml.MzML") as mock_mzml,
        patch("casanovoutils.mzmlutils.pyteomics.mgf.write") as mock_write,
    ):
        mock_mzml.return_value = _mzml_cm(spectra)
        sample_spectra("dummy.mzML", k=0.5, outfile=outfile, buffer_size=10, random_seed=42)
    mock_write.assert_called_once()


def test_sample_spectra_returns_none(tmp_path):
    spectra = make_spectra(10)
    outfile = tmp_path / "out.mgf"
    with (
        patch("casanovoutils.mzmlutils.pyteomics.mzml.MzML") as mock_mzml,
        patch("casanovoutils.mzmlutils.pyteomics.mgf.write"),
    ):
        mock_mzml.return_value = _mzml_cm(spectra)
        result = sample_spectra("dummy.mzML", k=0.5, outfile=outfile, buffer_size=10, random_seed=42)
    assert result is None


def test_sample_spectra_unsupported_extension_raises(tmp_path):
    spectra = make_spectra(10)
    outfile = tmp_path / "out.csv"
    with patch("casanovoutils.mzmlutils.pyteomics.mzml.MzML") as mock_mzml:
        mock_mzml.return_value = _mzml_cm(spectra)
        with pytest.raises(ValueError, match="Unsupported output extension"):
            sample_spectra("dummy.mzML", k=0.5, outfile=outfile, buffer_size=10, random_seed=42)


def test_sample_spectra_mzml_extension_raises(tmp_path):
    spectra = make_spectra(10)
    outfile = tmp_path / "out.mzml"
    with patch("casanovoutils.mzmlutils.pyteomics.mzml.MzML") as mock_mzml:
        mock_mzml.return_value = _mzml_cm(spectra)
        with pytest.raises(ValueError, match="Unsupported output extension"):
            sample_spectra("dummy.mzML", k=0.5, outfile=outfile, buffer_size=10, random_seed=42)


# ---------------------------------------------------------------------------
# _to_mgf_spectrum
# ---------------------------------------------------------------------------


def test_to_mgf_spectrum_has_params_key():
    s = make_spectrum(0)
    result = _to_mgf_spectrum(s)
    assert "params" in result


def test_to_mgf_spectrum_title_from_id():
    s = make_spectrum(0)
    result = _to_mgf_spectrum(s)
    assert result["params"]["TITLE"] == "scan=1"


def test_to_mgf_spectrum_preserves_arrays():
    s = make_spectrum(3)
    result = _to_mgf_spectrum(s)
    np.testing.assert_array_equal(result["m/z array"], s["m/z array"])
    np.testing.assert_array_equal(result["intensity array"], s["intensity array"])


def test_to_mgf_spectrum_no_precursor_no_pepmass():
    s = make_spectrum(0)
    result = _to_mgf_spectrum(s)
    assert "PEPMASS" not in result["params"]
    assert "CHARGE" not in result["params"]


def test_to_mgf_spectrum_with_precursor():
    s = make_spectrum(0)
    s["precursorList"] = {
        "count": 1,
        "precursor": [
            {
                "selectedIonList": {
                    "count": 1,
                    "selectedIon": [
                        {"selected ion m/z": 500.25, "peak intensity": 1e5, "charge state": 2}
                    ],
                }
            }
        ],
    }
    result = _to_mgf_spectrum(s)
    assert result["params"]["PEPMASS"] == 500.25
    assert result["params"]["CHARGE"] == "2+"


def test_to_mgf_spectrum_precursor_no_intensity():
    s = make_spectrum(0)
    s["precursorList"] = {
        "count": 1,
        "precursor": [
            {
                "selectedIonList": {
                    "count": 1,
                    "selectedIon": [{"selected ion m/z": 612.3, "charge state": 3}],
                }
            }
        ],
    }
    result = _to_mgf_spectrum(s)
    assert result["params"]["PEPMASS"] == 612.3
    assert result["params"]["CHARGE"] == "3+"


def test_to_mgf_spectrum_with_retention_time():
    s = make_spectrum(0)
    s["scanList"] = {
        "count": 1,
        "scan": [{"scan start time": 123.45}],
    }
    result = _to_mgf_spectrum(s)
    assert result["params"]["RTINSECONDS"] == 123.45


def test_to_mgf_spectrum_missing_id_uses_empty_title():
    s = make_spectrum(0)
    del s["id"]
    result = _to_mgf_spectrum(s)
    assert result["params"]["TITLE"] == ""


# ---------------------------------------------------------------------------
# _write_spectra — MGF round-trip
# ---------------------------------------------------------------------------


def test_write_spectra_mgf_dispatch(tmp_path):
    spectra = make_spectra(3)
    outfile = tmp_path / "out.mgf"
    with patch("casanovoutils.mzmlutils.pyteomics.mgf.write") as mock_write:
        _write_spectra(spectra, outfile)
    mock_write.assert_called_once()


def test_write_spectra_bad_extension_raises(tmp_path):
    with pytest.raises(ValueError, match="Unsupported output extension"):
        _write_spectra([], tmp_path / "out.txt")


def test_write_spectra_mzml_extension_raises(tmp_path):
    with pytest.raises(ValueError, match="Unsupported output extension"):
        _write_spectra([], tmp_path / "out.mzml")


def test_write_spectra_mgf_roundtrip(tmp_path):
    spectra = make_spectra(5)
    outfile = tmp_path / "out.mgf"
    _write_spectra(spectra, outfile)
    with pyteomics.mgf.read(str(outfile)) as reader:
        result = list(reader)
    assert len(result) == 5
    for orig, read in zip(spectra, result):
        np.testing.assert_allclose(orig["m/z array"], read["m/z array"], rtol=1e-5)
        np.testing.assert_allclose(
            orig["intensity array"], read["intensity array"], rtol=1e-5
        )


def test_write_spectra_mgf_roundtrip_titles(tmp_path):
    spectra = make_spectra(3)
    outfile = tmp_path / "out.mgf"
    _write_spectra(spectra, outfile)
    with pyteomics.mgf.read(str(outfile)) as reader:
        result = list(reader)
    titles = [r["params"]["title"] for r in result]
    assert titles == ["scan=1", "scan=2", "scan=3"]


def test_write_spectra_mgf_with_precursor_roundtrip(tmp_path):
    s = make_spectrum(0)
    s["precursorList"] = {
        "count": 1,
        "precursor": [
            {
                "selectedIonList": {
                    "count": 1,
                    "selectedIon": [{"selected ion m/z": 800.5, "charge state": 2}],
                }
            }
        ],
    }
    outfile = tmp_path / "out.mgf"
    _write_spectra([s], outfile)
    with pyteomics.mgf.read(str(outfile)) as reader:
        result = list(reader)
    assert len(result) == 1
    assert result[0]["params"]["charge"] == [2]
    assert abs(result[0]["params"]["pepmass"][0] - 800.5) < 1e-3


def test_write_spectra_mgf_empty(tmp_path):
    outfile = tmp_path / "out.mgf"
    _write_spectra([], outfile)
    with pyteomics.mgf.read(str(outfile)) as reader:
        result = list(reader)
    assert result == []
