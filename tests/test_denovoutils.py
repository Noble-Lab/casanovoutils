import polars as pl
import polars.testing
import pytest

from casanovoutils.constants import Constants
from casanovoutils.denovoutils import (
    get_ground_truth_df,
    get_mgf_psms_df,
    get_mztab_df,
    process_spectrum,
    read_dataframe,
    write_dataframe,
)


def test_process_spectrum_adds_n_peaks():
    spectrum = {"params": {"title": "spec1"}, "m/z array": [1.0, 2.0, 3.0]}
    result = process_spectrum(spectrum)
    assert result["n_peaks"] == 3


def test_process_spectrum_preserves_existing_params():
    spectrum = {"params": {"title": "spec1", "charge": 2}, "m/z array": [1.0]}
    result = process_spectrum(spectrum)
    assert result["title"] == "spec1"
    assert result["charge"] == 2


def test_process_spectrum_empty_mz_array():
    spectrum = {"params": {}, "m/z array": []}
    result = process_spectrum(spectrum)
    assert result["n_peaks"] == 0


@pytest.fixture
def simple_df():
    return pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def test_write_dataframe_unsupported_extension(tmp_path, simple_df):
    with pytest.raises(ValueError, match="Unsupported file type"):
        write_dataframe(simple_df, tmp_path / "test.xlsx")


def test_write_dataframe_creates_file(tmp_path, simple_df):
    path = tmp_path / "test.parquet"
    write_dataframe(simple_df, path)
    assert path.exists()


def test_read_dataframe_passthrough(simple_df):
    assert read_dataframe(simple_df) is simple_df


def test_read_dataframe_unsupported_extension(tmp_path):
    with pytest.raises(ValueError, match="Unsupported file type"):
        read_dataframe(tmp_path / "test.xlsx")


@pytest.mark.parametrize("suffix", [".parquet", ".pq", ".csv", ".tsv"])
def test_roundtrip(tmp_path, simple_df, suffix):
    path = tmp_path / f"test{suffix}"
    write_dataframe(simple_df, path)
    result = read_dataframe(path)
    polars.testing.assert_frame_equal(result, simple_df)


def test_roundtrip_preserves_row_count(tmp_path, simple_df):
    path = tmp_path / "test.parquet"
    write_dataframe(simple_df, path)
    assert len(read_dataframe(path)) == len(simple_df)


def test_roundtrip_preserves_columns(tmp_path, simple_df):
    path = tmp_path / "test.parquet"
    write_dataframe(simple_df, path)
    assert read_dataframe(path).columns == simple_df.columns


def test_get_mgf_psms_df_passthrough(simple_df):
    assert get_mgf_psms_df(simple_df) is simple_df


def test_get_mgf_psms_df_writes_output(tmp_path, simple_df):
    out_path = tmp_path / "out.parquet"
    get_mgf_psms_df(simple_df, out_path=out_path)
    assert out_path.exists()


def test_get_mztab_df_passthrough(simple_df):
    assert get_mztab_df(simple_df) is simple_df


def test_get_mztab_df_writes_output(tmp_path, simple_df):
    out_path = tmp_path / "out.parquet"
    get_mztab_df(simple_df, out_path=out_path)
    assert out_path.exists()


@pytest.fixture
def mgf_df():
    return pl.DataFrame(
        {
            "mgf_title": ["spec1", "spec2", "spec3"],
            "mgf_n_peaks": [10, 20, 30],
        }
    )


@pytest.fixture
def mgf_df1():
    return pl.DataFrame(
        {
            "mgf_title": ["spec1", "spec2"],
            "mgf_n_peaks": [10, 20],
        }
    )


@pytest.fixture
def mgf_df2():
    return pl.DataFrame(
        {
            "mgf_title": ["spec3"],
            "mgf_n_peaks": [30],
        }
    )


@pytest.fixture
def mztab_df():
    return pl.DataFrame(
        {
            "mztab_spectra_ref": [
                "ms_run[1]:index=0",
                "ms_run[1]:index=1",
                "ms_run[1]:index=2",
            ],
            "mztab_sequence": ["PEPTIDE", "ANOTHER", "THIRD"],
        }
    )


def test_get_ground_truth_df_joins_correctly(mgf_df, mztab_df):
    result = get_ground_truth_df(mgf_df, mztab_df)
    assert "mgf_title" in result.columns
    assert "mztab_sequence" in result.columns


def test_get_ground_truth_df_excludes_tmp_columns(mgf_df, mztab_df):
    result = get_ground_truth_df(mgf_df, mztab_df)
    assert not any(c.startswith("tmp_") for c in result.columns)


def test_get_ground_truth_df_correct_row_count(mgf_df, mztab_df):
    result = get_ground_truth_df(mgf_df, mztab_df)
    assert len(result) == 3


def test_get_ground_truth_df_writes_output(tmp_path, mgf_df, mztab_df):
    out_path = tmp_path / "out.parquet"
    get_ground_truth_df(mgf_df, mztab_df, out_path=out_path)
    assert out_path.exists()


def test_process_spectrum_meta_data_only_excludes_arrays():
    spectrum = {
        "params": {"title": "spec1"},
        "m/z array": [1.0, 2.0],
        "intensity array": [100.0, 200.0],
    }
    result = process_spectrum(spectrum, meta_data_only=True)
    assert "m_z_array" not in result
    assert "intensity_array" not in result


def test_process_spectrum_not_meta_data_only_includes_arrays():
    spectrum = {
        "params": {"title": "spec1"},
        "m/z array": [1.0, 2.0],
        "intensity array": [100.0, 200.0],
    }
    result = process_spectrum(spectrum, meta_data_only=False)
    assert "m_z_array" in result
    assert "intensity_array" in result


def test_process_spectrum_not_meta_data_only_array_values():
    mz = [1.0, 2.0, 3.0]
    intensities = [100.0, 200.0, 300.0]
    spectrum = {
        "params": {},
        "m/z array": mz,
        "intensity array": intensities,
    }
    result = process_spectrum(spectrum, meta_data_only=False)
    assert result["m_z_array"] == mz
    assert result["intensity_array"] == intensities


def test_process_spectrum_not_meta_data_only_still_adds_n_peaks():
    spectrum = {
        "params": {},
        "m/z array": [1.0, 2.0],
        "intensity array": [100.0, 200.0],
    }
    result = process_spectrum(spectrum, meta_data_only=False)
    assert result["n_peaks"] == 2


def test_process_spectrum_meta_data_only_default_is_true():
    """meta_data_only should default to True — arrays must be absent when omitted."""
    spectrum = {
        "params": {},
        "m/z array": [1.0],
        "intensity array": [100.0],
    }
    # existing tests call process_spectrum without meta_data_only;
    # confirm the default behaviour matches meta_data_only=True
    result = process_spectrum(spectrum)
    assert "m_z_array" not in result
    assert "intensity_array" not in result


def test_get_mgf_psms_df_passthrough_ignores_meta_data_only(simple_df):
    """When a DataFrame is passed directly, meta_data_only has no effect."""
    result = get_mgf_psms_df(simple_df, meta_data_only=False)
    assert result is simple_df


def test_get_mgf_psms_df_meta_data_only_excludes_array_columns(tmp_path):
    mgf_content = (
        "BEGIN IONS\n" "TITLE=spec1\n" "100.0 500.0\n" "200.0 300.0\n" "END IONS\n"
    )
    mgf_path = tmp_path / "test.mgf"
    mgf_path.write_text(mgf_content)

    result = get_mgf_psms_df(mgf_path, meta_data_only=True)
    assert "mgf_m_z_array" not in result.columns
    assert "mgf_intensity_array" not in result.columns


def test_get_mgf_psms_df_not_meta_data_only_includes_array_columns(tmp_path):
    mgf_content = (
        "BEGIN IONS\n" "TITLE=spec1\n" "100.0 500.0\n" "200.0 300.0\n" "END IONS\n"
    )
    mgf_path = tmp_path / "test.mgf"
    mgf_path.write_text(mgf_content)

    result = get_mgf_psms_df(mgf_path, meta_data_only=False)
    assert "mgf_m_z_array" in result.columns
    assert "mgf_intensity_array" in result.columns


def test_get_ground_truth_df_list_run2_joins_second_file(mgf_df1, mgf_df2):
    """ms_run[2]:index=N uses the global concat index, so N=2 maps to mgf_df2's first row."""
    mztab = pl.DataFrame(
        {
            "mztab_spectra_ref": [
                "ms_run[1]:index=0",
                "ms_run[1]:index=1",
                "ms_run[2]:index=0",
            ],
            "mztab_sequence": ["PEPTIDE", "ANOTHER", "THIRD"],
        }
    )
    result = get_ground_truth_df([mgf_df1, mgf_df2], mztab)
    # The row whose title is "spec3" (from mgf_df2) should be joined with "THIRD"
    row = result.filter(pl.col("mgf_title") == "spec3")
    assert row["mztab_sequence"][0] == "THIRD"


def test_get_mgf_psms_df_n_peaks_correct_from_file(tmp_path):
    mgf_content = (
        "BEGIN IONS\n"
        "TITLE=spec1\n"
        "100.0 500.0\n"
        "200.0 300.0\n"
        "300.0 100.0\n"
        "END IONS\n"
    )
    mgf_path = tmp_path / "test.mgf"
    mgf_path.write_text(mgf_content)

    result = get_mgf_psms_df(mgf_path, meta_data_only=True)
    assert result["mgf_n_peaks"][0] == 3


def test_get_mgf_psms_df_columns_have_mgf_prefix(tmp_path):
    mgf_content = "BEGIN IONS\n" "TITLE=spec1\n" "100.0 500.0\n" "END IONS\n"
    mgf_path = tmp_path / "test.mgf"
    mgf_path.write_text(mgf_content)

    result = get_mgf_psms_df(mgf_path)
    assert all(c.startswith("mgf_") for c in result.columns)
