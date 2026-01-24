import numpy as np
import pandas as pd
import pytest

# CHANGE THIS import to match your module path/name.
# e.g. from casanovoutils.evaluate import get_ground_truth, prec_cov
from casanovoutils.evaluate import get_ground_truth, prec_cov


def _write_mgf(tmp_path, seqs: list[str]) -> str:
    """Write a minimal MGF-like file containing SEQ= lines."""
    p = tmp_path / "gt.mgf"
    # Interleave junk lines to ensure we only parse SEQ=
    lines = []
    for s in seqs:
        lines.append("BEGIN IONS\n")
        lines.append(f"SEQ={s}\n")
        lines.append("END IONS\n")
    p.write_text("".join(lines))
    return str(p)


def _psm_df(
    indices: list[int], sequences: list[str], scores: list[float]
) -> pd.DataFrame:
    """Create a minimal PSM df matching the expected columns."""
    assert len(indices) == len(sequences) == len(scores)
    return pd.DataFrame(
        {
            "spectra_ref": [f"ms_run[1]:index={i}" for i in indices],
            "sequence": sequences,
            "search_engine_score[1]": scores,
        }
    )


# -------------------------
# get_ground_truth() tests
# -------------------------


def test_get_ground_truth_dataframe_aligns_by_spectra_index(tmp_path):
    """
    Predictions should be inserted into the output rows at positions derived
    from spectra_ref indices; missing spectra keep default fill values.
    """
    mgf_seqs = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    mgf_path = _write_mgf(tmp_path, mgf_seqs)

    # Only predict spectra 1 and 3
    psm = _psm_df(indices=[1, 3], sequences=["BBB", "XXX"], scores=[10.0, 2.5])

    out = get_ground_truth(psm, mgf_path, replace_i_l=False)

    # shape/columns
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == [
        "ground_truth",
        "predicted",
        "pep_score",
        "pep_correct",
    ]
    assert len(out) == len(mgf_seqs)

    # Ground truth copied
    assert out["ground_truth"].tolist() == mgf_seqs

    # Defaults for missing predictions
    assert out.loc[0, "predicted"] == ""
    assert out.loc[0, "pep_score"] == -1.0
    assert out.loc[2, "predicted"] == ""
    assert out.loc[2, "pep_score"] == -1.0
    assert out.loc[4, "predicted"] == ""
    assert out.loc[4, "pep_score"] == -1.0

    # Inserted predictions at proper indices
    assert out.loc[1, "predicted"] == "BBB"
    assert float(out.loc[1, "pep_score"]) == 10.0
    assert out.loc[3, "predicted"] == "XXX"
    assert float(out.loc[3, "pep_score"]) == 2.5

    # Correctness computed as exact string equality
    # index 1 matches, index 3 doesn't, missing predictions are false
    assert out["pep_correct"].tolist() == [False, True, False, False, False]


def test_get_ground_truth_replace_i_l_bug_is_exposed(tmp_path):
    """
    The current implementation calls .str.replace(...) without assignment,
    so replace_i_l=True should NOT change behavior (this test locks in the bug).

    If you later fix the bug, update/flip this test accordingly.
    """
    mgf_seqs = ["ILIL", "LLLL"]
    mgf_path = _write_mgf(tmp_path, mgf_seqs)

    # Predict "LLLL" for spectrum 0; would become correct if I->L replacement worked.
    psm = _psm_df(indices=[0], sequences=["LLLL"], scores=[1.0])

    out_no_replace = get_ground_truth(psm, mgf_path, replace_i_l=False)
    out_replace = get_ground_truth(psm, mgf_path, replace_i_l=True)

    # Both should be incorrect for row 0 given the current bug
    assert out_no_replace.loc[0, "pep_correct"] is False
    assert out_replace.loc[0, "pep_correct"] is False


def test_get_ground_truth_raises_on_bad_spectra_ref_format(tmp_path):
    """
    spectra_ref must begin with 'ms_run[1]:index=' and the remainder must be int.
    """
    mgf_path = _write_mgf(tmp_path, ["AAA", "BBB"])

    psm = pd.DataFrame(
        {
            "spectra_ref": ["totally-wrong-format"],
            "sequence": ["AAA"],
            "search_engine_score[1]": [1.0],
        }
    )

    # Could raise ValueError from int() conversion
    with pytest.raises(Exception):
        get_ground_truth(psm, mgf_path)


def test_get_ground_truth_mztab_path_uses_pyteomics_reader(monkeypatch, tmp_path):
    """
    If mztab_path is not a DataFrame, the function should call
    pyteomics.mztab.MzTab(...).spectrum_match_table.

    We monkeypatch MzTab to avoid real I/O.
    """
    mgf_path = _write_mgf(tmp_path, ["AAA", "BBB", "CCC"])

    dummy_psm = _psm_df(indices=[2], sequences=["CCC"], scores=[9.0])

    class DummyMzTab:
        def __init__(self, path):
            self.path = path
            self.spectrum_match_table = dummy_psm

    import pyteomics.mztab as mztab_mod

    monkeypatch.setattr(mztab_mod, "MzTab", DummyMzTab)

    out = get_ground_truth("fake.mztab", mgf_path)

    assert out.loc[2, "predicted"] == "CCC"
    assert float(out.loc[2, "pep_score"]) == 9.0
    assert out.loc[2, "pep_correct"] is True


# -------------------------
# prec_cov() tests
# -------------------------


def test_prec_cov_sorts_by_score_descending_and_computes_curve():
    """
    Verify sorting, precision/coverage lengths, endpoints, and expected values
    on a small hand-checkable example.
    """
    scores = np.array([0.2, 0.9, 0.5, 0.1])
    is_correct = np.array([0, 1, 1, 0], dtype=bool)

    precision, coverage, aupc = prec_cov(scores, is_correct)

    # Lengths match
    assert len(precision) == len(scores)
    assert len(coverage) == len(scores)

    # Coverage should end at 1.0
    assert np.isclose(coverage[-1], 1.0)

    # After sorting by score desc: indices [1,2,0,3] -> is_correct [1,1,0,0]
    # precision: [1/1, 2/2, 2/3, 2/4] = [1,1,0.666...,0.5]
    expected_precision = np.array([1.0, 1.0, 2 / 3, 0.5])
    expected_coverage = np.array([1 / 4, 2 / 4, 3 / 4, 1.0])

    assert np.allclose(precision, expected_precision)
    assert np.allclose(coverage, expected_coverage)

    # AUPC computed via trapezoid rule should be within [0,1]
    assert 0.0 <= aupc <= 1.0


@pytest.mark.parametrize(
    "scores,is_correct",
    [
        (np.array([1.0, 2.0, 3.0]), np.array([True, True, True])),
        (np.array([1.0, 2.0, 3.0]), np.array([False, False, False])),
    ],
)
def test_prec_cov_extremes(scores, is_correct):
    """
    All-correct => precision always 1, AUPC should be 1.
    All-wrong => precision always 0, AUPC should be 0.
    """
    precision, coverage, aupc = prec_cov(scores, is_correct)

    if is_correct.all():
        assert np.allclose(precision, 1.0)
        assert np.isclose(aupc, 1.0)
    else:
        assert np.allclose(precision, 0.0)
        assert np.isclose(aupc, 0.0)


def test_prec_cov_handles_ties_deterministically_enough():
    """
    With ties, argsort ordering can depend on implementation details, so we
    assert only invariants: coverage monotonic, precision within [0,1], etc.
    """
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    is_correct = np.array([True, False, True, False])

    precision, coverage, aupc = prec_cov(scores, is_correct)

    assert np.all(np.diff(coverage) > 0)
    assert np.all((precision >= 0.0) & (precision <= 1.0))
    assert 0.0 <= aupc <= 1.0
