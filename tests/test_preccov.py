import polars as pl
import polars.testing
import pytest

from casanovoutils.constants import Constants
from casanovoutils.preccov import (
    _aa_match_batch,
    _aa_match_prefix,
    align_tokens_with_gaps,
    calc_precision_coverage,
    fill_null_columns,
    load_ground_truth_df,
    mutate_row_as_dict,
)

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def pred_col():
    return "mztab_sequence"


@pytest.fixture
def pc_input_df():
    return pl.DataFrame(
        {
            Constants.predicted_tokens: ["A", "B", "C", "D"],
            Constants.ground_truth_tokens: ["A", "X", "C", "Y"],
            Constants.pep_score_column: [0.9, 0.8, 0.7, 0.6],
            Constants.aa_scores_column: ["", "", "", ""],
        }
    )


# ── fill_null_columns ─────────────────────────────────────────────────────────


@pytest.fixture
def nullable_df():
    return pl.DataFrame(
        {
            "mztab_sequence": [None, "PEPTIDE"],
            Constants.aa_scores_column: [None, "0.9,0.8"],
            Constants.pep_score_column: [None, 0.95],
        }
    )


def test_fill_null_columns_fills_predicted(nullable_df):
    result = fill_null_columns(nullable_df, "mztab_sequence")
    assert result["mztab_sequence"][0] == ""


def test_fill_null_columns_fills_aa_scores(nullable_df):
    result = fill_null_columns(nullable_df, "mztab_sequence")
    assert result[Constants.aa_scores_column][0] == ""


def test_fill_null_columns_fills_pep_score(nullable_df):
    result = fill_null_columns(nullable_df, "mztab_sequence")
    assert result[Constants.pep_score_column][0] == -1.0


def test_fill_null_columns_preserves_non_null(nullable_df):
    result = fill_null_columns(nullable_df, "mztab_sequence")
    assert result["mztab_sequence"][1] == "PEPTIDE"
    assert result[Constants.aa_scores_column][1] == "0.9,0.8"
    assert result[Constants.pep_score_column][1] == pytest.approx(0.95)


# ── load_ground_truth_df ──────────────────────────────────────────────────────


def test_load_ground_truth_df_raises_without_inputs():
    with pytest.raises(ValueError, match="--mgf_df and --mztab_df must be provided"):
        load_ground_truth_df(None, None, None)


@pytest.fixture
def simple_df():
    return pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def test_load_ground_truth_df_raises_with_only_mgf(simple_df):
    with pytest.raises(ValueError, match="--mgf_df and --mztab_df must be provided"):
        load_ground_truth_df(None, simple_df, None)


def test_load_ground_truth_df_raises_with_only_mztab(simple_df):
    with pytest.raises(ValueError, match="--mgf_df and --mztab_df must be provided"):
        load_ground_truth_df(None, None, simple_df)


def test_load_ground_truth_df_passthrough(simple_df):
    result = load_ground_truth_df(simple_df, None, None)
    polars.testing.assert_frame_equal(result, simple_df)


# ── align_tokens_with_gaps ────────────────────────────────────────────────────


def test_align_identical_sequences():
    tokens = ["A", "B", "C"]
    scores = [0.9, 0.8, 0.7]
    pred, gt, sc = align_tokens_with_gaps(tokens, tokens[:], scores)
    assert pred == tokens
    assert gt == tokens
    assert sc == scores


def test_align_output_lengths_equal():
    pred, gt, sc = align_tokens_with_gaps(
        predicted=["A", "C"],
        ground_truth=["A", "B", "C"],
        scores=[1.0, 1.0],
    )
    assert len(pred) == len(gt) == len(sc)


def test_align_inserts_gaps_in_predicted():
    pred, gt, sc = align_tokens_with_gaps(
        predicted=["A", "C"],
        ground_truth=["A", "B", "C"],
        scores=[1.0, 1.0],
    )
    assert "-" in pred


def test_align_gap_score_is_min_score():
    pred, gt, sc = align_tokens_with_gaps(
        predicted=["A"],
        ground_truth=["A", "B"],
        scores=[1.0],
    )
    gap_scores = [s for s, p in zip(sc, pred) if p == "-"]
    assert all(s == Constants.min_score for s in gap_scores)


def test_align_empty_predicted():
    pred, gt, sc = align_tokens_with_gaps([], ["A", "B"], [])
    assert all(p == "-" for p in pred)
    assert all(s == Constants.min_score for s in sc)


def test_align_empty_sequences():
    pred, gt, sc = align_tokens_with_gaps([], [], [])
    assert pred == []
    assert gt == []
    assert sc == []


# ── mutate_row_as_dict ────────────────────────────────────────────────────────


@pytest.fixture
def sample_row():
    return {
        Constants.predicted_tokens: ["A", "B", "C"],
        Constants.ground_truth_tokens: ["A", "X", "C"],
        Constants.aa_scores_column: [0.9, 0.8, 0.7],
        Constants.aa_idx_column: None,
    }


def test_mutate_row_as_dict_returns_dict(sample_row):
    result = mutate_row_as_dict(False, sample_row)
    assert isinstance(result, dict)


def test_mutate_row_as_dict_adds_aa_idx(sample_row):
    result = mutate_row_as_dict(False, sample_row)
    assert Constants.aa_idx_column in result
    assert result[Constants.aa_idx_column] == list(
        range(len(result[Constants.predicted_tokens]))
    )


def test_mutate_row_as_dict_aligned_lengths_equal(sample_row):
    result = mutate_row_as_dict(False, sample_row)
    n = len(result[Constants.predicted_tokens])
    assert len(result[Constants.ground_truth_tokens]) == n
    assert len(result[Constants.aa_scores_column]) == n
    assert len(result[Constants.aa_idx_column]) == n


# ── calc_precision_coverage ───────────────────────────────────────────────────


def test_calc_precision_coverage_output_columns(pc_input_df):
    result = calc_precision_coverage(pc_input_df, Constants.pep_score_column)
    assert Constants.precision_column in result.columns
    assert Constants.coverage_column in result.columns
    assert "pc_is_correct" in result.columns


def test_calc_precision_coverage_correctness_flag(pc_input_df):
    result = calc_precision_coverage(pc_input_df, Constants.pep_score_column)
    # sorted descending by score: A(0.9), B(0.8), C(0.7), D(0.6)
    # A vs A: same token → match.
    # B vs X: both unknown (mass 0) → cumulative delta 0 < 0.5 and ind delta 0 < 0.1 → match.
    # C vs C: same token → match.
    # D (115.03 Da) vs Y (163.06 Da): delta ~48 Da > 0.5 → no match.
    assert result["pc_is_correct"].to_list() == [True, True, True, False]


def test_calc_precision_coverage_precision_range(pc_input_df):
    result = calc_precision_coverage(pc_input_df, Constants.pep_score_column)
    assert all(0.0 <= p <= 1.0 for p in result[Constants.precision_column].to_list())


def test_calc_precision_coverage_coverage_range(pc_input_df):
    result = calc_precision_coverage(pc_input_df, Constants.pep_score_column)
    assert all(0.0 <= c <= 1.0 for c in result[Constants.coverage_column].to_list())


def test_calc_precision_coverage_ends_at_full_coverage(pc_input_df):
    result = calc_precision_coverage(pc_input_df, Constants.pep_score_column)
    assert result[Constants.coverage_column][-1] == pytest.approx(1.0)


def test_calc_precision_coverage_sorted_descending(pc_input_df):
    result = calc_precision_coverage(pc_input_df, Constants.pep_score_column)
    scores = result[Constants.pep_score_column].to_list()
    assert scores == sorted(scores, reverse=True)


def test_calc_precision_coverage_all_correct():
    df = pl.DataFrame(
        {
            Constants.predicted_tokens: ["A", "B", "C"],
            Constants.ground_truth_tokens: ["A", "B", "C"],
            Constants.pep_score_column: [0.9, 0.8, 0.7],
            Constants.aa_scores_column: ["", "", ""],
        }
    )
    result = calc_precision_coverage(df, Constants.pep_score_column)
    assert all(
        p == pytest.approx(1.0) for p in result[Constants.precision_column].to_list()
    )


def test_calc_precision_coverage_all_wrong():
    # Use known amino acids whose masses are clearly distinct so mass-based
    # matching correctly identifies them as non-matching.
    # A=71.04, Q=128.06, W=186.08 vs X(unk,0), Y=163.06, V=99.07 — all deltas > 0.5 Da.
    df = pl.DataFrame(
        {
            Constants.predicted_tokens: ["A", "Q", "W"],
            Constants.ground_truth_tokens: ["X", "Y", "V"],
            Constants.pep_score_column: [0.9, 0.8, 0.7],
            Constants.aa_scores_column: ["", "", ""],
        }
    )
    result = calc_precision_coverage(df, Constants.pep_score_column)
    assert all(
        p == pytest.approx(0.0) for p in result[Constants.precision_column].to_list()
    )


# ── mass-based matching helpers ───────────────────────────────────────────────

# Monoisotopic masses used in residues.yaml (a subset sufficient for these tests)
_RESIDUES = {
    "G": 57.021464,
    "A": 71.037114,
    "L": 113.084064,
    "I": 113.084064,  # same mass as L
    "D": 115.026943,
    "E": 129.042593,
}


def test_aa_match_prefix_identical():
    """Identical sequences fully match."""
    aa_matches, pep_match = _aa_match_prefix(
        ["A", "G", "L"], ["A", "G", "L"], _RESIDUES, 0.5, 0.1
    )
    assert pep_match is True
    assert aa_matches.all()


def test_aa_match_prefix_il_equivalent():
    """I and L have the same mass and are treated as a match."""
    aa_matches, pep_match = _aa_match_prefix(
        ["A", "I", "G"], ["A", "L", "G"], _RESIDUES, 0.5, 0.1
    )
    assert pep_match is True
    assert aa_matches.all()


def test_aa_match_prefix_mass_mismatch():
    """Residues with clearly different masses do not match."""
    aa_matches, pep_match = _aa_match_prefix(["A"], ["E"], _RESIDUES, 0.5, 0.1)
    assert pep_match is False
    assert not aa_matches.any()


def test_aa_match_prefix_length_mismatch():
    """Sequences of different lengths never fully match."""
    aa_matches, pep_match = _aa_match_prefix(
        ["A", "G"], ["A", "G", "L"], _RESIDUES, 0.5, 0.1
    )
    assert pep_match is False


def test_aa_match_batch_il_equivalence():
    """_aa_match_batch marks I/L substitutions as correct at peptide level."""
    batch, _, _ = _aa_match_batch(
        [["A", "I", "D"], ["G", "L"]],
        [["A", "L", "D"], ["G", "I"]],
        _RESIDUES,
    )
    assert batch[0][1] is True  # LAID ≈ WALD (I/L swap)
    assert batch[1][1] is True  # GL ≈ GI (I/L swap)


def test_calc_precision_coverage_il_match():
    """I/L-swapped predictions are counted as correct by default."""
    df = pl.DataFrame(
        {
            Constants.predicted_tokens: [["A", "I", "G"], ["A", "L", "G"]],
            Constants.ground_truth_tokens: [["A", "L", "G"], ["A", "I", "G"]],
            Constants.pep_score_column: [0.9, 0.8],
            Constants.aa_scores_column: ["", ""],
        }
    )
    result = calc_precision_coverage(df, Constants.pep_score_column)
    assert result["pc_is_correct"].to_list() == [True, True]


# ── tests for Constants.get_aa_scores_column ─────────────────────────────────


def test_get_aa_scores_column_opt_global():
    """opt_global_aa_scores (current pyteomics) is detected correctly."""
    df = pl.DataFrame({"mztab_opt_global_aa_scores": ["0.9,0.8"]})
    assert Constants.get_aa_scores_column(df) == "mztab_opt_global_aa_scores"


def test_get_aa_scores_column_opt_ms_run():
    """opt_ms_run[1]_aa_scores (older pyteomics) is detected as fallback."""
    df = pl.DataFrame({"mztab_opt_ms_run[1]_aa_scores": ["0.9,0.8"]})
    assert Constants.get_aa_scores_column(df) == "mztab_opt_ms_run[1]_aa_scores"


def test_get_aa_scores_column_prefers_opt_global():
    """opt_global_* is preferred when both columns are present."""
    df = pl.DataFrame(
        {
            "mztab_opt_global_aa_scores": ["0.9,0.8"],
            "mztab_opt_ms_run[1]_aa_scores": ["0.9,0.8"],
        }
    )
    assert Constants.get_aa_scores_column(df) == "mztab_opt_global_aa_scores"


def test_get_aa_scores_column_raises_when_missing():
    """ValueError is raised when neither column variant is present."""
    df = pl.DataFrame({"mztab_sequence": ["PEPTIDE"]})
    with pytest.raises(ValueError, match="mztab_opt_global_aa_scores"):
        Constants.get_aa_scores_column(df)


# ── tests for Constants.get_pred_sequence_column ─────────────────────────────


def test_get_pred_sequence_column_cv_proforma():
    """CV-term ProForma column (current Casanovo) is preferred."""
    col = "mztab_opt_global_cv_MS:1003169_proforma_peptidoform_sequence"
    df = pl.DataFrame({col: ["PEPTIDE"], "mztab_sequence": ["PEPTIDE"]})
    assert Constants.get_pred_sequence_column(df) == col


def test_get_pred_sequence_column_old_proforma():
    """Older opt_ms_run[1]_proforma column is detected as fallback."""
    df = pl.DataFrame(
        {"mztab_opt_ms_run[1]_proforma": ["PEPTIDE"], "mztab_sequence": ["PEPTIDE"]}
    )
    assert Constants.get_pred_sequence_column(df) == "mztab_opt_ms_run[1]_proforma"


def test_get_pred_sequence_column_cv_preferred_over_old():
    """CV-term ProForma column is preferred when both ProForma variants present."""
    cv_col = "mztab_opt_global_cv_MS:1003169_proforma_peptidoform_sequence"
    df = pl.DataFrame(
        {cv_col: ["PEPTIDE"], "mztab_opt_ms_run[1]_proforma": ["PEPTIDE"]}
    )
    assert Constants.get_pred_sequence_column(df) == cv_col


def test_get_pred_sequence_column_fallback_to_sequence():
    """Falls back to mztab_sequence when no ProForma column is present."""
    df = pl.DataFrame({"mztab_sequence": ["PEPTIDE"]})
    assert Constants.get_pred_sequence_column(df) == "mztab_sequence"
