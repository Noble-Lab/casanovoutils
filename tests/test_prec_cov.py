import polars as pl
import polars.testing
import pytest

from casanovoutils.prec_cov import (
    Constants,
    align_tokens_with_gaps,
    calc_precision_coverage,
    fill_null_columns,
    get_pred_sequence_column,
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
            "mztab_sequence": ["A", "B", "C", "D"],
            Constants.ground_truth_sequence_column: ["A", "X", "C", "Y"],
            Constants.pep_score_column: [0.9, 0.8, 0.7, 0.6],
            Constants.aa_scores_column: ["", "", "", ""],
        }
    )


# ── get_pred_sequence_column ──────────────────────────────────────────────────


def test_get_pred_sequence_column_prefers_proforma():
    df = pl.DataFrame(
        {
            "mztab_opt_ms_run[1]_proforma": ["PEPTIDE"],
            "mztab_sequence": ["PEPTIDE"],
        }
    )
    assert get_pred_sequence_column(df) == "mztab_opt_ms_run[1]_proforma"


def test_get_pred_sequence_column_falls_back_to_sequence():
    df = pl.DataFrame({"mztab_sequence": ["PEPTIDE"]})
    assert get_pred_sequence_column(df) == "mztab_sequence"


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


def test_align_gt_gap_score_is_min_score():
    pred, gt, sc = align_tokens_with_gaps(
        predicted=["A", "B"],
        ground_truth=["A"],
        scores=[1.0, 1.0],
    )
    gap_scores = [s for s, g in zip(sc, gt) if g == "-"]
    assert all(s != Constants.min_score for s in gap_scores)


def test_align_ignore_scores_same_alignment():
    tokens = ["A", "B", "C"]
    scores = [0.1, 0.1, 0.1]
    pred_normal, gt_normal, _ = align_tokens_with_gaps(tokens, tokens[:], scores)
    pred_ignored, gt_ignored, _ = align_tokens_with_gaps(
        tokens, tokens[:], scores, ignore_scores=True
    )
    assert pred_normal == pred_ignored
    assert gt_normal == gt_ignored


def test_align_ignore_scores_returns_original_scores():
    tokens = ["A", "B", "C"]
    scores = [0.1, 0.2, 0.3]
    _, _, sc = align_tokens_with_gaps(tokens, tokens[:], scores, ignore_scores=True)
    assert sc == scores


def test_align_empty_sequences():
    pred, gt, sc = align_tokens_with_gaps([], [], [])
    assert pred == []
    assert gt == []
    assert sc == []


def test_align_predicted_empty():
    pred, gt, sc = align_tokens_with_gaps([], ["A", "B"], [])
    assert all(p == "-" for p in pred)
    assert all(s == Constants.min_score for s in sc)


def test_align_ground_truth_empty():
    pred, gt, sc = align_tokens_with_gaps(["A", "B"], [], [1.0, 1.0])
    assert all(g == "-" for g in gt)


# ── mutate_row_as_dict ────────────────────────────────────────────────────────


@pytest.fixture
def sample_row():
    return {
        "mztab_sequence": ["A", "B", "C"],
        Constants.ground_truth_sequence_column: ["A", "X", "C"],
        Constants.aa_scores_column: [0.9, 0.8, 0.7],
        Constants.aa_idx_column: None,
    }


def test_mutate_row_as_dict_returns_dict(sample_row):
    result = mutate_row_as_dict("mztab_sequence", False, sample_row)
    assert isinstance(result, dict)


def test_mutate_row_as_dict_adds_aa_idx(sample_row):
    result = mutate_row_as_dict("mztab_sequence", False, sample_row)
    assert Constants.aa_idx_column in result
    assert result[Constants.aa_idx_column] == list(range(len(result["mztab_sequence"])))


def test_mutate_row_as_dict_aligned_lengths_equal(sample_row):
    result = mutate_row_as_dict("mztab_sequence", False, sample_row)
    n = len(result["mztab_sequence"])
    assert len(result[Constants.ground_truth_sequence_column]) == n
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
    # sorted descending by score: A(0.9)=correct, B(0.8)=wrong, C(0.7)=correct, D(0.6)=wrong
    assert result["pc_is_correct"].to_list() == [True, False, True, False]


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
            "mztab_sequence": ["A", "B", "C"],
            Constants.ground_truth_sequence_column: ["A", "B", "C"],
            Constants.pep_score_column: [0.9, 0.8, 0.7],
            Constants.aa_scores_column: ["", "", ""],
        }
    )
    result = calc_precision_coverage(df, Constants.pep_score_column)
    assert all(
        p == pytest.approx(1.0) for p in result[Constants.precision_column].to_list()
    )


def test_calc_precision_coverage_all_wrong():
    df = pl.DataFrame(
        {
            "mztab_sequence": ["A", "B", "C"],
            Constants.ground_truth_sequence_column: ["X", "Y", "Z"],
            Constants.pep_score_column: [0.9, 0.8, 0.7],
            Constants.aa_scores_column: ["", "", ""],
        }
    )
    result = calc_precision_coverage(df, Constants.pep_score_column)
    assert all(
        p == pytest.approx(0.0) for p in result[Constants.precision_column].to_list()
    )
