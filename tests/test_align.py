"""
Unit tests for casanovoutils.align.
"""

import numpy as np
import pytest

from casanovoutils.align import (
    align_scores,
    align_tokens_with_gaps,
    get_aligned_dp_array,
    recover_solution,
)
from casanovoutils.constants import Constants

# ---------------------------------------------------------------------------
# get_aligned_dp_array
# ---------------------------------------------------------------------------


class TestGetAlignedDpArray:

    def test_returns_correct_shape(self):
        short = ["A", "B"]
        long = ["A", "X", "B"]
        dp = get_aligned_dp_array(short, long)
        assert len(dp) == len(short) + 1
        assert all(len(row) == len(long) + 1 for row in dp)

    def test_rows_are_independent(self):
        # Regression test for the [[0]*l]*s shared-reference bug
        short = ["A", "B"]
        long = ["A", "X", "B"]
        dp = get_aligned_dp_array(short, long)
        dp[0][0] = 99
        assert dp[1][0] != 99

    def test_perfect_match(self):
        # short == long prefix: every token matches
        short = ["A", "B"]
        long = ["A", "B"]
        dp = get_aligned_dp_array(short, long)
        # Best score starting at (0, 0) should be 2
        assert dp[0][0] == 2

    def test_no_match(self):
        short = ["X"]
        long = ["A", "B", "C"]
        dp = get_aligned_dp_array(short, long)
        assert dp[0][0] == 0

    def test_single_token_match(self):
        dp = get_aligned_dp_array(["A"], ["A"])
        assert dp[0][0] == 1

    def test_single_token_no_match(self):
        dp = get_aligned_dp_array(["A"], ["B"])
        assert dp[0][0] == 0


# ---------------------------------------------------------------------------
# recover_solution
# ---------------------------------------------------------------------------


class TestRecoverSolution:

    def _align(self, short, long, tie_break_suffix=True):
        dp = get_aligned_dp_array(short, long)
        return recover_solution(dp, short, long, "-", tie_break_suffix)

    def test_output_length_equals_long(self):
        short = ["A", "B"]
        long = ["A", "X", "B"]
        result = self._align(short, long)
        assert len(result) == len(long)

    def test_gap_inserted_in_correct_position(self):
        short = ["A", "B"]
        long = ["A", "X", "B"]
        result = self._align(short, long)
        assert result == ["A", "-", "B"]

    def test_no_gaps_needed(self):
        short = ["A", "B"]
        long = ["A", "B"]
        result = self._align(short, long)
        assert result == ["A", "B"]

    def test_all_gaps(self):
        # short has one token, long has three non-matching tokens
        short = ["Z"]
        long = ["A", "B", "C"]
        result = self._align(short, long)
        assert len(result) == 3
        assert result.count("-") == 2

    def test_tie_break_suffix_true(self):
        # On a tie, suffix=True means prefer consuming a real token (gap comes later)
        short = ["A"]
        long = ["A", "A"]
        result = self._align(short, long, tie_break_suffix=True)
        assert result == ["-", "A"]

    def test_tie_break_suffix_false(self):
        # On a tie, suffix=False means prefer inserting a gap (real token comes first)
        short = ["A"]
        long = ["A", "A"]
        result = self._align(short, long, tie_break_suffix=False)
        assert result == ["A", "-"]

    def test_returns_list(self):
        result = self._align(["A"], ["A", "B"])
        assert isinstance(result, list)

    def test_regression_2026_03_24(self):
        short = [
            "N",
            "T",
            "G",
            "S",
            "Q",
            "F",
            "V",
            "M",
            "E",
            "G",
            "V",
            "K",
            "N",
            "L",
            "V",
            "L",
            "K",
            "Q",
            "Q",
            "N",
            "L",
            "P",
            "V",
            "T",
            "R",
        ]
        long = [
            "N",
            "T",
            "S",
            "G",
            "E",
            "F",
            "V",
            "T",
            "L",
            "L",
            "I",
            "P",
            "G",
            "S",
            "L",
            "S",
            "S",
            "E",
            "L",
            "L",
            "R",
            "D",
            "L",
            "S",
            "P",
            "R",
        ]
        result = self._align(short, long, tie_break_suffix=True)
        assert len(result) == len(long)


# ---------------------------------------------------------------------------
# align_scores
# ---------------------------------------------------------------------------


class TestAlignScores:

    def test_gap_positions_get_min_score(self):
        predicted = ["A", "-", "B"]
        scores = [0.9, 0.8]
        result = align_scores(predicted, scores, "-")
        assert result[1] == Constants.min_score

    def test_non_gap_positions_preserve_scores(self):
        predicted = ["A", "-", "B"]
        scores = [0.9, 0.8]
        result = align_scores(predicted, scores, "-")
        assert result[0] == pytest.approx(0.9)
        assert result[2] == pytest.approx(0.8)

    def test_output_length_equals_predicted(self):
        predicted = ["A", "-", "B", "-", "C"]
        scores = [0.9, 0.8, 0.7]
        result = align_scores(predicted, scores, "-")
        assert len(result) == len(predicted)

    def test_no_gaps(self):
        predicted = ["A", "B", "C"]
        scores = [0.1, 0.2, 0.3]
        result = align_scores(predicted, scores, "-")
        assert result == pytest.approx(scores)

    def test_all_gaps(self):
        predicted = ["-", "-", "-"]
        scores = []
        result = align_scores(predicted, scores, "-")
        assert result == [Constants.min_score, Constants.min_score, Constants.min_score]


# ---------------------------------------------------------------------------
# align_tokens_with_gaps
# ---------------------------------------------------------------------------


class TestAlignTokensWithGaps:

    def test_empty_predicted_returns_all_gaps(self):
        predicted, ground_truth, scores = align_tokens_with_gaps(
            [], ["A", "B", "C"], []
        )
        assert predicted == ["-", "-", "-"]
        assert ground_truth == ["A", "B", "C"]
        assert scores == [Constants.min_score, Constants.min_score, Constants.min_score]

    def test_equal_length_returns_unchanged(self):
        p = ["A", "B"]
        g = ["A", "B"]
        s = [0.9, 0.8]
        predicted, ground_truth, scores = align_tokens_with_gaps(p, g, s)
        assert predicted == p
        assert ground_truth == g
        assert scores == s

    def test_predicted_shorter_gaps_inserted_into_predicted(self):
        predicted, ground_truth, scores = align_tokens_with_gaps(
            ["A", "B"], ["A", "X", "B"], [0.9, 0.8]
        )
        assert len(predicted) == len(ground_truth)
        assert "-" in predicted
        assert ground_truth == ["A", "X", "B"]

    def test_predicted_longer_gaps_inserted_into_ground_truth(self):
        predicted, ground_truth, scores = align_tokens_with_gaps(
            ["A", "X", "B"], ["A", "B"], [0.9, 0.5, 0.8]
        )
        assert len(predicted) == len(ground_truth)
        assert "-" in ground_truth
        assert predicted == ["A", "X", "B"]

    def test_scores_realigned_when_predicted_shorter(self):
        predicted, ground_truth, scores = align_tokens_with_gaps(
            ["A", "B"], ["A", "X", "B"], [0.9, 0.8]
        )
        assert len(scores) == len(predicted)
        assert Constants.min_score in scores

    def test_scores_unchanged_when_predicted_longer(self):
        original_scores = [0.9, 0.5, 0.8]
        predicted, ground_truth, scores = align_tokens_with_gaps(
            ["A", "X", "B"], ["A", "B"], original_scores
        )
        assert scores == original_scores

    def test_output_sequences_equal_length(self):
        predicted, ground_truth, scores = align_tokens_with_gaps(
            ["A", "B", "C"], ["A", "X", "Y", "B", "C"], [0.9, 0.8, 0.7]
        )
        assert len(predicted) == len(ground_truth) == len(scores)

    def test_custom_gap_marker(self):
        predicted, ground_truth, scores = align_tokens_with_gaps(
            ["A", "B"], ["A", "X", "B"], [0.9, 0.8], gap="*"
        )
        assert "*" in predicted
        assert "-" not in predicted
