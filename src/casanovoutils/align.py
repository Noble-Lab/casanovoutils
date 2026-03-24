"""
Sequence alignment utilities for inserting gaps into token lists.

Provides a lightweight alignment algorithm that inserts gap markers into the
shorter of two token sequences to maximise the number of exact position-wise
matches with the longer sequence. Intended for aligning predicted and ground
truth peptide token sequences prior to per-position scoring.

The alignment is implemented as a backwards-filled DP table
(:func:`get_aligned_dp_array`) with a greedy traceback
(:func:`recover_solution`). A helper (:func:`align_scores`) keeps per-token
score arrays in sync after gaps are inserted. The top-level entry point
(:func:`align_tokens_with_gaps`) handles length-equality short-circuits and
dispatches to the correct argument order depending on which sequence is shorter.
"""

from .constants import Constants

def get_aligned_dp_array(short: list[str], long: list[str]) -> list[list[int]]:
    """
    Build a DP scoring table for aligning two token sequences.
    
    Fills the table backwards from the end of both sequences, scoring
    +1 for a match and 0 for a mismatch or gap. Cells where ``long``
    is exhausted before ``short`` are marked as illegal with a score of -1.
    
    Parameters
    ----------
    short : list[str]
        The shorter token sequence, into which gaps will be inserted.
    long : list[str]
        The longer token sequence, which is never modified.
        
    Returns
    -------
    list[list[int]]
        A 2D table of shape ``(len(short), len(long))`` where entry
        ``[i][j]`` is the best achievable alignment score from position
        ``i`` in ``short`` and ``j`` in ``long`` to the end of both sequences.
    """
    s = len(short)
    l = len(long)
    dp = [[0] * l for _ in range(s)]

    for curr_s in range(s - 1, -1, -1):
        for curr_l in range(l - 1, -1, -1):
            is_match = int(short[curr_s] == short[curr_l])
            if curr_s == s - 1 and curr_l == l - 1:
                # The end, yay
                score = is_match
            elif curr_s == s - 1:
                # We've consumed all of s, so we have to add a gap
                score = is_match + dp[curr_s][curr_l + 1]
            elif curr_l == l - 1:
                # We've consumed all of l without reaching the end
                # of s, which is an illegal state
                score = -1
            else:
                # We can either add a gap, consuming just l, or not add
                # a gap, consuming both s and l
                score = is_match + max(
                    dp[curr_s][curr_l + 1], dp[curr_s + 1][curr_l + 1]
                )

            dp[curr_s][curr_l] = score

    return dp


def recover_solution(
    dp: list[list[int]],
    short: list[str],
    long: list[str],
    gap: str,
    tie_break_suffix: bool,
) -> list[str]:
    """
    Traceback through a DP table to reconstruct the gap-inserted version of ``short``.
    
    At each step, decides whether to emit a gap or the next token from ``short``
    by comparing the scores of the two paths. Ties are broken according to
    ``tie_break_suffix``.
    
    Parameters
    ----------
    dp : list[list[int]]
        A DP table as returned by :func:`get_aligned_dp_array`.
    short : list[str]
        The shorter token sequence being aligned.
    long : list[str]
        The longer token sequence that ``short`` is being aligned to.
    gap : str
        The gap marker to insert, e.g. ``"-"``.
    tie_break_suffix : bool
        If ``True``, prefer consuming a real token on a tie (suffix-biased);
        if ``False``, prefer inserting a gap (prefix-biased).
        
    Returns
    -------
    list[str]
        A copy of ``short`` with gap markers inserted, of the same length as ``long``.
    """
    s, l = len(short), len(long)
    curr_s, curr_l = 0, 0
    short_aligned = []

    while True:
        if curr_s == s - 1:
            # We have to add a gap
            gap_wins = True
        else:
            if dp[curr_s + 1][curr_l + 1] < dp[curr_s][curr_l + 1]:
                gap_wins = True
            elif dp[curr_s + 1][curr_l + 1] == dp[curr_s][curr_l + 1]:
                gap_wins = not tie_break_suffix
            else:
                gap_wins = False

        next_short = gap if gap_wins else short[curr_s]
        short_aligned.append(next_short)

        if curr_s == s - 1 and curr_l == l - 1:
            break

        curr_l += 1

        if not gap_wins:
            curr_s += 1

    return short_aligned


def align_scores(predicted: list[str], scores: list[float], gap: str) -> list[float]:
    """
    Realign a score array to match a gap-inserted token sequence.
    
    After gaps are inserted into ``predicted``, the original ``scores`` array
    no longer corresponds index-for-index. This function inserts ``min_score``
    placeholders at every gap position to restore that correspondence.
    
    Parameters
    ----------
    predicted : list[str]
        The gap-inserted predicted token sequence.
    scores : list[float]
        The original scores, parallel to ``predicted`` before gap insertion.
    gap : str
        The gap marker used in ``predicted``, e.g. ``"-"``.
        
    Returns
    -------
    list[float]
        A score array of the same length as ``predicted``, with
        ``Constants.min_score`` at every gap position.
    """
    scores_idx = 0
    scores_aligned = []

    for curr in predicted:
        if curr == gap:
            scores_aligned.append(Constants.min_score)
        else:
            scores_aligned.append(scores[scores_idx])
            scores_idx += 1

    return scores_aligned


def align_tokens_with_gaps(
    predicted: list[str],
    ground_truth: list[str],
    scores: list[float],
    gap: str = "-",
    tie_break_suffix: bool = True,
) -> tuple[list[str], list[str], list[float]]:
    """
    Align two token sequences by inserting gaps to maximise exact matches.
    Gaps are always inserted into the shorter sequence, leaving the longer
    one untouched. Scoring awards +1 for a match, 0 for a mismatch or gap.
    
    Parameters
    ----------
    predicted : list[str]
        The predicted token sequence.
    ground_truth : list[str]
        The ground truth token sequence.
    scores : list[float]
        Per-token scores parallel to ``predicted``.
    gap : str, optional
        The gap marker to insert. Defaults to ``"-"``.
    tie_break_suffix : bool, optional
        Passed to :func:`recover_solution` to control tie-breaking behaviour.
        Defaults to ``True``.
        
    Returns
    -------
    tuple[list[str], list[str], list[float]]
        A three-tuple of ``(aligned_predicted, aligned_ground_truth, aligned_scores)``,
        all of equal length.
    """
    if len(predicted) == 0:
        aligned_predicted = [gap] * len(ground_truth)
        aligned_scores = [Constants.min_score] * len(ground_truth)
        return aligned_predicted, ground_truth, aligned_scores

    if len(predicted) == len(ground_truth):
        return predicted, ground_truth, scores

    if len(predicted) < len(ground_truth):
        dp = get_aligned_dp_array(predicted, ground_truth)
        predicted = recover_solution(dp, predicted, ground_truth, gap, tie_break_suffix)
        scores = align_scores(predicted, scores, gap)
    else:
        dp = get_aligned_dp_array(ground_truth, predicted)
        ground_truth = recover_solution(
            dp, ground_truth, predicted, gap, tie_break_suffix
        )

    return predicted, ground_truth, scores
