import dataclasses
from typing import Iterable

import numpy as np
import tqdm

Tokens = np.ndarray[str | float]


class UnknownTokenError(ValueError):
    pass


@dataclasses.dataclass
class PeptideSplitter:
    """
    Split peptide sequences into residue tokens (optionally mapped to masses).

    This class implements a greedy, vocabulary-based tokenizer for peptide
    sequences. It is designed for cases where "residues" are not necessarily
    single amino-acid characters, but may include multi-character tokens such as
    inline PTMs or alternative encodings (e.g., ``"M[Oxidation]"``, ``"ac-"``,
    ``"cm"``, etc.).

    Tokenization is performed by repeatedly matching the *longest* token in the
    vocabulary against the current prefix of the remaining sequence. Tokens are
    sorted by decreasing length during initialization to ensure that longer
    tokens take priority over their shorter prefixes (e.g., matching
    ``"M[Oxidation]"`` before ``"M"``).

    Parameters
    ----------
    residues : dict[str, float]
        Mapping from residue tokens to their mass values in Dalton. The keys
        form the token vocabulary used for splitting.
    progress : bool, default=False
        If True, ``split_all`` shows a progress bar.
    """

    residues: dict[str, float]
    progress: bool = False

    def __post_init__(self) -> None:
        """
        Prepare the internal vocabulary ordering for longest-prefix matching.

        Tokens are sorted in decreasing order of token length so that longer
        tokens are matched before their prefixes.
        """
        tokens = self.residues.keys()
        tokens = sorted(tokens, key=len, reverse=True)
        self._tokens = list(tokens)

    def split_seq(self, seq: str, to_mass: bool = False) -> Tokens | None:
        """
        Split a single peptide sequence into residue tokens.

        Tokenization is performed greedily using longest-prefix matching against
        the vocabulary provided by ``residues``. At each step, the longest token
        that matches the current prefix is consumed and appended to the output.

        Parameters
        ----------
        seq : str
            Peptide sequence encoded as a concatenation of residue tokens.
        to_mass : bool, default=False
            If True, map each matched residue token to its corresponding mass
            using the ``residues`` dictionary.

        Returns
        -------
        tokens : numpy.ndarray
            1D NumPy array of tokens. If ``to_mass=False`` the array contains
            strings. If ``to_mass=True`` the array contains floats.

        Raises
        ------
        UnknownTokenError
            If no token in the vocabulary matches the current prefix of ``seq``.

        Examples
        --------
        >>> residues = {"A": 71.03711, "M[Oxidation]": 147.0354, "M": 131.04049}
        >>> splitter = PeptideSplitter(residues)
        >>> splitter.split_seq("AM[Oxidation]M")
        array(['A', 'M[Oxidation]', 'M'], dtype='<U...')
        >>> splitter.split_seq("AM[Oxidation]M", to_mass=True)
        array([ 71.03711, 147.0354 , 131.04049])
        """
        if seq is None:
            return None
        
        # TODO: this should probably be made faster with a trie but residue
        # vocabularies tend to be short so this is fine for now
        out = []
        while len(seq) > 0:
            for token in self._tokens:
                if seq.startswith(token):
                    out.append(token)
                    seq = seq[len(token) :]
                    break
            else:
                raise UnknownTokenError(f"Prefix not found in vocabulary for: {seq}")

        if to_mass:
            out = list(map(self.residues, out))

        return np.array(out, dtype=np.float64 if to_mass else np.str_)

    def split_all(
        self, seqs: Iterable[str] | str, strict: bool = True, to_mass: bool = False
    ) -> list[Tokens | None]:
        """
        Split one or more peptide sequences into residue tokens.

        This is a convenience wrapper around ``split_seq`` that supports
        processing multiple sequences, optionally with a progress bar, and with
        configurable handling of unknown tokens.

        Parameters
        ----------
        seqs : Iterable[str] or str
            A single peptide sequence or an iterable of sequences to tokenize.
        strict : bool, default=True
            If True, unknown tokens raise ``UnknownTokenError`` immediately.
            If False, sequences containing unknown tokens yield ``None`` in the
            output list and tokenization continues for subsequent sequences.
        to_mass : bool, default=False
            If True, output arrays contain residue masses instead of token
            strings.

        Returns
        -------
        split : list[numpy.ndarray or None]
            A list of token arrays (one per input sequence). If
            ``strict=False``, entries corresponding to sequences that fail
            tokenization will be ``None``.

        Raises
        ------
        UnknownTokenError
            If ``strict=True`` and any sequence contains an unknown token.

        Examples
        --------
        >>> splitter.split_all(["PEPTIDE", "UNKNOWN"], strict=False)
        [array([...]), None]
        """
        if isinstance(seqs, str):
            seqs = [seqs]

        if self.progress:
            seqs = tqdm.tqdm(seqs)

        out = []
        for seq in seqs:
            try:
                out.append(self.split_seq(seq, to_mass=to_mass))
            except UnknownTokenError as e:
                if strict:
                    raise e
                else:
                    out.append(None)

        return out


@dataclasses.dataclass
class AAMatches:
    aa_matches: np.ndarray
    scores: np.ndarray
    pep_match: bool


def align_to_longer(
    short: np.ndarray,
    long: np.ndarray,
    numeric: bool,
    mass_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    m = len(long)
    n = len(short)

    if n == m:
        if numeric:
            correct = np.abs(short - long) < mass_tol
        else:
            correct = short == long

        is_gap = np.zeros(m, dtype=bool)
        return correct.astype(bool), is_gap

    if numeric:
        scores = np.abs(np.subtract.outer(short, long)) < mass_tol
    else:
        scores = short[:, None] == long[None, :]

    dp = np.empty_like(scores, dtype=int)

    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if j == m - 1:
                dp[i, j] = int(scores[i, j])
                continue

            gap_score = dp[i, j + 1]
            align_score = (
                int(scores[i, j])
                if i == n - 1
                else int(scores[i, j]) + dp[i + 1, j + 1]
            )
            dp[i, j] = max(gap_score, align_score)

    correct = np.zeros(m, dtype=bool)
    is_gap = np.ones(m, dtype=bool)
    i, j = 0, 0
    
    while i < n and j < m:
        gap_score = dp[i, j + 1] if (j + 1) < m else -1

        if j == m - 1:
            align_score = int(scores[i, j])
        elif i == n - 1:
            align_score = int(scores[i, j])
        else:
            align_score = int(scores[i, j]) + dp[i + 1, j + 1]

        if align_score >= gap_score and align_score == dp[i, j]:
            is_gap[j] = False
            correct[j] = bool(scores[i, j])
            i += 1
            j += 1
        else:
            j += 1

    return correct, is_gap


def aa_match(
    ground_truth: Tokens,
    pred: Tokens | None,
    scores: np.ndarray | None,
    by_mass: bool = False,
    mass_tol: float = 1e-8,
) -> AAMatches:
    if pred is None:
        return AAMatches(
            aa_matches=np.zeros_like(ground_truth, dtype=bool),
            scores=np.full_like(ground_truth, -1.0, dtype=np.float64),
            pep_match=False
        )
    
    gt_longer = len(ground_truth) > len(pred)

    if gt_longer:
        short = pred
        long = ground_truth
    else:
        short = ground_truth
        long = pred

    is_correct, is_gap = align_to_longer(short, long, by_mass, mass_tol)

    if gt_longer:
        new_scores = np.full_like(is_correct, -1.0, is_correct, dtype=np.float64)
        new_scores[~is_gap] = scores
        scores = new_scores

    return AAMatches(aa_matches=is_correct, scores=scores, pep_match=is_correct.all())


def aa_match_all(
    ground_truth: Iterable[Tokens],
    pred: Iterable[Tokens | None],
    scores: Iterable[np.ndarray | None],
    by_mass: bool = False,
    mass_tol: float = 1e-8,
    progress: bool = True,
) -> Iterable[AAMatches]:
    out = []
    iter = zip(ground_truth, pred, scores)
    if progress:
        iter = tqdm.tqdm(iter, desc="Checking peptides", unit="PSMs")

    for curr_gt, curr_pred, curr_scores in iter:
        out.append(
            aa_match(
                curr_gt, curr_pred, curr_scores, by_mass=by_mass, mass_tol=mass_tol
            )
        )

    return out
