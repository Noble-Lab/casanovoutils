import polars as pl


class Constants:
    """
    Global constants for column names and sentinel values.

    ground_truth_sequence_column : str
        Name of the column holding ground truth peptide sequences.
    aa_scores_column : str
        Name of the column holding per-amino-acid score strings.
    pep_score_column : str
        Name of the column holding peptide-level search engine scores.
    aa_idx_column : str
        Name of the column holding per-amino-acid positional indices,
        added during alignment and explosion.
    precision_column : str
        Name of the column holding cumulative precision values computed
        by ``calc_precision_coverage``.
    coverage_column : str
        Name of the column holding cumulative coverage values computed
        by ``calc_precision_coverage``.
    min_score : float
        Sentinel score assigned to gap positions during sequence alignment.
    """

    ground_truth_sequence_column: str = "mgf_seq"
    aa_scores_column: str = "mztab_opt_ms_run[1]_aa_scores"
    pep_score_column: str = "mztab_search_engine_score[1]"
    aa_idx_column: str = "pc_aa_idx"
    precision_column: str = "pc_precision"
    coverage_column: str = "pc_coverage"
    predicted_tokens: str = "mztab_tokens"
    ground_truth_tokens: str = "mgf_tokens"
    min_score: float = -1.0

    @staticmethod
    def get_pred_sequence_column(df: pl.DataFrame) -> str:
        """
        Determine the name of the predicted sequence column.

        Checks for the presence of a ProForma-formatted prediction column first,
        falling back to the plain mzTab sequence column if it is absent.

        Parameters
        ----------
        df : pl.DataFrame
            A DataFrame expected to contain either
            ``"mztab_opt_ms_run[1]_proforma"`` or ``"mztab_sequence"``.

        Returns
        -------
        str
            The name of the predicted sequence column.
        """
        if "mztab_opt_ms_run[1]_proforma" in df.columns:
            pred_col = "mztab_opt_ms_run[1]_proforma"
        else:
            pred_col = "mztab_sequence"

        return pred_col
