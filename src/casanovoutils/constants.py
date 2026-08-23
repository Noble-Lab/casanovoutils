"""
Shared constants for column names and sentinel values used across the package.
"""

import polars as pl


class Constants:
    """
    Global constants for column names and sentinel values.

    ground_truth_sequence_column : str
        Name of the column holding ground truth peptide sequences.
    aa_scores_column : str
        Canonical name of the column holding per-amino-acid score strings
        (``"mztab_opt_global_aa_scores"``).  Use
        :meth:`get_aa_scores_column` to detect whichever variant is
        present in a DataFrame loaded from an mzTab file.
    aa_scores_column_legacy : str
        Legacy column name emitted by older versions of pyteomics
        (``"mztab_opt_ms_run[1]_aa_scores"``), kept for backwards
        compatibility with code that referenced the old constant value.
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
    aa_scores_column: str = "mztab_opt_global_aa_scores"
    aa_scores_column_legacy: str = "mztab_opt_ms_run[1]_aa_scores"
    pep_score_column: str = "mztab_search_engine_score[1]"
    aa_idx_column: str = "pc_aa_idx"
    precision_column: str = "pc_precision"
    coverage_column: str = "pc_coverage"
    predicted_tokens: str = "mztab_tokens"
    ground_truth_tokens: str = "mgf_tokens"
    min_score: float = -1.0

    @staticmethod
    def get_aa_scores_column(df: pl.DataFrame) -> str:
        """
        Determine the name of the per-amino-acid scores column.

        The mzTab spec requires optional columns that apply globally to be named
        ``opt_global_*``.  Older versions of pyteomics incorrectly expanded these
        to ``opt_ms_run[1]_*``; newer versions preserve the spec-correct name.
        This method checks for both conventions so that casanovoutils works
        regardless of the pyteomics version used to read the mzTab file.

        Parameters
        ----------
        df : pl.DataFrame
            A DataFrame expected to contain the per-amino-acid scores column.

        Returns
        -------
        str
            The name of the per-amino-acid scores column present in *df*.

        Raises
        ------
        ValueError
            If neither ``"mztab_opt_global_aa_scores"`` nor
            ``"mztab_opt_ms_run[1]_aa_scores"`` is found in *df*.
        """
        if "mztab_opt_global_aa_scores" in df.columns:
            return "mztab_opt_global_aa_scores"
        if "mztab_opt_ms_run[1]_aa_scores" in df.columns:
            return "mztab_opt_ms_run[1]_aa_scores"
        raise ValueError(
            "Cannot find per-amino-acid scores column in DataFrame. "
            "Expected 'mztab_opt_global_aa_scores' (mzTab spec opt_global_* name, "
            "current pyteomics) or 'mztab_opt_ms_run[1]_aa_scores' "
            "(legacy opt_ms_run[1]_* expansion from older pyteomics). "
            f"Found columns: {df.columns}"
        )

    @staticmethod
    def get_pred_sequence_column(df: pl.DataFrame) -> str:
        """
        Determine the name of the predicted sequence column.

        Checks for ProForma-formatted prediction columns first (preferred,
        because they carry modification annotations), falling back to the
        plain mzTab sequence column if none is found.

        Two ProForma column naming conventions are supported:

        - ``"mztab_opt_global_cv_MS:1003169_proforma_peptidoform_sequence"``
          — written by current Casanovo (uses the CV-term name and the
          spec-correct ``opt_global_*`` prefix).
        - ``"mztab_opt_ms_run[1]_proforma"`` — written by older Casanovo
          versions.

        Parameters
        ----------
        df : pl.DataFrame
            A DataFrame expected to contain a predicted sequence column.

        Returns
        -------
        str
            The name of the predicted sequence column.
        """
        if "mztab_opt_global_cv_MS:1003169_proforma_peptidoform_sequence" in df.columns:
            return "mztab_opt_global_cv_MS:1003169_proforma_peptidoform_sequence"
        if "mztab_opt_ms_run[1]_proforma" in df.columns:
            return "mztab_opt_ms_run[1]_proforma"
        return "mztab_sequence"
