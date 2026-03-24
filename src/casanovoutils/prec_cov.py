"""
Precision-coverage computation for peptide and amino acid evaluation.

Provides an end-to-end pipeline that takes predicted and ground truth PSM
DataFrames, aligns token sequences with gap insertion, and computes cumulative
precision-coverage (Prec-Cov) curves. Results can be exported as DataFrames
or rendered as matplotlib figures.

The main entry points are:

- :func:`get_prec_cov_df` — builds a precision-coverage DataFrame at either
  peptide or amino acid level.
- :func:`graph_prec_cov` — plots pre-computed precision-coverage DataFrames.
- :class:`GraphPrecCov` — stateful plot builder, intended for programmatic or
  CLI use via Fire.

The module is also executable as a CLI via ``python -m casanovoutils.prec_cov``
(or the installed ``casanovoutils`` entry point), exposing ``get_pc_df`` and
``graph_prec_cov`` as subcommands.
"""

import dataclasses
import functools
import logging
import pathlib
from os import PathLike
from typing import Any, Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tqdm

from .align import align_tokens_with_gaps
from .constants import Constants
from .utils import (
    DfPath,
    configure_logging,
    get_ground_truth_df,
    read_dataframe,
    tokenize_sequences,
    write_dataframe,
)


@dataclasses.dataclass
class GraphPrecCov:
    """
    Plot and compare peptide-level precision-coverage (Prec-Cov) curves.

    This class accumulates multiple datasets onto a single precision-coverage
    plot. For each dataset, predicted peptide correctness and scores are
    extracted via ``get_ground_truth()``, and a precision-coverage curve is
    computed using ``prec_cov()``. The area under the precision-coverage curve
    (AUPC) is displayed in the legend.

    Designed for command-line use with Fire, multiple datasets can be added in
    a single process before saving or showing the figure.

    Parameters
    ----------
    fig_width : float, default=3.0
        Width of the matplotlib figure in inches.
    fig_height : float, default=3.0
        Height of the matplotlib figure in inches.
    fig_dpi : int, default=150
        Figure resolution in dots per inch.
    legend_border : bool, default=False
        Whether to draw a border around the legend frame.
    legend_location : str, default="lower left"
        Legend location string passed to ``matplotlib.axes.Axes.legend``.
    ax_x_label : str, default="Coverage"
        Label for the x-axis.
    ax_y_label : str, default="Precision"
        Label for the y-axis.
    ax_title : str, default=""
        Base title for the plot. "(Amino Acid)" is appended automatically.

    Notes
    -----
    Each call to ``add_peptides()`` adds a new curve to the same axes. Use
    ``clear()`` to reset the figure.

    All commands operate on the same instance, so state (the accumulated
    curves) is preserved.
    """

    fig_width: float = 4.0
    fig_height: float = 4.0
    fig_dpi: int = 150
    legend_border: bool = False
    legend_location: str = "lower left"
    ax_x_label: str = "Coverage"
    ax_y_label: str = "Precision"
    ax_title: str = ""

    def __post_init__(self):
        """Initialize an empty plot upon instantiation."""
        self.clear()

    def add_series(
        self,
        pc_df: pl.DataFrame,
        series_name: str,
        color: Optional[str] = None,
        linestyle: Optional[str] = None,
    ) -> None:
        """
        Add a precision-coverage curve for a single dataset to the plot.

        Extracts precision and coverage columns from ``pc_df``, computes the
        area under the precision-coverage curve (AUPC) via the trapezoidal
        rule, and plots the curve with ``series_name`` and the AUPC value
        in the legend label.

        Parameters
        ----------
        pc_df : pl.DataFrame
            A DataFrame containing ``Constants.precision_column`` and
            ``Constants.coverage_column`` columns, as produced by
            :func:`calc_precision_coverage`.
        series_name : str
            Display name for this dataset in the plot legend.
        color : str, optional
            Line color passed to ``matplotlib.axes.Axes.plot``. If ``None``
            (default), matplotlib's automatic color cycling is used.
        linestyle : str, optional
            Line style passed to ``matplotlib.axes.Axes.plot`` (e.g. ``"-"``,
            ``"--"``, ``":"``). If ``None`` (default), matplotlib's default
            solid line style is used.

        Returns
        -------
        None
        """
        precision = pc_df.get_column(Constants.precision_column).to_numpy()
        coverage = pc_df.get_column(Constants.coverage_column).to_numpy()
        aupc = np.trapz(precision, coverage)

        kwargs = {}
        if color is not None:
            kwargs["color"] = color
        if linestyle is not None:
            kwargs["linestyle"] = linestyle

        self.ax.plot(coverage, precision, label=f"{series_name} {aupc:.4f}", **kwargs)
        self.ax.legend(loc="lower left")

    def clear(self) -> None:
        """
        Reset the figures and axes to blank precision-coverage plots.

        Creates two matplotlib figures + axes:
        1) amino-acid-level precision/coverage plot
        2) peptide-level precision/coverage plot

        Returns
        -------
        None
        """
        self.fig, self.ax = plt.subplots(
            figsize=(self.fig_width, self.fig_height),
            dpi=self.fig_dpi,
        )

        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel(self.ax_x_label)
        self.ax.set_ylabel(self.ax_y_label)
        self.ax.set_title(self.ax_title)

    def save(self, save_path: PathLike) -> None:
        """
        Save the current plot to a file.

        Parameters
        ----------
        save_path : PathLike
            Output file path. The file extension (e.g., .png, .pdf, .svg)
            determines the format written by matplotlib.

        Returns
        -------
        None
        """
        self.fig.tight_layout()
        self.fig.savefig(save_path)

    def show(self) -> None:
        """
        Display the current precision-coverage plot.

        Returns
        -------
        None
        """
        self.fig.show()


def mutate_row_as_dict(tie_break_suffix: bool, row: dict[str, Any]) -> dict[str, Any]:
    """
    Align predicted and ground truth token sequences within a single row dict.

    Calls :func:`align_tokens_with_gaps` on the predicted tokens, ground truth
    tokens, and per-amino-acid scores from the row, then mutates the row in
    place with the aligned sequences, aligned scores, and a positional index
    list.

    Parameters
    ----------
    tie_break_suffix : bool
        Passed through to :func:`align_tokens_with_gaps`. Controls tie-breaking
        behaviour when the gap and no-gap paths score equally during traceback.
    row : dict[str, Any]
        A single row represented as a dict, as produced by
        ``DataFrame.iter_rows(named=True)``.

    Returns
    -------
    dict[str, Any]
        The same row dict with ``Constants.predicted_tokens``,
        ``Constants.ground_truth_tokens``,
        ``Constants.aa_scores_column``, and ``Constants.aa_idx_column``
        replaced by their gap-aligned counterparts.
    """
    aligned_predicted, aligned_ground_truth, aligned_scores = align_tokens_with_gaps(
        row[Constants.predicted_tokens],
        row[Constants.ground_truth_tokens],
        row[Constants.aa_scores_column],
        tie_break_suffix=tie_break_suffix,
    )

    row[Constants.predicted_tokens] = aligned_predicted
    row[Constants.ground_truth_sequence_column] = aligned_ground_truth
    row[Constants.aa_scores_column] = aligned_scores
    row[Constants.aa_idx_column] = list(range(len(aligned_predicted)))

    return row


def calc_precision_coverage(pc_df: pl.DataFrame, score_col: str) -> pl.DataFrame:
    """
    Compute cumulative precision and coverage curves sorted by score.

    Sorts the DataFrame by ``score_col`` in descending order, computes a
    boolean correctness column indicating where the predicted sequence matches
    the ground truth, then calculates cumulative precision and coverage at
    each rank threshold.

    Parameters
    ----------
    pc_df : pl.DataFrame
        Input DataFrame containing predicted and ground truth sequence columns
        and a score column.
    score_col : str
        Name of the column to sort by. Typically either the peptide-level
        score column or the per-amino-acid score column depending on whether
        evaluation is at peptide or amino acid level.

    Returns
    -------
    pl.DataFrame
        The input DataFrame sorted by ``score_col`` with three additional
        columns: ``"pc_is_correct"`` (bool), ``"pc_precision"`` (float),
        and ``"pc_coverage"`` (float).
    """
    logging.debug("Computing precision-coverage using score column '%s'", score_col)

    pc_df = pc_df.sort(score_col, descending=True)
    pc_df = pc_df.with_columns(
        (
            pl.col(Constants.ground_truth_tokens) == pl.col(Constants.predicted_tokens)
        ).alias("pc_is_correct")
    )

    is_correct = pc_df.get_column("pc_is_correct").to_numpy()
    n_correct = is_correct.sum()
    logging.info(
        "Correctness: %d / %d (%.1f%%)",
        n_correct,
        len(is_correct),
        100 * n_correct / len(is_correct),
    )

    total_coverage = np.arange(1, len(is_correct) + 1)
    total_precision = np.cumsum(is_correct)
    precision = total_precision / total_coverage
    coverage = total_coverage / total_coverage[-1]

    logging.debug(
        "Precision-coverage curve: precision at full coverage = %.3f", precision[-1]
    )

    pc_df = pc_df.with_columns(
        pl.Series("pc_precision", precision), pl.Series("pc_coverage", coverage)
    )

    return pc_df


def load_ground_truth_df(
    ground_truth_df: Optional[DfPath],
    mgf_df: Optional[DfPath],
    mztab_df: Optional[DfPath],
) -> pl.DataFrame:
    """
    Load or construct a ground truth PSM DataFrame.

    If ``ground_truth_df`` is provided, it is loaded via :func:`read_dataframe`.
    Otherwise, the ground truth is constructed from the provided MGF and mzTab
    files via :func:`get_ground_truth_df`.

    Parameters
    ----------
    ground_truth_df : DfPath, optional
        Path to or an already-loaded ground truth DataFrame.
    mgf_df : DfPath, optional
        Path to or an already-loaded MGF PSM DataFrame. Required when
        ``ground_truth_df`` is ``None``.
    mztab_df : DfPath, optional
        Path to or an already-loaded mzTab DataFrame. Required when
        ``ground_truth_df`` is ``None``.

    Returns
    -------
    pl.DataFrame
        The loaded or constructed ground truth DataFrame.

    Raises
    ------
    ValueError
        If ``ground_truth_df`` is ``None`` and either ``mgf_df`` or
        ``mztab_df`` is also ``None``.
    """
    if ground_truth_df is None and (mgf_df is None or mztab_df is None):
        raise ValueError(
            "--mgf_df and --mztab_df must be provided if --ground_truth_df is not"
            " provided."
        )

    if ground_truth_df is not None:
        logging.info("Loading ground truth DataFrame from provided source")
        result = read_dataframe(ground_truth_df)
        logging.info("Loaded ground truth DataFrame with %d rows", len(result))
        return result

    logging.info("Constructing ground truth DataFrame from MGF and mzTab sources")
    result = get_ground_truth_df(mgf_df, mztab_df)
    logging.info("Constructed ground truth DataFrame with %d rows", len(result))
    return result


def fill_null_columns(df: pl.DataFrame, pred_col: str) -> pl.DataFrame:
    """
    Replace null values in score and sequence columns with safe defaults.

    Fills nulls in the predicted sequence column and the per-amino-acid scores
    column with empty strings, and nulls in the peptide score column with
    ``-1.0``.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing the columns to fill.
    pred_col : str
        Name of the predicted sequence column.

    Returns
    -------
    pl.DataFrame
        The DataFrame with null values replaced.
    """
    n_null_pred = df[pred_col].null_count()
    n_null_aa = df[Constants.aa_scores_column].null_count()
    n_null_pep = df[Constants.pep_score_column].null_count()

    if any(n > 0 for n in [n_null_pred, n_null_aa, n_null_pep]):
        logging.debug(
            "Filling nulls — %s: %d, %s: %d, %s: %d",
            pred_col,
            n_null_pred,
            Constants.aa_scores_column,
            n_null_aa,
            Constants.pep_score_column,
            n_null_pep,
        )

    return df.with_columns(
        pl.col([pred_col, Constants.aa_scores_column]).fill_null("")
    ).with_columns(pl.col(Constants.pep_score_column).fill_null(-1.0))


def tokenize_and_parse_scores(
    df: pl.DataFrame,
    pred_col: str,
    residues_path: Optional[PathLike],
    replace_isoleucine_with_leucine: bool,
) -> pl.DataFrame:
    """
    Tokenize predicted and ground truth sequences and parse per-AA score strings.

    Applies :func:`tokenize_sequences` to both the ground truth and predicted
    sequence columns, then parses the comma-separated per-amino-acid score
    strings in the aa scores column into lists of floats.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing sequence and score columns.
    pred_col : str
        Name of the predicted sequence column.
    residues_path : PathLike, optional
        Path to a residue mass YAML file. If ``None``, the bundled
        ``residues.yaml`` is used.
    replace_isoleucine_with_leucine : bool
        If ``True``, isoleucine (I) is replaced with leucine (L) during
        tokenization, treating them as equivalent.

    Returns
    -------
    pl.DataFrame
        The DataFrame with added token columns and the aa scores column
        converted from comma-separated strings to lists of floats.
    """
    logging.info(
        "Tokenizing sequences (replace_isoleucine_with_leucine=%s)",
        replace_isoleucine_with_leucine,
    )
    logging.debug(
        "Tokenizing ground truth column '%s'", Constants.ground_truth_sequence_column
    )
    df = tokenize_sequences(
        df,
        seq_column=Constants.ground_truth_sequence_column,
        residues_path=residues_path,
        replace_isoleucine_with_leucine=replace_isoleucine_with_leucine,
    )

    logging.debug("Tokenizing predicted column '%s'", pred_col)
    df = tokenize_sequences(
        df,
        seq_column=pred_col,
        residues_path=residues_path,
        replace_isoleucine_with_leucine=replace_isoleucine_with_leucine,
    )

    logging.debug(
        "Parsing per-amino-acid score strings in column '%s'",
        Constants.aa_scores_column,
    )
    aa_score_fun = lambda x: [] if x == "" else [float(c) for c in x.split(",")]
    return df.with_columns(
        pl.col(Constants.aa_scores_column).map_elements(
            aa_score_fun, return_dtype=pl.List(pl.Float64)
        )
    )


def align_and_explode(
    df: pl.DataFrame,
    tie_break_suffix: bool,
) -> pl.DataFrame:
    """
    Align predicted and ground truth token sequences and explode to per-AA rows.

    Iterates over each row, aligns the predicted and ground truth token
    sequences with gap insertion via :func:`mutate_row_as_dict`, then explodes
    the resulting list columns so that each row corresponds to a single amino
    acid position.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with tokenized predicted and ground truth sequence
        columns and parsed per-amino-acid scores.
    tie_break_suffix : bool
        Passed through to :func:`mutate_row_as_dict`. Controls tie-breaking
        behavior when the gap and no-gap paths score equally during traceback.

    Returns
    -------
    pl.DataFrame
        A DataFrame exploded to one row per aligned amino acid position,
        with gap characters inserted where sequences do not align.
    """
    logging.info(
        "Aligning %d rows at amino acid level (tie_break_suffix=%s)",
        len(df),
        tie_break_suffix,
    )

    row_iter = df.iter_rows(named=True)
    row_iter = tqdm.tqdm(row_iter, desc="Aligning Amino Acids", total=len(df))
    row_iter = map(
        functools.partial(mutate_row_as_dict, tie_break_suffix),
        row_iter,
    )
    df = pl.from_dicts(row_iter)

    logging.debug("Exploding aligned columns to per-amino-acid rows")
    df = df.explode(
        [
            Constants.predicted_tokens,
            Constants.ground_truth_tokens,
            Constants.aa_scores_column,
            Constants.aa_idx_column,
        ]
    )
    logging.info("Exploded to %d per-amino-acid rows", len(df))
    return df


def get_prec_cov_df(
    ground_truth_df: Optional[DfPath] = None,
    mgf_df: Optional[DfPath] = None,
    mztab_df: Optional[DfPath] = None,
    residues_path: Optional[DfPath] = None,
    replace_isoleucine_with_leucine: bool = True,
    aa_level: bool = False,
    align_tie_beak_suffix: bool = True,
    out_path: Optional[PathLike] = None,
) -> pl.DataFrame:
    """
    Build a precision-coverage DataFrame from predicted and ground truth PSMs.

    Loads or constructs a ground truth DataFrame, tokenizes both predicted and
    ground truth sequences, parses per-amino-acid scores, and computes
    precision-coverage metrics. When ``aa_level`` is ``True``, sequences are
    first aligned with gap insertion and then exploded so that each row
    represents a single amino acid position rather than a full peptide.

    Parameters
    ----------
    ground_truth_df : DfPath, optional
        Path to or an already-loaded ground truth DataFrame. If ``None``,
        both ``mgf_df`` and ``mztab_df`` must be provided and the ground
        truth DataFrame will be constructed via :func:`get_ground_truth_df`.
    mgf_df : DfPath, optional
        Path to or an already-loaded MGF PSM DataFrame. Required when
        ``ground_truth_df`` is ``None``.
    mztab_df : DfPath, optional
        Path to or an already-loaded mzTab DataFrame. Required when
        ``ground_truth_df`` is ``None``.
    residues_path : DfPath, optional
        Path to a residue mass YAML file passed through to
        :func:`tokenize_sequences`. If ``None``, the bundled
        ``residues.yaml`` is used.
    replace_isoleucine_with_leucine : bool, optional
        If ``True`` (default), isoleucine (I) is replaced with leucine (L)
        during tokenization, treating them as equivalent.
    aa_level : bool, optional
        If ``True``, perform per-amino-acid alignment via gap insertion and
        explode the DataFrame so each row corresponds to a single amino acid
        position. If ``False`` (default), metrics are computed at the peptide
        level using the peptide-level score column.
    align_tie_beak_suffix : bool, optional
        Passed through to the alignment step when ``aa_level`` is ``True``.
        Controls tie-breaking behavior when the gap and no-gap paths score
        equally during traceback. Defaults to ``True``.
    out_path : PathLike, optional
        If provided, the resulting DataFrame is written to this path before
        being returned. The format is inferred from the file extension.

    Returns
    -------
    pl.DataFrame
        A DataFrame with precision and coverage metrics. At peptide level,
        each row is one PSM; at amino acid level (``aa_level=True``), each
        row is one aligned amino acid position.

    Raises
    ------
    ValueError
        If ``ground_truth_df`` is ``None`` and either ``mgf_df`` or
        ``mztab_df`` is also ``None``.
    """
    logging.info(
        "Building precision-coverage DataFrame (aa_level=%s, align_tie_beak_suffix=%s)",
        aa_level,
        align_tie_beak_suffix,
    )

    pc_df = load_ground_truth_df(ground_truth_df, mgf_df, mztab_df)

    pred_col = Constants.get_pred_sequence_column(pc_df)
    logging.debug("Using predicted sequence column '%s'", pred_col)

    pc_df = fill_null_columns(pc_df, pred_col)
    pc_df = tokenize_and_parse_scores(
        pc_df, pred_col, residues_path, replace_isoleucine_with_leucine
    )

    if aa_level:
        pc_df = align_and_explode(pc_df, align_tie_beak_suffix)

    score_col = Constants.aa_scores_column if aa_level else Constants.pep_score_column
    logging.debug("Computing precision-coverage with score column '%s'", score_col)
    pc_df = calc_precision_coverage(pc_df, score_col)

    logging.info(
        "Precision-coverage DataFrame complete: %d rows, %d columns",
        len(pc_df),
        len(pc_df.columns),
    )

    if out_path is not None:
        logging.info("Writing precision coverage DataFrame to %s", str(out_path))
        write_dataframe(pc_df, out_path)

    return pc_df


def graph_prec_cov(*pc_df_paths: PathLike, out_path: Optional[PathLike] = None) -> None:
    """
    Plot precision-coverage curves from one or more pre-computed DataFrames.

    Loads each DataFrame from ``pc_df_paths``, adds it as a series to a
    :class:`GraphPrecCov` plot using the file stem as the series name, and
    then either saves the figure, displays it, or both.

    Parameters
    ----------
    *pc_df_paths : PathLike
        One or more paths to DataFrames containing
        ``Constants.precision_column`` and ``Constants.coverage_column``
        columns, as produced by :func:`get_prec_cov_df`. The file stem of each
        path is used as the series label in the legend.
    out_path : PathLike, optional
        If provided, the figure is saved to this path. The file extension
        determines the format (e.g. ``.png``, ``.pdf``, ``.svg``).

    Returns
    -------
    None

    Warns
    -----
    Logs a warning if the plot cannot be displayed, which typically occurs
    when no graphical backend is available (e.g. in a headless environment).
    In that case, saving via ``out_path`` still works normally.
    """
    graph_pc = GraphPrecCov()
    for curr_path in pc_df_paths:
        curr_path = pathlib.Path(curr_path)
        curr_name = curr_path.stem
        curr_df = read_dataframe(curr_path)
        graph_pc.add_series(curr_df, curr_name)

    if out_path is not None:
        graph_pc.save(out_path)

    try:
        graph_pc.show()
    except:
        logging.warning("Tried to show precision coverage plot.")
        logging.warning("Is a graphical backend installed?")
        logging.warning(
            "Note: If you are just trying to save a plot you can ignore this."
        )


def main() -> None:
    """CLI entry"""
    configure_logging()
    fire.Fire({"get_pc_df": get_prec_cov_df, "graph_prec_cov": graph_prec_cov})


if __name__ == "__main__":
    main()
