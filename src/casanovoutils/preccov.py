from os import PathLike

import fire
import dataclasses
import matplotlib.pyplot as plt

from .evaluate import get_ground_truth, prec_cov


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

    Fire CLI Example
    ----------------
    .. code-block:: bash

        python script.py graph-prec-cov \
            --fig_width 6 --fig_height 4 \
            --legend_location "upper right" \
            add-peptides modelA.mztab modelA.mgf ModelA \
            add-peptides modelB.mztab modelB.mgf ModelB \
            save plot.png

    All commands operate on the same instance, so state (the accumulated
    curves) is preserved.
    """

    fig_width: float = 3.0
    fig_height: float = 3.0
    fig_dpi: int = 150
    legend_border: bool = False
    legend_location: str = "lower left"
    ax_x_label: str = "Coverage"
    ax_y_label: str = "Precision"
    ax_title: str = ""

    def __post_init__(self):
        """Initialize an empty plot upon instantiation."""
        self.clear()

    def add_peptides(
        self,
        mztab_path: PathLike,
        mgf_path: PathLike,
        name: str,
        replace_i_l: bool = True,
    ) -> None:
        """
        Add a peptide-level precision–coverage curve for one dataset.

        Predicted peptide scores and correctness labels are derived using
        ``get_ground_truth()``, then a precision–coverage curve is computed with
        ``prec_cov()`` and added to the current plot.

        Parameters
        ----------
        mztab_path : PathLike
            Path to an MzTab file containing peptide-spectrum match (PSM)
            predictions and associated scores.
        mgf_path : PathLike
            Path to the MGF file used to derive ground-truth peptide sequences.
            This is passed directly to ``get_ground_truth()``.
        name : str
            Dataset name used in the legend.
        replace_i_l : bool, default=True
            Whether to treat isoleucine (I) and leucine (L) as equivalent when
            determining correctness.

        Returns
        -------
        None
        """
        predictions_df = get_ground_truth(mztab_path, mgf_path, replace_i_l=replace_i_l)
        prec, cov, aupc = prec_cov(
            predictions_df["pep_score"].to_numpy(),
            predictions_df["pep_correct"].to_numpy(),
        )

        self.ax.plot(cov, prec, label=f"{name} {aupc:.3f}")
        self.ax.legend(loc="lower left")

    def add_amino_acids(*args, **kwargs) -> None:
        raise NotImplementedError()

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
        self.ax.set_title(f"{self.ax_title} (Amino Acid)")

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


def main() -> None:
    """CLI entry"""
    fire.Fire(GraphPrecCov)


if __name__ == "__main__":
    main()

    fig_width: float = 3.0
    fig_height: float = 3.0
    fig_dpi: int = 150
    legend_border: bool = False
    legend_location: str = "lower left"
    ax_x_label: str = "Coverage"
    ax_y_label: str = "Precision"
    ax_title: str = ""

    def __post_init__(self):
        """Initialize an empty plot upon instantiation."""
        self.clear()

    def add_peptides(
        self,
        mztab_path: PathLike,
        mgf_path: PathLike,
        name: str,
        replace_i_l: bool = True,
    ) -> None:
        """
        Add a precision-coverage curve trace for a dataset.

        This function extracts predicted peptide sequences and prediction scores
        from an MzTab file, obtains ground truth sequences either from the same
        file or a corresponding MGF file, evaluates amino-acid-level
        correctness, and plots the precision-coverage curve with an AUPC value
        in the legend.

        Parameters
        ----------
        mztab_path : PathLike
            Path to an MzTab file containing peptide-spectrum matches (PSMs).
        name : str
            Name of the dataset; used in the plot legend.
        pred_col : str, optional
            Column name in the MzTab file containing predicted peptide sequences.
            Defaults to "sequence".
        score_col : str, optional
            Column name in the MzTab file containing prediction scores used to rank
            PSMs. Defaults to "search_engine_score[1]".
        ground_truth_col : str, optional
            Column name in the MzTab containing ground truth peptide sequences.
            If provided, ground truth is taken from MzTab.
        ground_truth_mgf : PathLike, optional
            Path to an MGF file from which to extract ground truth sequences.
            Required if ``ground_truth_col`` is None.
        residues_path : PathLike, optional
            Path to a YAML file containing residue masses. Passed to
            ``get_residues()`` for amino acid evaluation.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If neither ``ground_truth_col`` nor ``ground_truth_mgf`` is provided.
        """
        predictions_df = get_ground_truth(mztab_path, mgf_path, replace_i_l=replace_i_l)
        prec, cov, aupc = prec_cov(
            predictions_df["pep_score"].to_numpy(),
            predictions_df["pep_correct"].to_numpy(),
        )

        self.ax.plot(cov, prec, label=f"{name} {aupc:.3f}")
        self.ax.legend(loc="lower left")

    def add_amino_acids(*args, **kwargs) -> None:
        raise NotImplementedError()

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
        # Amino-acid-level figure
        self.fig, self.ax = plt.subplots(
            figsize=(self.fig_width, self.fig_height),
            dpi=self.fig_dpi,
        )

        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel(self.ax_x_label)
        self.ax.set_ylabel(self.ax_y_label)
        self.ax.set_title(f"{self.ax_title} (Amino Acid)")

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


def main() -> None:
    """CLI entry"""
    fire.Fire(GraphPrecCov)


if __name__ == "__main__":
    main()
