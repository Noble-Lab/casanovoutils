from os import PathLike
from typing import Optional

import fire
import pyteomics.mztab
import numpy as np
import dataclasses
import matplotlib.pyplot as plt

from ..casanovoutils import prec_cov, get_aa_matches

@dataclasses.dataclass
class GraphPrecCov:
    """
    Plot and compare precision-coverage curves for peptide identification
    results.

    This class provides utilities to extract predicted sequences and ground
    truth sequences from MzTab (or MGF) files, compute amino-acid-level
    correctness, and visualize the resulting precision-coverage (Prec-Cov)
    curves with AUPC (area under the precision-coverage curve) values in the
    legend.

    Multiple datasets can be added to the same figure via ``add_peptides()``,
    enabling comparative benchmarking.

    Parameters
    ----------
    fig_width : float, optional
        Width of the matplotlib figure in inches. Defaults to 3.0.
    fig_height : float, default=3.0
        Height of the matplotlib figure in inches. Defaults to 3.0.
    fig_dpi : int, default=150
        Resolution of the figure in dots per inch.
    legend_border : bool, default=False
        Whether to draw a border frame around the plot legend.
    legend_location : str, default="lower left"
        Location of the legend on the plot. Any valid matplotlib location string
        (e.g., "upper right", "lower left"). Defaults to "lower left".

    To plot multiple datasets and save the combined precision-coverage plot in a
    single command-line invocation, make each plotting method return ``self`` so
    that Fire can chain subcommands on the same instance. For example:

    Command Line Example (Fire CLI)
    -------------------------------
    To plot multiple datasets and save the combined precision-coverage plot via
    the command line, chain multiple calls to ``add-peptides`` before invoking
    the ``save`` function. For example:

    .. code-block:: bash

        python script.py graph-prec-cov \
            --fig_width 6 --fig_height 4 \
            --legend_location upper right \
            add-peptides \
                --mztab_path modelA.mztab \
                --name ModelA \
                --ground_truth_col true_sequence \
            add-peptides \
                --mztab_path modelB.mztab \
                --name ModelB \
                --ground_truth_mgf ground_truthB.mgf \
            add-peptides \
                --mztab_path modelC.mztab \
                --name ModelC \
                --ground_truth_col peptide \
                --residues_path custom_residues.yaml \
            save plot.png

    In this example:

    - ``GraphPrecCov`` is instantiated **once** at the beginning.
    - ``ModelA`` uses a ground truth sequence column from the MzTab file.
    - ``ModelB`` extracts ground truth sequences from an MGF file.
    - ``ModelC`` uses a ground truth column and a **custom residue mass table**.
    - The legend location is modified using a class-level argument.
    - All three datasets are added to the same plot.
    - The combined figure is saved to ``plot.png`` at the end.
    - All operations occur within a **single process**, so state is preserved.
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
        name: str,
        pred_col: str = "sequence",
        score_col: str = "search_engine_score[1]",
        ground_truth_col: Optional[str] = None,
        ground_truth_mgf: Optional[PathLike] = None,
        residues_path: Optional[PathLike] = None,
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
        psm_df = pyteomics.mztab.MzTab(mztab_path).spectrum_match_table
        aa_matches, ground_truth, pred = get_aa_matches(
            psm_df,
            pred_col=pred_col,
            ground_truth_col=ground_truth_col,
            ground_truth_mgf=ground_truth_mgf,
            residues_path=residues_path,
        )

        scores = np.full_like(pred, -1.0, dtype=float)
        scores[pred != None] = psm_df[score_col].to_numpy(dtype=float)
        pep_correct = np.array([curr[1] for curr in aa_matches])

        prec, cov, aupc = prec_cov(scores, pep_correct)
        self.ax.plot(cov, prec, label=f"{name}: {aupc:.4f}")
        self.ax.legend(loc=self.legend_location, frameon=self.legend_border)

    def add_amino_acids(
        self,
        mztab_path: PathLike,
        name: str,
        pred_col: str = "sequence",
        scores_col: str = "opt_ms_run[1]_aa_scores",
        ground_truth_col: Optional[str] = None,
        ground_truth_mgf: Optional[PathLike] = None,
        residues_path: Optional[PathLike] = None,
    ) -> None:
        """
        Add a precision-coverage curve trace at the amino-acid level.

        This function extracts per-amino-acid correctness indicators and
        per-amino-acid scores for each PSM, aggregates them across all PSMs in
        the MzTab file, and plots an amino-acid-level precision-coverage curve
        with the AUPC value included in the legend.

        Parameters
        ----------
        mztab_path : PathLike
            Path to an MzTab file containing peptide-spectrum matches (PSMs).
        name : str
            Name of the dataset; used in the plot legend.
        pred_col : str, optional
            Column name in the MzTab file containing predicted peptide
            sequences. Defaults to "sequence". This is passed through to
            ``get_aa_matches()``.
        scores_col : str, optional
            Column name in the MzTab file containing per-amino-acid scores for
            each PSM. The values are expected to be comma-separated strings of
            scores. Defaults to "opt_ms_run[1]_aa_scores".
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
        """
        psm_df = pyteomics.mztab.MzTab(mztab_path).spectrum_match_table
        aa_matches, _, pred = get_aa_matches(
            psm_df,
            pred_col=pred_col,
            ground_truth_col=ground_truth_col,
            ground_truth_mgf=ground_truth_mgf,
            residues_path=residues_path,
        )

        aa_correct, aa_scores = [], []
        for curr_aa_match, curr_scores, curr_pred in zip(
            aa_matches, psm_df[scores_col], pred
        ):
            curr_aa_match = curr_aa_match[0]
            if curr_pred is None:
                curr_scores = np.zeros_like(curr_aa_match, dtype=float)
            else:
                curr_scores = np.array([float(c) for c in curr_scores.split(",")])

            match_len = min(len(curr_scores), len(curr_aa_match))
            aa_correct.append(curr_aa_match[:match_len])
            aa_scores.append(curr_scores[:match_len])

        aa_correct = np.concatenate(aa_correct)
        aa_scores = np.concatenate(aa_scores)
        prec, cov, aupc = prec_cov(aa_scores, aa_correct)
        self.ax.plot(cov, prec, label=f"{name}: {aupc:.4f}")
        self.ax.legend(loc=self.legend_location, frameon=self.legend_border)

    def clear(self) -> None:
        """
        Reset the figure and axes to a blank precision-coverage plot.

        Creates a new matplotlib figure and axes using the configured figure
        size and DPI, and sets axis labels for precision and coverage.

        This is automatically called on initialization, but can be used manually
        to start a new plot.

        Returns
        -------
        None
        """
        self.fig, self.ax = plt.subplots(
            figsize=(self.fig_width, self.fig_height), dpi=self.fig_dpi
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

def main() -> None:
    """CLI entry"""
    fire.Fire(GraphPrecCov)

if __name__ == "__main__":
    main()
