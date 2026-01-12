import dataclasses
import random
import itertools
import pathlib
import shutil
import os
from os import PathLike
from typing import Any, Iterable, Optional, List, Dict
from collections import defaultdict

import numpy as np
import tqdm
import fire
import pyteomics.mgf
import logging
import re
import pyteomics.mztab
import matplotlib.pyplot as plt
import casanovo.denovo.evaluate
import pandas as pd
import yaml


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
        self.fig.savefig(save_path)

    def show(self) -> None:
        """
        Display the current precision-coverage plot.

        Returns
        -------
        None
        """
        self.fig.show()

@dataclasses.dataclass 
class TrainTestValidationSplitting: 
    """
    Split MGF files by peptide sequence into train/test/val sets.

    This script recursively finds all MGF files in an input directory, groups
    spectra by peptide sequence, splits peptides into train/test/val sets, and
    writes the corresponding spectra to separate output MGF files.
    """
    random_seed: int | float = 42

    def find_mgf_files(
        input_dir: pathlib.Path
    ) -> Iterable[pathlib.Path]:
        """
        Recursively find all MGF files in a directory.
        
        Parameters
        ----------
        input_dir : pathlib.Path
            Root directory to search
            
        Returns
        -------
        List[pathlib.Path]
            List of paths to MGF files
        """
        mgf_files = Iterable(input_dir.rglob("*.mgf"))
        return mgf_files

    def split_peptides(
        self,
        peptides: List[str],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> tuple[List[str], List[str], List[str]]:
        """
        Split peptide sequences into train/val/test sets.
        
        Parameters
        ----------
        peptides : List[str]
            List of unique peptide sequences
        train_ratio : float
            Fraction of peptides for training set
        val_ratio : float
            Fraction of peptides for validation set
        test_ratio : float
            Fraction of peptides for test set
        random_seed : int
            Random seed for reproducibility
            
        Returns
        -------
        tuple[List[str], List[str], List[str]]
            (train_peptides, val_peptides, test_peptides)
        """
        rng = np.random.default_rng(self.random_seed)
        peptides = list(peptides)
        rng.shuffle(peptides)
        
        n = len(peptides)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_peptides = peptides[:train_end]
        val_peptides = peptides[train_end:val_end]
        test_peptides = peptides[val_end:]
        
        return train_peptides, val_peptides, test_peptides


    def write_split_mgf(
        peptide_dict: Dict[str, List[Dict[str, Any]]],
        peptide_list: List[str],
        output_path: pathlib.Path
    ) -> None:
        """
        Write spectra for a list of peptides to an MGF file.
        
        Parameters
        ----------
        peptide_dict : Dict[str, List[Dict[str, Any]]]
            Dictionary mapping peptides to their spectra
        peptide_list : List[str]
            List of peptides to include in this split
        output_path : pathlib.Path
            Output MGF file path
        """
        spectra = []
        for peptide in peptide_list:
            spectra.extend(peptide_dict[peptide])
        
        print(f"Writing {len(spectra)} spectra to {output_path}")
        pyteomics.mgf.write(spectra, str(output_path))


    def split(
        self,
        input_dir: PathLike, 
        output_dir: PathLike, 
        train_ratio: float, 
        test_ratio: float, 
        val_ratio: float,
    ):
        os.mkdir(output_dir)
        
        mgf_files = self.find_mgf_files(input_dir)
        peptide_dict = get_pep_dict_mgf(mgf_files)
        
        train_peptides, val_peptides, test_peptides = self.split_peptides(
            list(peptide_dict.keys()),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        
        self.write_split_mgf(peptide_dict, train_peptides, output_dir / "train")
        self.write_split_mgf(peptide_dict, val_peptides, output_dir / "val")
        self.write_split_mgf(peptide_dict, test_peptides, output_dir / "test")

@dataclasses.dataclass
class DownsampleMS:
    """
    Utilities for downsampling data from one or more PSM files.

    Parameters
    ----------
    random_seed : int | float, default=42
        Random seed used for reproducible downsampling.
    """

    random_seed: int | float = 42

    def downsample_mgf_pep(
        self,
        *infiles: Iterable[PathLike],
        outfile: PathLike = "out.mgf",
        k: int = 1,
        shuffle: bool = True,
    ) -> None:
        """
        Downsample MGF files by sampling up to ``k`` PSMs per unique peptide
        sequence, optionally shuffling, and writing the resulting spectra to an
        output MGF file.

        Parameters
        ----------
        *infiles : Iterable[PathLike]
            One or more input MGF file paths. All spectra from all files are
            aggregated before downsampling.
        outfile : PathLike, default="out.mgf"
            Output MGF file path. Defaults to "out.mgf".
        k : int, default=1
            Maximum number of PSMs to sample per peptide sequence.
        shuffle : bool, default=True
            Whether to shuffle the pooled sample of spectra prior to writing.

        Returns
        -------
        None
        """
        random.seed(self.random_seed)
        pep_dict = get_pep_dict_mgf(infiles)

        for pep, psms in tqdm.tqdm(
            pep_dict.items(), desc="Sampling peptides", unit="peptide"
        ):
            pep_dict[pep] = random.sample(psms, min(len(psms), k))

        spectra = list(itertools.chain.from_iterable(pep_dict.values()))
        if shuffle:
            random.shuffle(spectra)

        pyteomics.mgf.write(
            tqdm.tqdm(spectra, desc=f"Writing output file {outfile}", unit="psm")
        )


def get_pep_dict_mgf(mgf_files: Iterable[PathLike]) -> dict[str, list[dict[str, Any]]]:
    """
    Read one or more MGF files and group spectra by peptide sequence.

    Parameters
    ----------
    mgf_files : Iterable[PathLike]
        Iterable of paths to MGF files. Each file is parsed into PSMs.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        A mapping from peptide sequence (str) to a list of corresponding
        pyteomics MGF spectrum dictionaries.
    """
    mgf_iter = tqdm.tqdm(
        pyteomics.mgf.from_iterable(mgf_files), desc=f"Reading mgf files", unit="psm"
    )

    out = {}
    for curr in mgf_iter:
        seq = curr["params"]["seq"]
        if curr["params"]["seq"] not in out:
            out[seq] = []
        out[seq].append(curr)

    return out


def prec_cov(
    scores: np.ndarray, is_correct: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the precision-coverage curve and its area-under-curve (AUPC) for a
    set of scored predictions.

    Parameters
    ----------
    scores : np.ndarray
        1D array of prediction scores, where higher values indicate greater
        confidence.
    is_correct : np.ndarray
        1D boolean or binary array indicating whether each prediction is correct
        (1/True) or incorrect (0/False). Must be the same length as ``scores``.

    Returns
    -------
    precision : np.ndarray
        Precision values at each coverage step after sorting by score. Length
        ``N``.
    coverage : np.ndarray
        Coverage values, normalized to the range [0, 1], where ``coverage[i]``
        is the fraction of samples included up to index ``i`` in the ranked
        list.
    aupc : float
        Area under the precision-coverage curve.
    """
    sort_idx = np.argsort(scores)[::-1]
    is_correct = is_correct[sort_idx]
    total_coverage = np.arange(1, len(is_correct) + 1)
    total_precision = np.cumsum(is_correct)

    precision = total_precision / total_coverage
    coverage = total_coverage / total_coverage[-1]
    aupc = np.trapz(precision, coverage)
    return precision, coverage, aupc


def get_residues(residues_path: Optional[PathLike] = None) -> dict[str, float]:
    """
    Load a mapping of amino acid residue names to masses from a YAML file.

    If ``residues_path`` is not provided, the function loads a default
    ``residues.yaml`` file located in the same directory as this module.

    Parameters
    ----------
    residues_path : PathLike, optional
        Path to a YAML file containing residue mass information.
        If ``None`` (default), the bundled ``residues.yaml`` file is used.

    Returns
    -------
    dict[str, float]
        A dictionary mapping residue identifiers (typically one-letter or
        multi-character amino acid codes) to their corresponding masses.
    """
    if residues_path is None:
        residues_path = pathlib.Path(__file__).parent / "residues.yaml"
    with open(residues_path) as f:
        return yaml.safe_load(f)


def dump_residues(destination_path: PathLike) -> None:
    """
    Copy the default ``residues.yaml`` file included with this package to a
    specified destination.

    Parameters
    ----------
    destination_path : PathLike
        Path to copy the YAML file to. May be a directory or a file path.

    Returns
    -------
    None
    """
    residues_path = pathlib.Path(__file__).parent / "residues.yaml"
    shutil.copy(residues_path, destination_path)


def get_aa_matches(
    mztab_path: PathLike | pd.DataFrame,
    pred_col: str = "sequence",
    ground_truth_col: Optional[str] = None,
    ground_truth_mgf: Optional[PathLike] = None,
    residues_path: Optional[PathLike] = None,
) -> tuple[tuple[np.ndarray, bool], np.ndarray, np.ndarray]:
    """
    Extract predicted peptide sequences, ground-truth sequences, and amino-acid
    match information for a batch of PSMs.

    This function loads peptide-spectrum matches (PSMs) from either an mzTab
    file or a pre-existing pandas DataFrame, retrieves predicted peptide
    sequences, obtains ground-truth sequences from an mzTab column or an MGF
    file, aligns the predictions to the correct spectrum indices when needed,
    and computes amino-acid-level correctness using Casanovo's
    ``aa_match_batch`` evaluator.

    Parameters
    ----------
    mztab_path : PathLike or pandas.DataFrame
        Path to an mzTab file containing a ``spectrum_match_table`` section, or
        an already-loaded DataFrame containing equivalent columns.
    pred_col : str, optional
        Name of the column containing predicted peptide sequences.
        Defaults to ``"sequence"``.
    ground_truth_col : str, optional
        Name of a column in the mzTab file containing ground-truth peptide
        sequences. Mutually exclusive with ``ground_truth_mgf``.
    ground_truth_mgf : PathLike, optional
        Path to an MGF file containing ground-truth peptide sequences encoded
        using ``SEQ=<peptide>`` lines. If provided, the ground truth is read
        from the MGF file and aligned to mzTab PSM rows via ``spectra_ref``.
    residues_path : PathLike, optional
        Path to a residue-mass YAML file used by ``get_residues()`` to
        construct an amino-acid mass dictionary for Casanovo's evaluator.

    Returns
    -------
    aa_matches : tuple[np.ndarray, bool]
        Per-PSM amino-acid match results produced by
        ``casanovo.denovo.evaluate.aa_match_batch``.
        For each PSM:
            - ``aa_matches[i][0]`` is a 1D array of per-AA correctness
              values (e.g., 1 = correct, 0 = incorrect).
            - ``aa_matches[i][1]`` is a Boolean peptide-level correctness flag.
    ground_truth : np.ndarray
        Array of ground-truth peptide sequences aligned to the PSMs. If the
        ground truth originates from an MGF file, this array is ordered by
        spectrum index and matches the mzTab rows via ``spectra_ref``.
    pred : np.ndarray
        Array of predicted peptide sequences aligned to the PSMs. If an MGF
        ground-truth file is used, predictions are re-ordered so that
        ``pred[i]`` corresponds to ``ground_truth[i]`` for the same spectrum.

    Raises
    ------
    ValueError
        If neither ``ground_truth_col`` nor ``ground_truth_mgf`` is provided.
    """
    if not isinstance(mztab_path, pd.DataFrame):
        psm_df = pyteomics.mztab.MzTab(mztab_path).spectrum_match_table
    else:
        psm_df = mztab_path

    pred = psm_df[pred_col].to_numpy()
    if ground_truth_col is None and ground_truth_mgf is None:
        raise ValueError(
            "Either a ground truth mztab column or ground truth mgf file must be"
            " provided"
        )
    elif ground_truth_col is not None:
        ground_truth = psm_df[ground_truth_col].to_numpy()
    else:
        ground_truth = []
        with open(ground_truth_mgf) as f:
            for line in tqdm.tqdm(f, desc=f"Reading mgf file: {ground_truth_mgf}"):
                if line.startswith("SEQ="):
                    ground_truth.append(line.removeprefix("SEQ=").strip())

        spectra_idx = (
            psm_df["spectra_ref"].str[len("ms_run[1]:index=") :].apply(int).to_numpy()
        )

        ground_truth = np.array(ground_truth)
        pred_aligned = np.full_like(ground_truth, None, dtype=str)
        pred_aligned[spectra_idx] = pred
        pred = pred_aligned

    aa_dict = get_residues(residues_path)
    aa_matches, _, _ = casanovo.denovo.evaluate.aa_match_batch(
        tqdm.tqdm(pred, desc="Checking peptides", unit="peptide"),
        ground_truth,
        aa_dict,
    )

    return aa_matches, ground_truth, pred


def main() -> None:
    """CLI Entry"""
    fire.Fire(
        {
            "downsample-ms": DownsampleMS,
            "dump-residues": dump_residues,
            "graph-prec-cov": GraphPrecCov,
            "train/test/validation-split": TrainTestValidationSplitting,
        }
    )
