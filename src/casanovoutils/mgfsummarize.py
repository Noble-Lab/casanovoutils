import pathlib
from os import PathLike
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

from .utils import get_mgf_psms_df, tokenize_sequences

MgfPath = PathLike | pl.DataFrame


def read_spectra(mgf_path: MgfPath) -> pl.DataFrame:
    if isinstance(mgf_path, pl.DataFrame):
        return mgf_path

    mgf_path = pathlib.Path(mgf_path)
    if mgf_path.suffix == ".mgf":
        return get_mgf_psms_df(mgf_path)
    elif mgf_path.suffix in [".parquet", ".pq"]:
        return pl.read_parquet(mgf_path)
    else:
        raise ValueError(f"Unsupported file type for file: {mgf_path}")


def histplot_wrapper(
    data: pl.DataFrame,
    x_value: str,
    x_label: str,
    figsize: int | float = 4,
    dpi: int | float = 150,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)

    sns.histplot(
        data=data.to_pandas(),
        x=x_value,
        discrete=True,
        ax=ax,
    )

    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    fig.tight_layout()

    return fig, ax


def barplot_wrapper(
    data: pl.DataFrame,
    x_value: str,
    x_label: str,
    figsize: int | float = 4,
    dpi: int | float = 150,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)

    sns.barplot(
        data=data.to_pandas(),
        x=x_value,
        y="count",
        ax=ax,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")
    fig.tight_layout()

    return fig, ax


def graph_peptide_frequency(
    mgf_path: MgfPath,
    graph_save_path: Optional[PathLike] = None,
    params_save_path: Optional[PathLike] = None,
) -> pl.DataFrame:
    mgf_path = get_mgf_psms_df(mgf_path)
    if params_save_path is not None:
        mgf_path.write_parquet(params_save_path)

    results_df = (
        read_spectra(mgf_path)
        .select(pl.col("mgf_seq").value_counts())
        .unnest("mgf_seq")
    )

    fig, ax = histplot_wrapper(results_df, "count", "Peptide Frequency")

    if graph_save_path is not None:
        fig.savefig(graph_save_path)
    else:
        fig.show(warn=True)

    return results_df


def graph_charge_state_frequency(
    mgf_path: MgfPath,
    graph_save_path: Optional[PathLike] = None,
    params_save_path: Optional[PathLike] = None,
) -> pl.DataFrame:
    mgf_path = get_mgf_psms_df(mgf_path)
    if params_save_path is not None:
        mgf_path.write_parquet(params_save_path)

    results_df = (
        mgf_path.select(pl.col("mgf_charge").list.first().value_counts())
        .unnest("mgf_charge")
        .sort(by="mgf_charge")
    )

    fig, ax = barplot_wrapper(results_df, "mgf_charge", "Charge State")

    if graph_save_path is not None:
        fig.savefig(graph_save_path)
    else:
        fig.show(warn=True)

    return results_df


def graph_sequence_lengths(
    mgf_path: MgfPath,
    graph_save_path: Optional[PathLike] = None,
    params_save_path: Optional[PathLike] = None,
) -> pl.DataFrame:
    mgf_path = get_mgf_psms_df(mgf_path)
    if params_save_path is not None:
        mgf_path.write_parquet(params_save_path)

    results_df = tokenize_sequences(mgf_path, "mgf_seq")
    fig, ax = histplot_wrapper(results_df, "mgf_seq_len", "Sequence Length")

    if graph_save_path is not None:
        fig.savefig(graph_save_path)
    else:
        fig.show(warn=True)

    return results_df

