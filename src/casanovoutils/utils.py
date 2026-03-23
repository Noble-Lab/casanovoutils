import logging
from os import PathLike
from typing import Any, Optional

import tqdm
import polars as pl
import pyteomics.mgf
import depthcharge.tokenizers
import yaml

import pathlib
import shutil
from os import PathLike
from typing import Any

PyteomicsSpectrum = dict[str, dict[str, Any] | list[Any]]


def process_spectrum(spectrum: PyteomicsSpectrum) -> dict[str, dict[str, Any]]:
    params = spectrum["params"]
    params["n_peaks"] = len(spectrum["m/z array"])
    return params


def get_mgf_psms_df(mgf_path: PathLike) -> pl.DataFrame:
    logging.info("Reading file %s", str(mgf_path))
    mgf_iter = pyteomics.mgf.read(str(mgf_path), use_index=False, convert_arrays=0)
    mgf_iter = tqdm.tqdm(mgf_iter, desc="reading params", unit="spectra")
    mgf_iter = map(process_spectrum, mgf_iter)

    spectrum_df = pl.from_dicts(mgf_iter)
    spectrum_df = spectrum_df.rename({c: f"mgf_{c}" for c in spectrum_df.columns})

    return spectrum_df


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


def tokenize_helper(
    seq: str,
    tokenizer: depthcharge.tokenizers.PeptideTokenizer,
    combine_n_term: bool = True,
) -> list[str]:
    out = tokenizer.split(seq)

    if out[0].startswith("[") and combine_n_term:
        n_term_token = out[0]
        out = out[1:]
        out[0] = f"{n_term_token}{out[0]}"

    return out


def tokenize_sequences(
    data_df: pl.DataFrame,
    seq_column: str,
    out_prefix: Optional[str] = None,
    combine_n_term: bool = True,
    residues_path: Optional[PathLike] = None,
) -> pl.DataFrame:
    residues = get_residues(residues_path)
    tokenizer = depthcharge.tokenizers.peptides.MskbPeptideTokenizer(residues=residues)

    if out_prefix is None:
        out_prefix = seq_column.split("_")[0]

    combine_fun = lambda seq: tokenize_helper(
        seq, tokenizer, combine_n_term=combine_n_term
    )

    data_df = data_df.with_columns(
        pl.col(seq_column)
        .map_elements(combine_fun, return_dtype=pl.List(pl.Utf8))
        .alias(f"{out_prefix}_tokens")
    ).with_columns(
        pl.col(f"{out_prefix}_tokens").len().alias(f"{out_prefix}_sequence_len")
    )

    return data_df
