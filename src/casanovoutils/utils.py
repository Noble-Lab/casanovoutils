import logging
import pathlib
import sys
from os import PathLike
from typing import Any, Optional

import depthcharge.tokenizers
import fire
import polars as pl
import polars.selectors as cs
import pyteomics.mgf
import pyteomics.mztab
import tqdm
import yaml

PyteomicsSpectrum = dict[str, dict[str, Any] | list[Any]]
DfPath = PathLike | pl.DataFrame


def process_spectrum(
    spectrum: PyteomicsSpectrum, meta_data_only: bool = True
) -> dict[str, dict[str, Any]]:
    """
    Extract and augment parameter metadata from a single spectrum.

    Retrieves the ``params`` dict from a Pyteomics spectrum object and
    annotates it with the number of peaks in the spectrum. If ``meta_data_only``
    is ``False``, the intensity and m/z arrays are also included in the output.

    Parameters
    ----------
    spectrum : PyteomicsSpectrum
        A spectrum dict as returned by ``pyteomics.mgf.read``, containing
        at least a ``"params"`` key, an ``"m/z array"`` key, and an
        ``"intensity array"`` key.
    meta_data_only : bool, optional
        If ``True``, only scalar metadata is returned (no spectral arrays).
        If ``False``, ``"intensity_array"`` and ``"m_z_array"`` are added
        to the output dict.

    Returns
    -------
    dict[str, Any]
        The spectrum's parameter dict with an added ``"n_peaks"`` entry, and
        optionally ``"intensity_array"`` and ``"m_z_array"`` entries.
    """
    params = spectrum["params"]
    params["n_peaks"] = len(spectrum["m/z array"])

    if not meta_data_only:
        params["intensity_array"] = spectrum["intensity array"]
        params["m_z_array"] = spectrum["m/z array"]

    return params


def write_dataframe(data_df: pl.DataFrame, out_path: PathLike) -> None:
    """
    Write a DataFrame to a file, inferring the format from the extension.

    Parameters
    ----------
    data_df : pl.DataFrame
        The DataFrame to write.
    out_path : PathLike
        Destination path. The file format is inferred from the extension:
        ``.parquet`` / ``.pq`` for Parquet, ``.csv`` for comma-separated,
        and ``.tsv`` for tab-separated.

    Raises
    ------
    ValueError
        If the file extension is not one of the supported types.
    """
    out_path = pathlib.Path(out_path)
    if out_path.suffix in [".parquet", ".pq"]:
        data_df.write_parquet(out_path)
    elif out_path.suffix == ".csv":
        data_df.write_csv(out_path)
    elif out_path.suffix == ".tsv":
        data_df.write_csv(out_path, separator="\t")
    else:
        raise ValueError(f"Unsupported file type for file: {out_path}")


def get_mgf_psms_df(
    mgf_path: DfPath,
    out_path: Optional[PathLike] = None,
    meta_data_only: bool = True,
) -> pl.DataFrame:
    """
    Load PSM metadata from an MGF file into a Polars DataFrame.

    If ``mgf_path`` is already a :class:`polars.DataFrame`, it is returned
    as-is (and optionally written to ``out_path``). Otherwise, the MGF file
    is parsed with Pyteomics, per-spectrum parameters are extracted via
    :func:`process_spectrum`, and all columns are prefixed with ``mgf_``.

    Parameters
    ----------
    mgf_path : DfPath
        Path to an MGF file, or an already-loaded :class:`polars.DataFrame`.
    out_path : PathLike, optional
        If provided, the resulting DataFrame is written to this path before
        being returned. The format is inferred from the file extension via
        :func:`write_dataframe`.
    meta_data_only : bool, optional
        Passed through to :func:`process_spectrum`. If ``True`` (default),
        only scalar spectrum metadata is loaded (no m/z or intensity arrays).
        If ``False``, ``mgf_intensity_array`` and ``mgf_m_z_array`` columns
        are included in the returned DataFrame.

    Returns
    -------
    pl.DataFrame
        A DataFrame with one row per spectrum and columns prefixed with
        ``mgf_``, including an ``mgf_n_peaks`` column.

    Raises
    ------
    ValueError
        Propagated from :func:`write_dataframe` if ``out_path`` has an
        unsupported file extension.
    """
    if isinstance(mgf_path, pl.DataFrame):
        if out_path is not None:
            logging.info("Writing MGF DataFrame to %s", str(out_path))
            write_dataframe(mgf_path, out_path)

        logging.debug("mgf_path is already a DataFrame, returning as-is")
        return mgf_path

    logging.info("Reading MGF file %s", str(mgf_path))
    mgf_iter = pyteomics.mgf.read(str(mgf_path), use_index=False, convert_arrays=0)
    mgf_iter = tqdm.tqdm(mgf_iter, desc="reading params", unit="spectra")
    mgf_iter = map(
        lambda x: process_spectrum(x, meta_data_only=meta_data_only), mgf_iter
    )

    spectrum_df = pl.from_dicts(mgf_iter)
    logging.info("Read %d spectra from %s", len(spectrum_df), str(mgf_path))
    spectrum_df = spectrum_df.rename({c: f"mgf_{c}" for c in spectrum_df.columns})

    if out_path is not None:
        logging.info("Writing MGF DataFrame to %s", str(out_path))
        write_dataframe(spectrum_df, out_path)

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
    """
    Split a peptide sequence into tokens.

    Delegates to ``tokenizer.split`` and, when ``combine_n_term`` is ``True``,
    fuses a leading modification token (e.g. ``"[UNIMOD:x]"``) onto the first
    residue token so that the modification is not a stand-alone element.

    Parameters
    ----------
    seq : str
        A peptide sequence string, optionally containing modification
        annotations recognised by ``tokenizer``.
    tokenizer : depthcharge.tokenizers.PeptideTokenizer
        A tokenizer instance used to split the sequence.
    combine_n_term : bool, optional
        If ``True`` (default), merge a leading modification token with the
        first residue token.

    Returns
    -------
    list[str]
        Ordered list of token strings representing the peptide.
    """
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
    replace_isoleucine_with_leucine: bool = True,
) -> pl.DataFrame:
    """
    Tokenize a peptide sequence column and append token and length columns.

    Loads residue masses via :func:`get_residues`, constructs an
    ``MskbPeptideTokenizer``, and applies :func:`tokenize_helper` to each
    value in ``seq_column``. Two new columns are added to the DataFrame:
    ``{out_prefix}_tokens`` (a list of token strings) and
    ``{out_prefix}_sequence_len`` (the number of tokens).

    Parameters
    ----------
    data_df : pl.DataFrame
        Input DataFrame containing the sequence column to tokenize.
    seq_column : str
        Name of the column holding peptide sequence strings.
    out_prefix : str, optional
        Prefix for the output columns. If ``None`` (default), the portion
        of ``seq_column`` before the first underscore is used.
    combine_n_term : bool, optional
        Passed through to :func:`tokenize_helper`. If ``True`` (default),
        N-terminal modification tokens are merged with the first residue.
    residues_path : PathLike, optional
        Path to a residue mass YAML file. If ``None`` (default), the
        bundled ``residues.yaml`` is used.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with two additional columns:
        ``{out_prefix}_tokens`` and ``{out_prefix}_sequence_len``.
    """
    residues = get_residues(residues_path)
    tokenizer = depthcharge.tokenizers.peptides.MskbPeptideTokenizer(
        residues=residues,
        replace_isoleucine_with_leucine=replace_isoleucine_with_leucine,
    )

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


def read_dataframe(df_path: DfPath) -> pl.DataFrame:
    """
    Read a DataFrame from a file path, inferring the format from the extension.

    Parameters
    ----------
    df_path : DfPath
        Path to a ``.parquet`` / ``.pq``, ``.csv``, or ``.tsv`` file, or
        an already-loaded :class:`polars.DataFrame` which is returned as-is.

    Returns
    -------
    pl.DataFrame
        The loaded DataFrame.

    Raises
    ------
    ValueError
        If the file extension is not one of the supported types.
    """
    if isinstance(df_path, pl.DataFrame):
        return df_path

    df_path = pathlib.Path(df_path)
    if df_path.suffix in [".parquet", ".pq"]:
        return pl.read_parquet(df_path)
    elif df_path.suffix == ".csv":
        return pl.read_csv(df_path)
    elif df_path.suffix == ".tsv":
        return pl.read_csv(df_path, separator="\t")
    else:
        raise ValueError(f"Unsupported file type for file: {df_path}")


def get_mztab_df(
    mztab_path: DfPath, out_path: Optional[PathLike] = None
) -> pl.DataFrame:
    """
    Load the spectrum match table from an mzTab file into a Polars DataFrame.

    If ``mztab_path`` is already a DataFrame, it is returned as-is.
    Otherwise, the file is parsed with Pyteomics, converted from pandas,
    and given a row index. All columns are prefixed with ``mztab_``.

    Parameters
    ----------
    mztab_path : PathLike
        Path to an mzTab file, or an already-loaded :class:`polars.DataFrame`.
    out_path : PathLike, optional
        If provided, the resulting DataFrame is written to this path before
        being returned. The format is inferred from the file extension.

    Returns
    -------
    pl.DataFrame
        A DataFrame with one row per spectrum match and columns prefixed
        with ``mztab_``.
    """
    if isinstance(mztab_path, pl.DataFrame):
        if out_path is not None:
            logging.info("Writing mzTab DataFrame to %s", str(out_path))
            write_dataframe(mztab_path, out_path)

        logging.debug("mztab_path is already a DataFrame, returning as-is")
        return mztab_path

    logging.info("Reading mzTab file %s", str(mztab_path))
    result = pyteomics.mztab.MzTab(mztab_path).spectrum_match_table
    result = pl.from_pandas(result)
    logging.info("Read %d spectrum matches from %s", len(result), str(mztab_path))
    result = result.rename({c: f"mztab_{c}" for c in result.columns})

    if out_path is not None:
        logging.info("Writing mzTab DataFrame to %s", str(out_path))
        write_dataframe(result, out_path)

    return result


def get_ground_truth_df(
    mgf_path: DfPath,
    mztab_path: DfPath,
    out_path: Optional[PathLike] = None,
) -> pl.DataFrame:
    """
    Join MGF PSM metadata with mzTab spectrum match annotations.

    Loads both sources, aligns them on the MGF spectrum index encoded in
    the mzTab ``spectra_ref`` field, performs a left join, and drops all
    temporary ``tmp_`` columns from the result.

    Parameters
    ----------
    mgf_path : PathLike
        Path to the MGF file, or an already-loaded :class:`polars.DataFrame`.
    mztab_path : PathLike
        Path to the mzTab file, or an already-loaded :class:`polars.DataFrame`.
    out_path : PathLike, optional
        If provided, the resulting DataFrame is written to this path before
        being returned. The format is inferred from the file extension.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing all MGF parameter columns (prefixed ``mgf_``)
        left-joined with mzTab annotation columns (prefixed ``mztab_``).
    """
    logging.info("Building merged groundtruth DataFrame")
    mgf_df = get_mgf_psms_df(mgf_path)
    mgf_df = mgf_df.with_row_index("tmp_mgf_idx")

    index_fun = lambda x: int(x[len("ms_run[1]:index=") :])
    index_expr = (
        pl.col("mztab_spectra_ref")
        .map_elements(index_fun, return_dtype=pl.Int64)
        .alias("tmp_mgf_idx")
    )

    mztab_df = get_mztab_df(mztab_path).with_columns(index_expr)

    logging.debug(
        "Joining MGF (%d rows) with mzTab (%d rows)", len(mgf_df), len(mztab_df)
    )
    result_df = mgf_df.join(mztab_df, on="tmp_mgf_idx", how="left").select(
        cs.exclude("^tmp_.*$")
    )
    logging.info(
        "Merged DataFrame has %d rows and %d columns",
        len(result_df),
        len(result_df.columns),
    )

    if out_path is not None:
        logging.info("Writing merged DataFrame to %s", str(out_path))
        write_dataframe(result_df, out_path)

    return result_df


def main() -> None:
    """
    Configure logging and expose data loading functions as a CLI.

    Sets up a stdout logger at INFO level, then delegates to
    :func:`fire.Fire` which maps subcommands to their corresponding
    functions:

    - ``get_mgf_psms``   → :func:`get_mgf_psms_df`
    - ``get_mztab``      → :func:`get_mztab_df`
    - ``get_groundtruth`` → :func:`get_merged_groundtruth_df`

    Examples
    --------
    .. code-block:: bash

        python module.py get_mgf_psms path/to/file.mgf --out_path out.parquet
        python module.py get_mztab path/to/file.mztab --out_path out.parquet
        python module.py get_groundtruth path/to/file.mgf path/to/file.mztab --out_path out.parquet
    """
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    fire.Fire(
        {
            "get_mgf_psms": get_mgf_psms_df,
            "get_mztab": get_mztab_df,
            "get_groundtruth": get_ground_truth_df,
        }
    )


if __name__ == "__main__":
    main()
